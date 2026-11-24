//! Modal dialog for exporting a zoom movie.

use crate::engine::bookmark::{Bookmark, BookmarkView, FolderPath, folder_path, folders};
use crate::engine::color::ColorSettings;
use crate::engine::escape::FractalKind;
use crate::engine::keyframe::{Block, KEYFRAME_MAX_PIXELS, Quality, estimated_cost, plan};
use crate::engine::movie::{MovieError, MovieProgress, MovieSpec, find_ffmpeg, render_movie};
use crate::engine::render::CancelToken;
use crate::engine::route::{Stop, outside_previous};
use crate::engine::viewport::{DEFAULT_SPAN, MaxIter, Viewport, min_pixel_scale};
use crate::ui::bookmarks::BookmarksUi;
use eframe::egui;
use num_complex::Complex64;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, channel};

/// Messages from the movie thread.
enum Msg {
    Progress(MovieProgress),
    Finished(Result<PathBuf, MovieError>),
}

struct MovieJob {
    rx: Receiver<Msg>,
    generation: Arc<AtomicU64>,
}

enum Status {
    Idle,
    Rendering {
        done: usize,
        total: usize,
        /// Planned work done and in all (see `MovieProgress::Frame`).
        work: (f64, f64),
        elapsed_ms: u64,
        /// Set while a keyframe renders: (index from 0, count).
        keyframe: Option<(usize, usize)>,
    },
    Encoding,
    Done(PathBuf),
    /// Cancelled; says whether the frames so far were kept.
    Cancelled(String),
    Failed(String),
}

/// The explorer state a movie starts from.
#[derive(Clone, Copy)]
pub struct MovieStart {
    pub kind: FractalKind,
    pub view: Viewport,
    pub max_iter: MaxIter,
    pub color: ColorSettings,
}

const ERROR: egui::Color32 = egui::Color32::from_rgb(240, 90, 90);
const WARNING: egui::Color32 = egui::Color32::from_rgb(230, 170, 60);

/// Top of the duration slider; longer values can be typed in.
const MAX_SLIDER_SECONDS: f64 = 3600.0;

/// "1 h 05 min 03 s" style duration.
fn format_duration(seconds: f64) -> String {
    let s = seconds.round() as u64;
    let (h, m, s) = (s / 3600, s / 60 % 60, s % 60);
    match (h, m) {
        (0, 0) => format!("{s} s"),
        (0, _) => format!("{m} min {s:02} s"),
        _ => format!("{h} h {m:02} min {s:02} s"),
    }
}

/// How fast a zoom of `total_zoom` over `seconds` feels.
fn zoom_per_second(total_zoom: f64, seconds: f64) -> f64 {
    total_zoom.powf(1.0 / seconds)
}

/// Seconds left of a render `elapsed` seconds in with `done` of `total`
/// planned work finished, assuming the rest goes at the same rate.
fn seconds_left(elapsed: f64, done: f64, total: f64) -> Option<f64> {
    (done > 0.0).then(|| (elapsed * (total - done) / done).max(0.0))
}

/// The cost estimate as a sentence, relative to rendering every frame.
fn cost_text(cost: f64) -> String {
    if (0.9..=1.1).contains(&cost) {
        "about the same as Exact".to_string()
    } else if cost < 1.0 {
        let speedup = 1.0 / cost;
        // Whatever prints as 10.0 or more drops the decimal.
        if speedup >= 9.95 {
            format!("≈ {speedup:.0}× faster than Exact")
        } else {
            format!("≈ {speedup:.1}× faster than Exact")
        }
    } else {
        format!("≈ {cost:.1}× slower than Exact — few frames per keyframe")
    }
}

/// What the planner makes of the current settings.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Estimate {
    cost: f64,
    /// Some frames use lower quality (or Exact) to fit in memory.
    lowered: bool,
}

/// The spec fields the estimate depends on.
#[derive(Clone, PartialEq)]
struct EstimateKey {
    kind: FractalKind,
    stops: Vec<Stop>,
    max_iter: MaxIter,
    fps: u32,
    seconds: f64,
    size: (usize, usize),
    quality: Quality,
}

fn compute_estimate(spec: &MovieSpec) -> Estimate {
    let Some(q) = spec.quality.q() else {
        return Estimate {
            cost: 1.0,
            lowered: false,
        };
    };
    let lowered = plan(spec, q, KEYFRAME_MAX_PIXELS).iter().any(|b| match b {
        Block::Keyframe(s) => s.q < q,
        Block::Exact(_) => true,
    });
    Estimate {
        cost: estimated_cost(spec),
        lowered,
    }
}

const NEEDS_TWO_STOPS: &str = "this folder needs at least two bookmarks of the same fractal";
const SWINGS_ACROSS: &str = "this stop isn't inside the one before — the camera will swing across";

const RESOLUTIONS: [(usize, usize); 3] = [(1280, 720), (1920, 1080), (854, 480)];

impl MovieStart {
    /// The movie start a bookmark describes, at the given frame size.
    /// IFS bookmarks can't start a zoom movie.
    pub fn from_bookmark(view: &BookmarkView, width: usize, height: usize) -> Option<Self> {
        match *view {
            BookmarkView::Escape {
                kind,
                center,
                span,
                max_iter,
                color,
            } => Some(Self {
                kind,
                view: Viewport::new(center, span, width, height),
                max_iter,
                color,
            }),
            BookmarkView::Ifs { .. } => None,
        }
    }
}

/// What the movie flies through.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum RouteMode {
    /// From the start view to the end view.
    #[default]
    StartEnd,
    /// Through a bookmark folder's bookmarks, shallowest to deepest.
    Folder(String),
}

pub struct MovieDialog {
    open: bool,
    route: RouteMode,
    /// The explorer's view when the dialog opened.
    current: Option<MovieStart>,
    /// Where the movie starts: `current` or a bookmark.
    start: Option<MovieStart>,
    /// Name of the start bookmark (`None` = current view).
    start_name: Option<String>,
    orbit_point: Option<Complex64>,
    target: Complex64,
    end_magnification: f64,
    fps: u32,
    seconds: f64,
    /// Degrees per second the picture turns.
    rotation: f64,
    /// How frames are rendered: every one exactly, or from keyframes.
    quality: Quality,
    /// The last estimate, with the settings it was made for.
    estimate: Option<(EstimateKey, Estimate)>,
    /// How many times the planner has run (for tests).
    #[cfg(test)]
    estimate_runs: usize,
    width: usize,
    height: usize,
    output: String,
    status: Status,
    job: Option<MovieJob>,
    /// ffmpeg's location, looked up when the dialog opens.
    ffmpeg: Option<PathBuf>,
}

fn default_output() -> String {
    let dir = dirs::video_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."));
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    dir.join(format!("fractal-{stamp}.mp4"))
        .display()
        .to_string()
}

impl Default for MovieDialog {
    fn default() -> Self {
        Self {
            open: false,
            route: RouteMode::StartEnd,
            current: None,
            start: None,
            start_name: None,
            orbit_point: None,
            target: Complex64::new(0.0, 0.0),
            end_magnification: 1e10,
            fps: 30,
            seconds: 20.0,
            rotation: 0.0,
            quality: Quality::Smooth,
            estimate: None,
            #[cfg(test)]
            estimate_runs: 0,
            width: 1280,
            height: 720,
            output: String::new(),
            status: Status::Idle,
            job: None,
            ffmpeg: None,
        }
    }
}

impl MovieDialog {
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Open the dialog starting from the current explorer view.
    pub fn open(&mut self, start: MovieStart, orbit_point: Option<Complex64>) {
        self.open = true;
        self.current = Some(start);
        self.choose_start(None, start);
        self.orbit_point = orbit_point;
        self.target = start.view.center;
        self.output = default_output();
        self.status = Status::Idle;
        self.ffmpeg = find_ffmpeg();
    }

    /// Start the movie from `start`; `name` is its bookmark (`None` = the
    /// current view).
    pub fn choose_start(&mut self, name: Option<String>, start: MovieStart) {
        self.start = Some(start);
        self.start_name = name;
    }

    /// Use a bookmark's center and zoom as the movie's destination.
    pub fn set_target(&mut self, target: Complex64, magnification: f64) {
        self.target = target;
        self.end_magnification = magnification;
    }

    /// Duration, frame count, and zoom speed, under the duration slider.
    fn pace_text(&self, spec: Option<&MovieSpec>) -> String {
        let frames = (self.fps as f64 * self.seconds).round().max(2.0);
        let mut text = format!("{} · {frames:.0} frames", format_duration(self.seconds));
        if let Some(spec) = spec
            && let (Some(first), Some(last)) = (spec.stops.first(), spec.stops.last())
        {
            let total = first.span / last.span;
            if total > 1.0 {
                let rate = zoom_per_second(total, self.seconds);
                text += &format!(" · zooms ×{rate:.3} per second");
            }
        }
        text
    }

    fn spec(&self, start: &MovieStart) -> MovieSpec {
        MovieSpec {
            kind: start.kind,
            stops: vec![
                Stop {
                    center: start.view.center,
                    span: start.view.span,
                    rotation: 0.0,
                },
                Stop {
                    center: self.target,
                    span: DEFAULT_SPAN / self.end_magnification,
                    rotation: self.rotation,
                },
            ],
            fps: self.fps,
            seconds: self.seconds,
            width: self.width,
            height: self.height,
            max_iter: start.max_iter,
            color: start.color,
            output: PathBuf::from(&self.output),
            quality: self.quality,
        }
    }

    /// The movie along a folder's path, with this dialog's frame settings.
    fn folder_spec(&self, path: &FolderPath) -> MovieSpec {
        MovieSpec {
            kind: path.kind,
            stops: path.stops.iter().map(|(_, s)| *s).collect(),
            fps: self.fps,
            seconds: self.seconds,
            width: self.width,
            height: self.height,
            max_iter: path.max_iter,
            color: path.color,
            output: PathBuf::from(&self.output),
            quality: self.quality,
        }
    }

    /// The movie the current settings describe; `None` when a folder has
    /// fewer than two usable stops (or the dialog was never opened).
    fn current_spec(&self, items: &[Bookmark]) -> Option<MovieSpec> {
        match &self.route {
            RouteMode::StartEnd => self.start.map(|start| self.spec(&start)),
            RouteMode::Folder(f) => folder_path(items, f)
                .filter(|p| p.stops.len() >= 2)
                .map(|p| self.folder_spec(&p)),
        }
    }

    /// Go back to Start → End if the chosen folder no longer exists.
    fn refresh_mode(&mut self, items: &[Bookmark]) {
        if let RouteMode::Folder(f) = &self.route
            && !folders(items).contains(f)
        {
            self.route = RouteMode::StartEnd;
        }
    }

    /// Cost and memory fit of the current settings; the planner walks every
    /// frame, so it only reruns when a setting changes, and not while a
    /// slider is being dragged (`dragging`): the last estimate stands until
    /// the drag ends.
    fn estimate(&mut self, spec: &MovieSpec, dragging: bool) -> Estimate {
        if dragging && let Some((_, e)) = &self.estimate {
            return *e;
        }
        let key = EstimateKey {
            kind: spec.kind,
            stops: spec.stops.clone(),
            max_iter: spec.max_iter,
            fps: spec.fps,
            seconds: spec.seconds,
            size: (spec.width, spec.height),
            quality: spec.quality,
        };
        if let Some((k, e)) = &self.estimate
            && *k == key
        {
            return *e;
        }
        #[cfg(test)]
        {
            self.estimate_runs += 1;
        }
        let e = compute_estimate(spec);
        self.estimate = Some((key, e));
        e
    }

    fn launch(&mut self, spec: MovieSpec, ffmpeg: PathBuf, threads: usize, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let generation = Arc::new(AtomicU64::new(0));
        let token = CancelToken::new(generation.clone(), 0);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let ptx = tx.clone();
            let pctx = ctx.clone();
            let result = render_movie(&spec, threads, &ffmpeg, &token, &mut |p| {
                let _ = ptx.send(Msg::Progress(p));
                pctx.request_repaint();
            });
            let _ = tx.send(Msg::Finished(result));
            ctx.request_repaint();
        });
        self.job = Some(MovieJob { rx, generation });
        self.status = Status::Rendering {
            done: 0,
            total: 0,
            work: (0.0, 0.0),
            elapsed_ms: 0,
            keyframe: None,
        };
    }

    fn receive(&mut self) {
        let Some(job) = &self.job else { return };
        let mut finished = false;
        for msg in job.rx.try_iter() {
            match msg {
                Msg::Progress(MovieProgress::Frame {
                    done,
                    total,
                    work,
                    elapsed_ms,
                }) => {
                    self.status = Status::Rendering {
                        done,
                        total,
                        work,
                        elapsed_ms,
                        keyframe: None,
                    };
                }
                Msg::Progress(MovieProgress::Keyframe { done, total: k }) => {
                    if let Status::Rendering { keyframe, .. } = &mut self.status {
                        *keyframe = Some((done, k));
                    }
                }
                Msg::Progress(MovieProgress::Encoding) => self.status = Status::Encoding,
                Msg::Progress(MovieProgress::Done(_)) => {}
                Msg::Finished(Ok(path)) => {
                    self.status = Status::Done(path);
                    finished = true;
                }
                Msg::Finished(Err(e @ MovieError::Cancelled { .. })) => {
                    self.status = Status::Cancelled(e.to_string());
                    finished = true;
                }
                Msg::Finished(Err(e)) => {
                    self.status = Status::Failed(e.to_string());
                    finished = true;
                }
            }
        }
        if finished {
            self.job = None;
        }
    }

    /// Show the dialog (if open). `threads` = cores to render with.
    pub fn show(&mut self, ctx: &egui::Context, threads: usize, bookmarks: &mut BookmarksUi) {
        if !self.open {
            return;
        }
        self.receive();
        self.refresh_mode(&bookmarks.items);
        let Some(start) = self.start else { return };
        let running = self.job.is_some();

        let modal = egui::Modal::new(egui::Id::unique("movie_dialog")).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.heading("Export zoom movie");
            match &self.route {
                RouteMode::StartEnd => {
                    ui.label(format!(
                        "Start ({}): {} at {:+.6} {:+.6}i, zoom {:.2e}",
                        self.start_name.as_deref().unwrap_or("current view"),
                        start.kind.label(),
                        start.view.center.re,
                        start.view.center.im,
                        start.view.magnification()
                    ));
                    ui.small(
                        "The movie zooms from the start view to the end view, \
                         optionally turning (Rotation). The start also sets the \
                         fractal, colors and iterations; the end only sets where to \
                         zoom to and how deep.",
                    );
                }
                RouteMode::Folder(f) => {
                    ui.label(format!("Path: folder {f}"));
                    ui.small(
                        "The movie flies through the folder's bookmarks from \
                         shallowest to deepest, turning on each step as set below. \
                         The first bookmark sets the fractal, colors and iterations.",
                    );
                }
            }
            ui.add_enabled_ui(!running, |ui| self.settings(ui, bookmarks));
            ui.separator();
            if self.ffmpeg.is_none() && !matches!(self.status, Status::Failed(_)) {
                ui.colored_label(WARNING, MovieError::FfmpegMissing.to_string());
            }
            self.status_ui(ui);
            ui.horizontal(|ui| {
                if running {
                    if ui.button("Cancel").clicked()
                        && let Some(job) = &self.job
                    {
                        job.generation.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    let spec = self.current_spec(&bookmarks.items);
                    if ui
                        .add_enabled(spec.is_some(), egui::Button::new("Render"))
                        .clicked()
                        && let Some(spec) = spec
                    {
                        // Look again: ffmpeg may have been installed meanwhile.
                        self.ffmpeg = find_ffmpeg();
                        let checked = spec
                            .validate()
                            .and_then(|()| self.ffmpeg.clone().ok_or(MovieError::FfmpegMissing));
                        match checked {
                            Ok(ffmpeg) => self.launch(spec, ffmpeg, threads, ctx),
                            Err(e) => self.status = Status::Failed(e.to_string()),
                        }
                    }
                    if ui.button("Close").clicked() {
                        self.open = false;
                    }
                }
            });
        });
        if modal.should_close() && !running {
            self.open = false;
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui, bookmarks: &mut BookmarksUi) {
        egui::Grid::new("movie_settings")
            .num_columns(2)
            .show(ui, |ui| {
                self.route_picker(ui, &bookmarks.items);
                if let RouteMode::Folder(f) = &self.route {
                    let f = f.clone();
                    self.folder_rows(ui, bookmarks, &f);
                } else {
                    self.start_end_rows(ui, &bookmarks.items);
                }
                ui.label("Frame rate");
                egui::ComboBox::from_id_salt("fps")
                    .selected_text(format!("{} fps", self.fps))
                    .show_ui(ui, |ui| {
                        for f in [24, 30, 60] {
                            ui.selectable_value(&mut self.fps, f, format!("{f} fps"));
                        }
                    });
                ui.end_row();
                ui.label("Duration");
                ui.add(
                    egui::Slider::new(&mut self.seconds, 1.0..=MAX_SLIDER_SECONDS)
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Never)
                        .suffix(" s"),
                );
                ui.end_row();
                ui.label("");
                ui.small(self.pace_text(self.current_spec(&bookmarks.items).as_ref()));
                ui.end_row();
                if self.route == RouteMode::StartEnd {
                    ui.label("Rotation");
                    ui.add(egui::Slider::new(&mut self.rotation, -30.0..=30.0).suffix(" °/s"));
                    ui.end_row();
                }
                ui.label("Quality");
                egui::ComboBox::from_id_salt("movie_quality")
                    .selected_text(self.quality.label())
                    .show_ui(ui, |ui| {
                        for q in Quality::ALL {
                            ui.selectable_value(&mut self.quality, q, q.label());
                        }
                    });
                ui.end_row();
                if self.quality != Quality::Exact
                    && let Some(spec) = self.current_spec(&bookmarks.items)
                {
                    let estimate = self.estimate(&spec, ui.ctx().dragged_id().is_some());
                    ui.label("");
                    ui.vertical(|ui| {
                        ui.small(cost_text(estimate.cost));
                        if estimate.lowered {
                            ui.small("some frames use Fast quality to fit in memory");
                        }
                    });
                    ui.end_row();
                }
                ui.label("Resolution");
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("res")
                        .selected_text(format!("{}×{}", self.width, self.height))
                        .show_ui(ui, |ui| {
                            for (w, h) in RESOLUTIONS {
                                if ui
                                    .selectable_label(
                                        (self.width, self.height) == (w, h),
                                        format!("{w}×{h}"),
                                    )
                                    .clicked()
                                {
                                    (self.width, self.height) = (w, h);
                                }
                            }
                        });
                    ui.add(egui::DragValue::new(&mut self.width).range(16..=7680));
                    ui.label("×");
                    ui.add(egui::DragValue::new(&mut self.height).range(16..=4320));
                });
                ui.end_row();
                ui.label("Save to");
                ui.text_edit_singleline(&mut self.output);
                ui.end_row();
            });
    }

    /// Start, end and their helpers (Start → End mode).
    fn start_end_rows(&mut self, ui: &mut egui::Ui, bookmarks: &[Bookmark]) {
        let targets: Vec<(&str, Complex64, f64)> = bookmarks
            .iter()
            .filter_map(|b| match b.view {
                BookmarkView::Escape { center, span, .. } => Some((b.name.as_str(), center, span)),
                BookmarkView::Ifs { .. } => None,
            })
            .collect();
        self.start_picker(ui, bookmarks);
        if !targets.is_empty() {
            ui.label("End from bookmark");
            egui::ComboBox::from_id_salt("movie_bookmark")
                .selected_text("choose…")
                .show_ui(ui, |ui| {
                    for (name, center, span) in &targets {
                        if ui.selectable_label(false, *name).clicked() {
                            self.set_target(*center, DEFAULT_SPAN / span);
                        }
                    }
                });
            ui.end_row();
        }
        ui.label("End center");
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.target.re)
                    .speed(1e-6)
                    .max_decimals(15),
            );
            ui.add(
                egui::DragValue::new(&mut self.target.im)
                    .speed(1e-6)
                    .max_decimals(15),
            );
            ui.label("i");
        });
        ui.end_row();
        if let Some(p) = self.orbit_point {
            ui.label("");
            if ui.button("Center end on orbit point").clicked() {
                self.target = p;
            }
            ui.end_row();
        }
        ui.label("End zoom");
        let floor_mag = DEFAULT_SPAN / (min_pixel_scale(self.target) * self.height as f64);
        ui.add(egui::Slider::new(&mut self.end_magnification, 1.0..=floor_mag).logarithmic(true));
        ui.end_row();
        ui.label("");
        ui.small(format!("precision limit here ≈ {floor_mag:.1e}"));
        ui.end_row();
    }

    /// "Route" row: Start → End, or a bookmark folder.
    fn route_picker(&mut self, ui: &mut egui::Ui, items: &[Bookmark]) {
        let names = folders(items);
        if names.is_empty() {
            return;
        }
        ui.label("Route");
        let text = match &self.route {
            RouteMode::StartEnd => "Start → End".to_string(),
            RouteMode::Folder(f) => format!("📁 {f}"),
        };
        egui::ComboBox::from_id_salt("movie_route")
            .selected_text(text)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.route, RouteMode::StartEnd, "Start → End");
                for f in names {
                    let label = format!("📁 {f}");
                    ui.selectable_value(&mut self.route, RouteMode::Folder(f), label);
                }
            });
        ui.end_row();
    }

    /// The folder's stops in order, with a rotation slider on each step.
    fn folder_rows(&mut self, ui: &mut egui::Ui, bookmarks: &mut BookmarksUi, folder: &str) {
        let Some(path) = folder_path(&bookmarks.items, folder) else {
            ui.label("");
            ui.colored_label(WARNING, NEEDS_TWO_STOPS);
            ui.end_row();
            return;
        };
        let mut prev: Option<Stop> = None;
        for (k, (idx, stop)) in path.stops.iter().enumerate() {
            if let Some(p) = prev {
                ui.label("  turn");
                let mut rate = bookmarks.items[*idx].rotation;
                let resp = ui.add(egui::Slider::new(&mut rate, -30.0..=30.0).suffix(" °/s"));
                if resp.changed() {
                    // Shown at once; written to disk when the edit ends.
                    bookmarks.items[*idx].rotation = rate;
                }
                if resp.drag_stopped() || (resp.changed() && !resp.dragged()) {
                    bookmarks.set_rotation(*idx, rate);
                }
                ui.end_row();
                if outside_previous(&p, stop, self.width, self.height) {
                    ui.label("");
                    ui.colored_label(WARNING, SWINGS_ACROSS);
                    ui.end_row();
                }
            }
            ui.label(format!("Stop {}", k + 1));
            ui.label(format!(
                "{}  ×{:.1e}",
                bookmarks.items[*idx].name,
                DEFAULT_SPAN / stop.span
            ));
            ui.end_row();
            prev = Some(*stop);
        }
        if path.skipped > 0 {
            ui.label("");
            ui.small(format!(
                "{} bookmark{} skipped (different fractal or IFS)",
                path.skipped,
                if path.skipped == 1 { "" } else { "s" }
            ));
            ui.end_row();
        }
        if path.stops.len() < 2 {
            ui.label("");
            ui.colored_label(WARNING, NEEDS_TWO_STOPS);
            ui.end_row();
        }
    }

    /// "Start" row: the current view or any escape-time bookmark.
    fn start_picker(&mut self, ui: &mut egui::Ui, bookmarks: &[Bookmark]) {
        let Some(current) = self.current else { return };
        let (w, h) = (current.view.width, current.view.height);
        let starts: Vec<(&str, MovieStart)> = bookmarks
            .iter()
            .filter_map(|b| Some((b.name.as_str(), MovieStart::from_bookmark(&b.view, w, h)?)))
            .collect();
        if starts.is_empty() {
            return;
        }
        ui.label("Start from");
        let selected = self.start_name.clone();
        egui::ComboBox::from_id_salt("movie_start")
            .selected_text(selected.as_deref().unwrap_or("current view"))
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(selected.is_none(), "current view")
                    .clicked()
                {
                    self.choose_start(None, current);
                }
                for (name, start) in starts {
                    let is_sel = selected.as_deref() == Some(name);
                    if ui.selectable_label(is_sel, name).clicked() {
                        self.choose_start(Some(name.to_string()), start);
                    }
                }
            });
        ui.end_row();
    }

    fn status_ui(&self, ui: &mut egui::Ui) {
        match &self.status {
            Status::Idle => {}
            Status::Rendering {
                done,
                total,
                work: (work_done, work_total),
                elapsed_ms,
                keyframe,
            } => {
                let frac = if *total > 0 {
                    *done as f32 / *total as f32
                } else {
                    0.0
                };
                let eta = match seconds_left(*elapsed_ms as f64 / 1000.0, *work_done, *work_total) {
                    Some(secs) => format!(" · about {} left", format_duration(secs)),
                    None => String::new(),
                };
                ui.add(egui::ProgressBar::new(frac).text(format!("frame {done}/{total}{eta}")));
                if let Some((k, of)) = keyframe {
                    ui.label(format!("rendering keyframe {}/{of}", k + 1));
                }
            }
            Status::Encoding => {
                ui.add(egui::ProgressBar::new(1.0).text("finishing encoding…"));
            }
            Status::Done(path) => {
                ui.label(format!("Saved {}", path.display()));
            }
            Status::Cancelled(msg) => {
                ui.label(msg);
            }
            Status::Failed(msg) => {
                ui.colored_label(ERROR, msg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::color::PaletteId;

    fn current() -> MovieStart {
        MovieStart {
            kind: FractalKind::Mandelbrot,
            view: Viewport::new(Complex64::new(-0.5, 0.0), 3.0, 800, 600),
            max_iter: MaxIter::Auto,
            color: ColorSettings::default(),
        }
    }

    fn bookmarked() -> BookmarkView {
        BookmarkView::Escape {
            kind: FractalKind::Multibrot { power: 3 },
            center: Complex64::new(0.1, 0.2),
            span: 0.5,
            max_iter: MaxIter::Fixed(900),
            color: ColorSettings {
                palette: PaletteId::ALL[1],
                offset: 0.3,
                scale: 0.1,
            },
        }
    }

    #[test]
    fn quality_defaults_to_smooth_and_reaches_the_spec() {
        let mut d = MovieDialog::default();
        assert_eq!(d.spec(&current()).quality, Quality::Smooth);
        d.quality = Quality::Exact;
        assert_eq!(d.spec(&current()).quality, Quality::Exact);
    }

    #[test]
    fn cost_reads_naturally() {
        assert_eq!(cost_text(0.025), "≈ 40× faster than Exact");
        assert_eq!(
            cost_text(1.6),
            "≈ 1.6× slower than Exact — few frames per keyframe"
        );
        assert_eq!(cost_text(1.0), "about the same as Exact");
        assert_eq!(cost_text(0.5), "≈ 2.0× faster than Exact");
        // 1 / 0.1003 ≈ 9.97 rounds to 10: no "10.0×".
        assert_eq!(cost_text(0.1003), "≈ 10× faster than Exact");
        assert_eq!(cost_text(0.1007), "≈ 9.9× faster than Exact");
    }

    #[test]
    fn eta_scales_elapsed_time_by_the_work_left() {
        assert_eq!(seconds_left(5.0, 0.0, 100.0), None);
        assert_eq!(seconds_left(10.0, 25.0, 100.0), Some(30.0));
        assert_eq!(seconds_left(10.0, 100.0, 100.0), Some(0.0));
        // A keyframe of 1000 samples' work then one frame of 10: 20 s in,
        // the 99 frames left are about as much work again, not 99 × 20 s.
        let left = seconds_left(20.0, 1010.0, 1000.0 + 100.0 * 10.0).unwrap();
        assert!((left - 19.6).abs() < 0.1, "{left}");
    }

    #[test]
    fn estimate_is_cached_until_a_setting_changes() {
        let mut d = MovieDialog {
            seconds: 2.0,
            fps: 24,
            width: 64,
            height: 48,
            ..MovieDialog::default()
        };
        let start = current();
        let a = d.estimate(&d.spec(&start), false);
        assert_eq!(d.estimate_runs, 1);
        assert_eq!(d.estimate(&d.spec(&start), false), a);
        assert_eq!(d.estimate_runs, 1);
        d.rotation = 10.0;
        d.estimate(&d.spec(&start), false);
        assert_eq!(d.estimate_runs, 2);
    }

    #[test]
    fn rotation_reaches_the_spec() {
        let mut d = MovieDialog::default();
        assert_eq!(d.spec(&current()).stops[1].rotation, 0.0);
        d.rotation = -6.0;
        assert_eq!(d.spec(&current()).stops[1].rotation, -6.0);
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(format_duration(42.0), "42 s");
        assert_eq!(format_duration(125.0), "2 min 05 s");
        assert_eq!(format_duration(3903.0), "1 h 05 min 03 s");
    }

    #[test]
    fn zoom_speed_halves_in_log_when_duration_doubles() {
        let fast = zoom_per_second(1e10, 60.0);
        let slow = zoom_per_second(1e10, 120.0);
        assert!((slow * slow - fast).abs() < 1e-12);
        assert!((fast.powf(60.0) / 1e10 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn start_from_bookmark_takes_its_whole_view() {
        let s = MovieStart::from_bookmark(&bookmarked(), 800, 600).unwrap();
        assert_eq!(s.kind, FractalKind::Multibrot { power: 3 });
        assert_eq!(s.view.center, Complex64::new(0.1, 0.2));
        assert_eq!(s.view.span, 0.5);
        assert_eq!((s.view.width, s.view.height), (800, 600));
        assert_eq!(s.max_iter, MaxIter::Fixed(900));
        assert_eq!(s.color.offset, 0.3);
    }

    #[test]
    fn ifs_bookmark_cannot_start_a_movie() {
        let ifs = BookmarkView::Ifs {
            system: crate::engine::ifs::IfsSystem::barnsley_fern(),
            center: Complex64::new(0.0, 5.0),
            span: 10.5,
            points: 1000,
            color: ColorSettings::default(),
        };
        assert!(MovieStart::from_bookmark(&ifs, 800, 600).is_none());
    }

    #[test]
    fn chosen_start_drives_the_spec_and_can_go_back_to_current() {
        let mut d = MovieDialog::default();
        d.open(current(), None);
        d.choose_start(
            Some("deep".into()),
            MovieStart::from_bookmark(&bookmarked(), 800, 600).unwrap(),
        );
        let spec = d.spec(&d.start.unwrap());
        assert_eq!(spec.stops[0].center, Complex64::new(0.1, 0.2));
        assert_eq!(spec.stops[0].span, 0.5);
        assert_eq!(spec.kind, FractalKind::Multibrot { power: 3 });
        assert_eq!(d.start_name.as_deref(), Some("deep"));

        d.choose_start(None, d.current.unwrap());
        assert_eq!(
            d.spec(&d.start.unwrap()).stops[0].center,
            Complex64::new(-0.5, 0.0)
        );
        assert_eq!(d.start_name, None);
    }

    fn filed(name: &str, folder: &str, view: BookmarkView, rotation: f64) -> Bookmark {
        Bookmark {
            name: name.into(),
            view,
            folder: Some(folder.into()),
            rotation,
        }
    }

    fn mandel(re: f64, im: f64, span: f64, offset: f32) -> BookmarkView {
        BookmarkView::Escape {
            kind: FractalKind::Mandelbrot,
            center: Complex64::new(re, im),
            span,
            max_iter: MaxIter::Fixed(500),
            color: ColorSettings {
                offset,
                ..ColorSettings::default()
            },
        }
    }

    fn dive() -> Vec<Bookmark> {
        vec![
            filed("deep", "dive", mandel(-0.745, 0.105, 3e-5, 0.75), -20.0),
            filed("top", "dive", mandel(-0.5, 0.0, 3.0, 0.25), 5.0),
            filed("mid", "dive", mandel(-0.75, 0.1, 0.03, 0.5), 10.0),
        ]
    }

    #[test]
    fn folder_spec_uses_path_stops_and_first_colors() {
        let mut d = MovieDialog::default();
        d.open(current(), None);
        let items = dive();
        d.route = RouteMode::Folder("dive".into());
        let spec = d.current_spec(&items).unwrap();
        let spans: Vec<f64> = spec.stops.iter().map(|s| s.span).collect();
        assert_eq!(spans, [3.0, 0.03, 3e-5]);
        assert_eq!(spec.stops[2].rotation, -20.0);
        assert_eq!(spec.color.offset, 0.25);
        assert_eq!(spec.max_iter, MaxIter::Fixed(500));
    }

    #[test]
    fn folder_with_one_usable_stop_cannot_render() {
        let mut d = MovieDialog::default();
        d.open(current(), None);
        let ifs = BookmarkView::Ifs {
            system: crate::engine::ifs::IfsSystem::barnsley_fern(),
            center: Complex64::new(0.0, 5.0),
            span: 10.5,
            points: 1000,
            color: ColorSettings::default(),
        };
        let items = vec![
            filed("top", "dive", mandel(-0.5, 0.0, 3.0, 0.0), 0.0),
            filed("fern", "dive", ifs, 0.0),
        ];
        d.route = RouteMode::Folder("dive".into());
        assert!(d.current_spec(&items).is_none());
    }

    #[test]
    fn vanished_folder_falls_back_to_start_end() {
        let mut d = MovieDialog::default();
        d.open(current(), None);
        d.route = RouteMode::Folder("dive".into());
        d.refresh_mode(&dive());
        assert_eq!(d.route, RouteMode::Folder("dive".into()));
        d.refresh_mode(&[]);
        assert_eq!(d.route, RouteMode::StartEnd);
        assert!(d.current_spec(&[]).is_some());
    }

    #[test]
    fn estimate_waits_while_a_slider_is_dragged() {
        let mut d = MovieDialog {
            seconds: 2.0,
            fps: 24,
            width: 64,
            height: 48,
            ..MovieDialog::default()
        };
        let start = current();
        let a = d.estimate(&d.spec(&start), false);
        d.rotation = 10.0;
        assert_eq!(d.estimate(&d.spec(&start), true), a);
        assert_eq!(d.estimate_runs, 1);
        d.estimate(&d.spec(&start), false);
        assert_eq!(d.estimate_runs, 2);
    }
}
