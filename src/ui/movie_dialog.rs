//! Modal dialog for exporting a zoom movie.

use crate::engine::bookmark::{Bookmark, BookmarkView};
use crate::engine::color::ColorSettings;
use crate::engine::escape::FractalKind;
use crate::engine::movie::{MovieError, MovieProgress, MovieSpec, find_ffmpeg, render_movie};
use crate::engine::render::CancelToken;
use crate::engine::viewport::{DEFAULT_SPAN, MaxIter, Viewport, min_pixel_scale};
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
        total_ms: u64,
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

pub struct MovieDialog {
    open: bool,
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
            current: None,
            start: None,
            start_name: None,
            orbit_point: None,
            target: Complex64::new(0.0, 0.0),
            end_magnification: 1e10,
            fps: 30,
            seconds: 20.0,
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
    fn pace_text(&self) -> String {
        let frames = (self.fps as f64 * self.seconds).round().max(2.0);
        let mut text = format!("{} · {frames:.0} frames", format_duration(self.seconds));
        if let Some(start) = &self.start {
            let total = start.view.span / (DEFAULT_SPAN / self.end_magnification);
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
            start_center: start.view.center,
            start_span: start.view.span,
            target: self.target,
            end_span: DEFAULT_SPAN / self.end_magnification,
            fps: self.fps,
            seconds: self.seconds,
            width: self.width,
            height: self.height,
            max_iter: start.max_iter,
            color: start.color,
            output: PathBuf::from(&self.output),
        }
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
            total_ms: 0,
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
                    last_frame_ms,
                }) => {
                    let prev = match self.status {
                        Status::Rendering { total_ms, .. } => total_ms,
                        _ => 0,
                    };
                    self.status = Status::Rendering {
                        done,
                        total,
                        total_ms: prev + last_frame_ms,
                    };
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
    pub fn show(&mut self, ctx: &egui::Context, threads: usize, bookmarks: &[Bookmark]) {
        if !self.open {
            return;
        }
        self.receive();
        let Some(start) = self.start else { return };
        let running = self.job.is_some();

        let modal = egui::Modal::new(egui::Id::unique("movie_dialog")).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.heading("Export zoom movie");
            ui.label(format!(
                "Start ({}): {} at {:+.6} {:+.6}i, zoom {:.2e}",
                self.start_name.as_deref().unwrap_or("current view"),
                start.kind.label(),
                start.view.center.re,
                start.view.center.im,
                start.view.magnification()
            ));
            ui.small(
                "The movie zooms in a straight line from the start view to the \
                 end view. The start also sets the fractal, colors and iterations; \
                 the end only sets where to zoom to and how deep.",
            );
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
                    if ui.button("Render").clicked() {
                        // Look again: ffmpeg may have been installed meanwhile.
                        self.ffmpeg = find_ffmpeg();
                        let spec = self.spec(&start);
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

    fn settings(&mut self, ui: &mut egui::Ui, bookmarks: &[Bookmark]) {
        egui::Grid::new("movie_settings")
            .num_columns(2)
            .show(ui, |ui| {
                let targets: Vec<(&str, Complex64, f64)> = bookmarks
                    .iter()
                    .filter_map(|b| match b.view {
                        BookmarkView::Escape { center, span, .. } => {
                            Some((b.name.as_str(), center, span))
                        }
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
                ui.add(
                    egui::Slider::new(&mut self.end_magnification, 1.0..=floor_mag)
                        .logarithmic(true),
                );
                ui.end_row();
                ui.label("");
                ui.small(format!("precision limit here ≈ {floor_mag:.1e}"));
                ui.end_row();
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
                ui.small(self.pace_text());
                ui.end_row();
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
                total_ms,
            } => {
                let frac = if *total > 0 {
                    *done as f32 / *total as f32
                } else {
                    0.0
                };
                let eta = if *done > 0 {
                    let secs = *total_ms as f64 / *done as f64 * (*total - *done) as f64 / 1000.0;
                    format!(" · about {secs:.0} s left")
                } else {
                    String::new()
                };
                ui.add(egui::ProgressBar::new(frac).text(format!("frame {done}/{total}{eta}")));
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
        assert_eq!(spec.start_center, Complex64::new(0.1, 0.2));
        assert_eq!(spec.start_span, 0.5);
        assert_eq!(spec.kind, FractalKind::Multibrot { power: 3 });
        assert_eq!(d.start_name.as_deref(), Some("deep"));

        d.choose_start(None, d.current.unwrap());
        assert_eq!(
            d.spec(&d.start.unwrap()).start_center,
            Complex64::new(-0.5, 0.0)
        );
        assert_eq!(d.start_name, None);
    }
}
