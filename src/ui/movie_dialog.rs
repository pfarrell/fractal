//! Modal dialog for exporting a zoom movie.

use crate::engine::color::ColorSettings;
use crate::engine::escape::FractalKind;
use crate::engine::movie::{MovieError, MovieProgress, MovieSpec, render_movie};
use crate::engine::render::CancelToken;
use crate::engine::viewport::{DEFAULT_SPAN, MaxIter, Viewport, min_pixel_scale};
use eframe::egui;
use num_complex::Complex64;
use std::path::{Path, PathBuf};
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

const RESOLUTIONS: [(usize, usize); 3] = [(1280, 720), (1920, 1080), (854, 480)];

pub struct MovieDialog {
    open: bool,
    start: Option<MovieStart>,
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
            start: None,
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
        self.start = Some(start);
        self.orbit_point = orbit_point;
        self.target = start.view.center;
        self.output = default_output();
        self.status = Status::Idle;
    }

    /// Use a bookmark's center and zoom as the movie's destination.
    pub fn set_target(&mut self, target: Complex64, magnification: f64) {
        self.target = target;
        self.end_magnification = magnification;
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

    fn launch(&mut self, spec: MovieSpec, threads: usize, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let generation = Arc::new(AtomicU64::new(0));
        let token = CancelToken::new(generation.clone(), 0);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let ptx = tx.clone();
            let pctx = ctx.clone();
            let result = render_movie(&spec, threads, Path::new("ffmpeg"), &token, &mut |p| {
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
    pub fn show(&mut self, ctx: &egui::Context, threads: usize) {
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
                "From: {} at {:+.6} {:+.6}i, zoom {:.2e}",
                start.kind.label(),
                start.view.center.re,
                start.view.center.im,
                start.view.magnification()
            ));
            ui.add_enabled_ui(!running, |ui| self.settings(ui));
            ui.separator();
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
                        let spec = self.spec(&start);
                        match spec.validate() {
                            Ok(()) => self.launch(spec, threads, ctx),
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

    fn settings(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("movie_settings")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Zoom toward");
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
                    if ui.button("Use orbit point").clicked() {
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
                ui.add(egui::Slider::new(&mut self.seconds, 1.0..=120.0).suffix(" s"));
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
                ui.add(egui::ProgressBar::new(1.0).text("encoding with ffmpeg…"));
            }
            Status::Done(path) => {
                ui.label(format!("Saved {}", path.display()));
            }
            Status::Failed(msg) => {
                ui.colored_label(egui::Color32::from_rgb(240, 90, 90), msg);
            }
        }
    }
}
