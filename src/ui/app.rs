//! Top-level application state and layout.

use crate::engine::bookmark::BookmarkView;
use crate::engine::color::{ColorSettings, colorize};
use crate::engine::escape::{FractalKind, Sample, iterate, orbit};
use crate::engine::jobs::{RenderRequest, RenderSlot};
use crate::engine::render::F64Sampler;
use crate::engine::viewport::{MaxIter, Viewport};
use crate::ui::bookmarks::BookmarksUi;
use crate::ui::ifs_table::IfsState;
use crate::ui::julia::{JuliaAction, JuliaPanel};
use crate::ui::movie_dialog::{MovieDialog, MovieStart};
use crate::ui::panel::{explore_panel, orbit_panel, palette_combo, threads_slider};
use crate::ui::view::{draw_orbit, show_view};
use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use num_complex::Complex64;
use std::sync::Arc;
use std::time::Duration;

/// Which kind of fractal the window is showing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Escape,
    Ifs,
}

/// Everything about the escape-time explorer's current picture.
pub struct ExploreState {
    pub kind: FractalKind,
    pub view: Viewport,
    pub max_iter: MaxIter,
    pub color: ColorSettings,
    /// The most recent render and the view it was rendered for.
    pub samples: Vec<Sample>,
    pub samples_view: Viewport,
    /// (time, threads, pixels) of the last full-resolution pass.
    pub last_render: Option<(Duration, usize, usize)>,
}

impl Default for ExploreState {
    fn default() -> Self {
        let view = Viewport::default_for(0, 0);
        Self {
            kind: FractalKind::Mandelbrot,
            view,
            max_iter: MaxIter::Auto,
            color: ColorSettings::default(),
            samples: Vec::new(),
            samples_view: view,
            last_render: None,
        }
    }
}

/// The orbit inspector's current point.
pub struct OrbitState {
    pub point: Complex64,
    pub path: Vec<Complex64>,
    pub outcome: Sample,
}

/// Orbits drawn on screen are capped at this many steps.
const ORBIT_MAX_STEPS: u32 = 1000;

impl OrbitState {
    pub fn compute(kind: &FractalKind, point: Complex64, max_iter: u32) -> Self {
        let n = max_iter.min(ORBIT_MAX_STEPS);
        Self {
            point,
            path: orbit(kind, point, n),
            outcome: iterate(kind, point, n),
        }
    }
}

/// Upload colored pixels into `texture` (creating it if needed).
pub fn upload(
    ctx: &egui::Context,
    texture: &mut Option<TextureHandle>,
    name: &str,
    size: [usize; 2],
    pixels: &[[u8; 4]],
) {
    let image = ColorImage::from_rgba_unmultiplied(size, pixels.as_flattened());
    match texture {
        Some(t) => t.set(image, TextureOptions::LINEAR),
        None => *texture = Some(ctx.load_texture(name, image, TextureOptions::LINEAR)),
    }
}

/// A render slot whose results trigger a repaint.
pub fn repainting_slot(ctx: &egui::Context, threads: usize) -> RenderSlot {
    let ctx = ctx.clone();
    RenderSlot::new(threads, Arc::new(move || ctx.request_repaint()))
}

/// The eframe application.
pub struct FractalApp {
    pub mode: Mode,
    pub explore: ExploreState,
    main_slot: RenderSlot,
    max_threads: usize,
    texture: Option<TextureHandle>,
    pixels: Vec<[u8; 4]>,
    needs_render: bool,
    cursor: Option<Complex64>,
    julia: JuliaPanel,
    orbit: Option<OrbitState>,
    ifs: IfsState,
    movie: MovieDialog,
    bookmarks: BookmarksUi,
}

impl FractalApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let max_threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        Self {
            mode: Mode::Escape,
            explore: ExploreState::default(),
            main_slot: repainting_slot(&cc.egui_ctx, max_threads),
            max_threads,
            texture: None,
            pixels: Vec::new(),
            needs_render: true,
            cursor: None,
            julia: JuliaPanel::new(&cc.egui_ctx, max_threads),
            orbit: None,
            ifs: IfsState::new(&cc.egui_ctx, max_threads),
            movie: MovieDialog::default(),
            bookmarks: BookmarksUi::default(),
        }
    }

    fn submit(&mut self) {
        let s = &self.explore;
        if s.view.is_empty() {
            return;
        }
        let max_iter = s.max_iter.resolve(s.view.magnification());
        self.main_slot.submit(RenderRequest {
            sampler: Arc::new(F64Sampler {
                kind: s.kind,
                max_iter,
            }),
            view: s.view,
            progressive: true,
        });
        self.needs_render = false;
    }

    fn recolor(&mut self, ctx: &egui::Context) {
        let s = &self.explore;
        if s.samples.is_empty() {
            return;
        }
        self.pixels.resize(s.samples.len(), [0; 4]);
        colorize(&s.samples, &s.color, &mut self.pixels);
        let size = [s.samples_view.width, s.samples_view.height];
        upload(ctx, &mut self.texture, "main", size, &self.pixels);
    }

    fn receive(&mut self, ctx: &egui::Context) {
        if let Some(out) = self.main_slot.poll() {
            if out.final_pass {
                self.explore.last_render =
                    Some((out.elapsed, out.threads, out.view.width * out.view.height));
            }
            self.explore.samples = out.samples;
            self.explore.samples_view = out.view;
            self.recolor(ctx);
        }
    }

    /// Recompute the orbit after the fractal or its parameters change.
    fn refresh_orbit(&mut self) {
        if let Some(o) = &self.orbit {
            let max_iter = self
                .explore
                .max_iter
                .resolve(self.explore.view.magnification());
            self.orbit = Some(OrbitState::compute(&self.explore.kind, o.point, max_iter));
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Bookmarks").clicked() {
                self.bookmarks.open = !self.bookmarks.open;
            }
            ui.separator();
            match self.cursor {
                Some(z) => ui.monospace(format!("{:+.12} {:+.12}i", z.re, z.im)),
                None => ui.monospace("—"),
            };
            ui.separator();
            ui.label(format!("zoom {:.2e}", self.explore.view.magnification()));
            if self.explore.view.at_precision_floor() {
                ui.separator();
                ui.colored_label(Color32::from_rgb(240, 170, 40), "precision limit");
            }
        });
    }
}

impl FractalApp {
    fn escape_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.receive(ctx);
        let movie_was_open = self.movie.is_open();
        self.movie
            .show(ctx, self.max_threads, &self.bookmarks.items);
        if movie_was_open && !self.movie.is_open() {
            self.needs_render = true; // the dialog cancelled the live renders
        }

        egui::Panel::left("controls")
            .default_size(260.0)
            .show(ui, |ui| {
                self.mode_switch(ui);
                let ch = explore_panel(
                    ui,
                    &mut self.explore,
                    self.main_slot.threads(),
                    self.max_threads,
                );
                if let Some(n) = ch.threads {
                    self.main_slot.set_threads(n);
                    self.julia.set_main_threads(n);
                    self.ifs.set_threads(n);
                    self.needs_render = true;
                }
                if ch.render {
                    self.needs_render = true;
                    self.refresh_orbit();
                }
                if ch.color {
                    self.recolor(ctx);
                    self.julia.invalidate();
                }
                if ch.open_movie {
                    let e = &self.explore;
                    let start = MovieStart {
                        kind: e.kind,
                        view: e.view,
                        max_iter: e.max_iter,
                        color: e.color,
                    };
                    self.movie.open(start, self.orbit.as_ref().map(|o| o.point));
                    self.main_slot.cancel();
                    self.julia.cancel();
                }
                if self.explore.kind == FractalKind::Mandelbrot {
                    ui.checkbox(&mut self.julia.visible, "Show Julia panel");
                }
                if let Some(o) = &self.orbit {
                    ui.separator();
                    if orbit_panel(ui, o) {
                        self.orbit = None;
                    }
                }
            });
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));

        let julia_on = self.julia.visible && self.explore.kind == FractalKind::Mandelbrot;
        if julia_on {
            let color = self.explore.color;
            let action = egui::Panel::right("julia")
                .exact_size(340.0)
                .show(ui, |ui| self.julia.show(ui, &color))
                .inner;
            if let JuliaAction::OpenInMain(c) = action {
                self.explore.kind = FractalKind::Julia { c };
                self.explore.view = self
                    .explore
                    .kind
                    .default_view(self.explore.view.width, self.explore.view.height);
                self.orbit = None;
                self.julia.cancel();
                self.needs_render = true;
            }
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let image = self
                    .texture
                    .as_ref()
                    .map(|t| (t, &self.explore.samples_view));
                let resp = show_view(ui, image, &mut self.explore.view);
                let (w, h) = resp.mapping.pixel_size();
                let v = &mut self.explore.view;
                if (v.width, v.height) != (w, h) {
                    v.width = w;
                    v.height = h;
                    self.needs_render = true;
                }
                self.needs_render |= resp.view_changed;
                self.cursor = resp.hovered;
                if julia_on {
                    self.julia.hover(resp.hovered);
                }
                if let Some(p) = resp.clicked {
                    let max_iter = self
                        .explore
                        .max_iter
                        .resolve(self.explore.view.magnification());
                    self.orbit = Some(OrbitState::compute(&self.explore.kind, p, max_iter));
                    if julia_on {
                        self.julia.pinned = Some(p);
                    }
                }
                if let Some(o) = &self.orbit {
                    draw_orbit(ui, &resp.mapping, &self.explore.view, &o.path);
                }
            });

        if self.needs_render && self.mode == Mode::Escape && !self.movie.is_open() {
            self.submit();
        }
    }
}

impl FractalApp {
    /// "Escape-time | IFS" switch at the top of the left panel.
    fn mode_switch(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let before = self.mode;
            ui.selectable_value(&mut self.mode, Mode::Escape, "Escape-time");
            ui.selectable_value(&mut self.mode, Mode::Ifs, "IFS");
            if self.mode != before {
                match self.mode {
                    Mode::Escape => {
                        self.ifs.cancel();
                        self.needs_render = true;
                    }
                    Mode::Ifs => {
                        self.main_slot.cancel();
                        self.julia.cancel();
                        self.ifs.mark_dirty();
                    }
                }
            }
        });
        ui.separator();
    }

    fn ifs_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::Panel::left("controls")
            .default_size(260.0)
            .show(ui, |ui| {
                self.mode_switch(ui);
                self.ifs.controls(ui);
                ui.separator();
                if palette_combo(ui, &mut self.explore.color.palette) {
                    self.ifs.recolor(ctx, &self.explore.color);
                }
                if let Some(n) = threads_slider(ui, self.main_slot.threads(), self.max_threads) {
                    self.main_slot.set_threads(n);
                    self.julia.set_main_threads(n);
                    self.ifs.set_threads(n);
                }
            });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Bookmarks").clicked() {
                    self.bookmarks.open = !self.bookmarks.open;
                }
                ui.separator();
                match self.cursor {
                    Some(z) => ui.monospace(format!("x {:+.9}  y {:+.9}", z.re, z.im)),
                    None => ui.monospace("—"),
                }
            });
        });
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                self.cursor = self.ifs.show_view(ui, &self.explore.color);
            });
    }
}

impl FractalApp {
    /// The picture on screen, as a bookmark.
    fn current_bookmark_view(&self) -> BookmarkView {
        let e = &self.explore;
        match self.mode {
            Mode::Escape => BookmarkView::Escape {
                kind: e.kind,
                center: e.view.center,
                span: e.view.span,
                max_iter: e.max_iter,
                color: e.color,
            },
            Mode::Ifs => BookmarkView::Ifs {
                system: self.ifs.system.clone(),
                center: self.ifs.view.center,
                span: self.ifs.view.span,
                points: self.ifs.points,
                color: e.color,
            },
        }
    }

    fn go_to(&mut self, view: BookmarkView) {
        match view {
            BookmarkView::Escape {
                kind,
                center,
                span,
                max_iter,
                color,
            } => {
                self.mode = Mode::Escape;
                self.ifs.cancel();
                let e = &mut self.explore;
                e.kind = kind;
                e.view.center = center;
                e.view.span = span;
                e.max_iter = max_iter;
                e.color = color;
                self.orbit = None;
                self.needs_render = true;
            }
            BookmarkView::Ifs {
                system,
                center,
                span,
                points,
                color,
            } => {
                self.mode = Mode::Ifs;
                self.main_slot.cancel();
                self.julia.cancel();
                self.ifs.system = system;
                self.ifs.set_view(center, span);
                self.ifs.points = points;
                self.explore.color = color;
                self.ifs.mark_dirty();
            }
        }
    }
}

impl eframe::App for FractalApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let current = self.current_bookmark_view();
        if let Some(view) = self.bookmarks.show(&ctx, || current) {
            self.go_to(view);
        }
        match self.mode {
            Mode::Escape => self.escape_ui(ui, &ctx),
            Mode::Ifs => self.ifs_ui(ui, &ctx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_of_minus_one_is_a_two_cycle() {
        let o = OrbitState::compute(&FractalKind::Mandelbrot, Complex64::new(-1.0, 0.0), 50_000);
        assert_eq!(o.outcome, Sample::Inside);
        assert_eq!(o.path.len(), ORBIT_MAX_STEPS as usize + 1);
        assert_eq!(o.path[1], Complex64::new(-1.0, 0.0));
        assert_eq!(o.path[2], Complex64::new(0.0, 0.0));
    }
}
