//! Top-level application state and layout.

use crate::engine::color::{ColorSettings, colorize};
use crate::engine::escape::{FractalKind, Sample};
use crate::engine::jobs::{RenderRequest, RenderSlot};
use crate::engine::render::F64Sampler;
use crate::engine::viewport::{MaxIter, Viewport};
use crate::ui::panel::explore_panel;
use crate::ui::view::show_view;
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

    fn status_bar(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
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

impl eframe::App for FractalApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.receive(&ctx);

        egui::Panel::left("controls")
            .default_size(260.0)
            .show(ui, |ui| {
                let ch = explore_panel(
                    ui,
                    &mut self.explore,
                    self.main_slot.threads(),
                    self.max_threads,
                );
                if let Some(n) = ch.threads {
                    self.main_slot.set_threads(n);
                    self.needs_render = true;
                }
                self.needs_render |= ch.render;
                if ch.color {
                    self.recolor(&ctx);
                }
            });
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));

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
            });

        if self.needs_render {
            self.submit();
        }
    }
}
