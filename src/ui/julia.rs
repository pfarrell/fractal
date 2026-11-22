//! The Julia panel: shows the Julia set for the point c under the cursor
//! on the Mandelbrot set.
//!
//! While the cursor moves, it renders at quarter resolution so it can keep
//! up; once the cursor rests for [`SETTLE`], it renders at full resolution.
//! It has its own [`RenderSlot`] (and thread pool), so it never cancels
//! the main view's work.

use crate::engine::color::{ColorSettings, colorize};
use crate::engine::escape::FractalKind;
use crate::engine::jobs::{RenderRequest, RenderSlot};
use crate::engine::render::F64Sampler;
use crate::engine::viewport::Viewport;
use crate::ui::app::{repainting_slot, upload};
use crate::ui::notes::{NOTE_JULIA_LINK, note};
use eframe::egui::{self, TextureHandle};
use num_complex::Complex64;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long the cursor must rest before rendering at full resolution.
const SETTLE: Duration = Duration::from_millis(150);
/// Side of the square Julia image, in points.
const IMAGE_POINTS: f32 = 320.0;
const JULIA_SPAN: f64 = 3.2;
const JULIA_MAX_ITER: u32 = 500;
const PREVIEW_DIVISOR: usize = 4;

/// Julia slot gets a quarter of the main view's threads (at least one), so
/// the main view keeps priority.
pub fn julia_threads(main_threads: usize) -> usize {
    (main_threads / 4).max(1)
}

/// Something the user asked for from the panel.
pub enum JuliaAction {
    None,
    /// Stop pinning: resume following the cursor.
    Unpin,
    /// Switch the main view to this Julia set.
    OpenInMain(Complex64),
}

pub struct JuliaPanel {
    pub visible: bool,
    /// c fixed by a frozen orbit (set by the app); the panel stops
    /// following the cursor.
    pub pinned: Option<Complex64>,
    /// c under the cursor.
    pub follow: Option<Complex64>,
    slot: RenderSlot,
    texture: Option<TextureHandle>,
    pixels: Vec<[u8; 4]>,
    last_move: Instant,
    /// (c, full resolution?, image side in px) of the last submission.
    submitted: Option<(Complex64, bool, usize)>,
}

impl JuliaPanel {
    pub fn new(ctx: &egui::Context, main_threads: usize) -> Self {
        Self {
            visible: true,
            pinned: None,
            follow: None,
            slot: repainting_slot(ctx, julia_threads(main_threads)),
            texture: None,
            pixels: Vec::new(),
            last_move: Instant::now(),
            submitted: None,
        }
    }

    pub fn current_c(&self) -> Option<Complex64> {
        self.pinned.or(self.follow)
    }

    pub fn set_main_threads(&mut self, main_threads: usize) {
        self.slot.set_threads(julia_threads(main_threads));
        self.submitted = None;
    }

    /// Feed the point under the cursor (ignored while pinned).
    pub fn hover(&mut self, c: Option<Complex64>) {
        if self.pinned.is_none() && c.is_some() && c != self.follow {
            self.follow = c;
            self.last_move = Instant::now();
        }
    }

    pub fn cancel(&self) {
        self.slot.cancel();
    }

    fn update_render(&mut self, ctx: &egui::Context, side: usize, color: &ColorSettings) {
        if let Some(out) = self.slot.poll() {
            self.pixels.resize(out.samples.len(), [0; 4]);
            colorize(&out.samples, color, &mut self.pixels);
            let size = [out.view.width, out.view.height];
            upload(ctx, &mut self.texture, "julia", size, &self.pixels);
        }
        let Some(c) = self.current_c() else { return };
        let settled = self.pinned.is_some() || self.last_move.elapsed() >= SETTLE;
        if !settled {
            ctx.request_repaint_after(SETTLE);
        }
        if self.submitted == Some((c, settled, side)) || side == 0 {
            return;
        }
        let n = if settled {
            side
        } else {
            (side / PREVIEW_DIVISOR).max(1)
        };
        self.slot.submit(RenderRequest {
            sampler: Arc::new(F64Sampler {
                kind: FractalKind::Julia { c },
                max_iter: JULIA_MAX_ITER,
            }),
            view: Viewport::new(Complex64::new(0.0, 0.0), JULIA_SPAN, n, n),
            progressive: false,
        });
        self.submitted = Some((c, settled, side));
    }

    /// Recolor the current image after a palette change.
    pub fn invalidate(&mut self) {
        self.submitted = None;
    }

    pub fn show(&mut self, ui: &mut egui::Ui, color: &ColorSettings) -> JuliaAction {
        let mut action = JuliaAction::None;
        let ppp = ui.ctx().pixels_per_point();
        let side = (IMAGE_POINTS * ppp).round() as usize;
        self.update_render(&ui.ctx().clone(), side, color);

        ui.heading("Julia set");
        note(ui, "julia_note", NOTE_JULIA_LINK);
        match self.current_c() {
            Some(c) => {
                ui.monospace(format!("c = {:+.6} {:+.6}i", c.re, c.im));
                if let Some(t) = &self.texture {
                    ui.add(
                        egui::Image::new(t)
                            .fit_to_exact_size(egui::vec2(IMAGE_POINTS, IMAGE_POINTS)),
                    );
                }
                ui.horizontal(|ui| {
                    if self.pinned.is_some() {
                        if ui.button("Unpin").clicked() {
                            action = JuliaAction::Unpin;
                        }
                    } else {
                        ui.label("click twice to pin");
                    }
                    if ui.button("Open in main view").clicked() {
                        action = JuliaAction::OpenInMain(c);
                    }
                });
            }
            None => {
                ui.label("Hover over the Mandelbrot set: each point c has its own Julia set.");
            }
        }
        action
    }
}

#[cfg(test)]
mod tests {
    use super::julia_threads;

    #[test]
    fn julia_gets_a_quarter_of_main_threads() {
        assert_eq!(julia_threads(12), 3);
        assert_eq!(julia_threads(2), 1);
        assert_eq!(julia_threads(1), 1);
    }
}
