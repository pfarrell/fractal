//! IFS mode: the map table editor, the chaos-game render, and the map
//! outlines drawn over the picture.

use crate::engine::color::{ColorSettings, colorize_density};
use crate::engine::ifs::{AffineMap, Density, IfsSystem, chaos_game};
use crate::engine::jobs::JobSlot;
use crate::engine::viewport::Viewport;
use crate::ui::app::upload;
use crate::ui::notes::{NOTE_IFS, note};
use crate::ui::view::{ScreenMapping, show_view};
use eframe::egui::{self, Color32, Stroke, TextureHandle};
use num_complex::Complex64;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One color per map, used for its table swatch and its outline.
pub const MAP_COLORS: [Color32; 8] = [
    Color32::from_rgb(255, 99, 99),
    Color32::from_rgb(99, 210, 120),
    Color32::from_rgb(110, 160, 255),
    Color32::from_rgb(255, 210, 80),
    Color32::from_rgb(220, 120, 255),
    Color32::from_rgb(80, 220, 220),
    Color32::from_rgb(255, 150, 60),
    Color32::from_rgb(200, 200, 200),
];

const SEED: u64 = 1;

/// Edit the maps as a table. Returns true if anything changed.
pub fn ifs_table(ui: &mut egui::Ui, sys: &mut IfsSystem) -> bool {
    let mut changed = false;
    let mut remove = None;
    egui::Grid::new("ifs_maps")
        .striped(true)
        .spacing([4.0, 2.0])
        .show(ui, |ui| {
            ui.label("");
            for h in ["a", "b", "c", "d", "e", "f", "p", ""] {
                ui.label(h);
            }
            ui.end_row();
            let can_remove = sys.maps.len() > 1;
            for (i, m) in sys.maps.iter_mut().enumerate() {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, 2.0, MAP_COLORS[i % MAP_COLORS.len()]);
                for v in [
                    &mut m.a, &mut m.b, &mut m.c, &mut m.d, &mut m.e, &mut m.f, &mut m.p,
                ] {
                    changed |= ui
                        .add(egui::DragValue::new(v).speed(0.005).max_decimals(3))
                        .changed();
                }
                if ui
                    .add_enabled(can_remove, egui::Button::new("✕").small())
                    .clicked()
                {
                    remove = Some(i);
                }
                ui.end_row();
            }
        });
    if let Some(i) = remove {
        sys.maps.remove(i);
        changed = true;
    }
    if ui.button("Add map").clicked() {
        sys.maps
            .push(AffineMap::new(0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1));
        changed = true;
    }
    changed
}

struct IfsOutput {
    density: Density,
    view: Viewport,
    elapsed: Duration,
}

/// Everything about the IFS picture.
pub struct IfsState {
    pub system: IfsSystem,
    pub view: Viewport,
    pub points: u64,
    pub show_outlines: bool,
    pub density: Option<Density>,
    density_view: Viewport,
    last_time: Option<Duration>,
    slot: JobSlot<IfsOutput>,
    texture: Option<TextureHandle>,
    pixels: Vec<[u8; 4]>,
    needs_render: bool,
    /// The view has been framed or set deliberately (don't auto-fit again).
    placed: bool,
}

impl IfsState {
    pub fn new(ctx: &egui::Context, threads: usize) -> Self {
        let ctx = ctx.clone();
        let view = Viewport::default_for(0, 0);
        Self {
            system: IfsSystem::barnsley_fern(),
            view,
            points: 5_000_000,
            show_outlines: true,
            density: None,
            density_view: view,
            last_time: None,
            slot: JobSlot::new(threads, Arc::new(move || ctx.request_repaint())),
            texture: None,
            pixels: Vec::new(),
            needs_render: true,
            placed: false,
        }
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.slot.set_threads(threads);
        self.needs_render = true;
    }

    pub fn cancel(&self) {
        self.slot.cancel();
    }

    /// Fit the view to the attractor (keeping the current pixel size).
    pub fn fit(&mut self) {
        self.view = self.system.fit_view(self.view.width, self.view.height);
        self.placed = !self.view.is_empty();
        self.needs_render = true;
    }

    pub fn recolor(&mut self, ctx: &egui::Context, color: &ColorSettings) {
        if let Some(d) = &self.density {
            self.pixels.resize(d.counts.len(), [0; 4]);
            colorize_density(&d.counts, color, &mut self.pixels);
            upload(
                ctx,
                &mut self.texture,
                "ifs",
                [d.width, d.height],
                &self.pixels,
            );
        }
    }

    /// Left-panel controls (below the shared color/thread controls).
    pub fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Iterated function system");
        note(ui, "ifs_note", NOTE_IFS);
        egui::ComboBox::from_id_salt("preset")
            .selected_text(&self.system.name)
            .show_ui(ui, |ui| {
                for p in IfsSystem::presets() {
                    if ui
                        .selectable_label(p.name == self.system.name, &p.name)
                        .clicked()
                    {
                        self.system = p;
                        self.fit();
                    }
                }
            });
        ui.label("Each row maps (x, y) → (a·x + b·y + e, c·x + d·y + f), picked with weight p.");
        egui::ScrollArea::horizontal().show(ui, |ui| {
            self.needs_render |= ifs_table(ui, &mut self.system);
        });
        let mut pts = self.points as f64;
        if crate::ui::slider(
            ui,
            egui::Slider::new(&mut pts, 1e5..=5e7)
                .logarithmic(true)
                .text("points"),
        )
        .changed()
        {
            self.points = pts as u64;
            self.needs_render = true;
        }
        ui.checkbox(&mut self.show_outlines, "Show map outlines");
        if ui.button("Fit view").clicked() {
            self.fit();
        }
        if let Some(t) = self.last_time {
            let secs = t.as_secs_f64().max(1e-9);
            ui.label(format!(
                "{:.0} ms · {} threads · {:.0} M points/s",
                secs * 1000.0,
                self.slot.threads(),
                self.points as f64 / secs / 1e6
            ));
        }
    }

    /// Set the view's center and span (e.g. from a bookmark).
    pub fn set_view(&mut self, center: Complex64, span: f64) {
        self.view.center = center;
        self.view.span = span;
        self.placed = true;
        self.needs_render = true;
    }

    /// Track the on-screen size. The first time there is a real size, and
    /// nothing has placed the view yet, frame the whole attractor.
    pub fn resize(&mut self, width: usize, height: usize) {
        if (self.view.width, self.view.height) == (width, height) {
            return;
        }
        self.view.width = width;
        self.view.height = height;
        if !self.placed && !self.view.is_empty() {
            self.fit();
        }
        self.needs_render = true;
    }

    pub fn mark_dirty(&mut self) {
        self.needs_render = true;
    }

    /// The central view. Returns the complex point under the cursor.
    pub fn show_view(&mut self, ui: &mut egui::Ui, color: &ColorSettings) -> Option<Complex64> {
        let ctx = ui.ctx().clone();
        if let Some(out) = self.slot.poll() {
            self.last_time = Some(out.elapsed);
            self.density = Some(out.density);
            self.density_view = out.view;
            self.recolor(&ctx, color);
        }

        let image = self.texture.as_ref().map(|t| (t, &self.density_view));
        let resp = show_view(ui, image, &mut self.view);
        let (w, h) = resp.mapping.pixel_size();
        self.resize(w, h);
        self.needs_render |= resp.view_changed;
        if self.show_outlines {
            draw_outlines(ui, &resp.mapping, &self.view, &self.system);
        }

        if self.needs_render && !self.view.is_empty() {
            let (sys, view, points) = (self.system.clone(), self.view, self.points);
            self.slot.run(move |pool, token, emit| {
                let start = Instant::now();
                if let Some(density) = chaos_game(&sys, &view, points, SEED, pool, token) {
                    emit(IfsOutput {
                        density,
                        view,
                        elapsed: start.elapsed(),
                    });
                }
            });
            self.needs_render = false;
        }
        resp.hovered
    }
}

/// Draw each map's image of the attractor's bounding box.
fn draw_outlines(ui: &egui::Ui, mapping: &ScreenMapping, view: &Viewport, sys: &IfsSystem) {
    let painter = ui.painter_at(mapping.rect);
    for i in 0..sys.maps.len() {
        if let Some(pts) = outline_points(mapping, view, sys.map_outline(i)) {
            let stroke = Stroke::new(1.5, MAP_COLORS[i % MAP_COLORS.len()]);
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
    }
}

/// Screen points for an outline, or `None` if any corner is non-finite or
/// absurdly far off screen (a map edited to diverge). Such coordinates
/// would otherwise reach egui's tessellator as inf/NaN and panic.
fn outline_points(
    mapping: &ScreenMapping,
    view: &Viewport,
    quad: [(f64, f64); 4],
) -> Option<Vec<egui::Pos2>> {
    let limit = mapping
        .rect
        .expand(mapping.rect.width().max(mapping.rect.height()) * 4.0);
    quad.iter()
        .map(|&(x, y)| {
            let p = mapping.to_screen(view, Complex64::new(x, y));
            (p.x.is_finite() && p.y.is_finite() && limit.contains(p)).then_some(p)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{Pos2, Rect};

    fn mapping() -> ScreenMapping {
        ScreenMapping {
            rect: Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 600.0)),
            ppp: 1.0,
        }
    }

    #[test]
    fn diverging_outline_is_skipped() {
        let mut sys = IfsSystem::sierpinski();
        sys.maps[0].a = 5.0;
        let view = IfsSystem::sierpinski().fit_view(800, 600);
        assert!(outline_points(&mapping(), &view, sys.map_outline(0)).is_none());
        let inf = [
            (f64::INFINITY, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (f64::NAN, 1.0),
        ];
        assert!(outline_points(&mapping(), &view, inf).is_none());
    }

    #[test]
    fn normal_outline_is_drawn() {
        let sys = IfsSystem::sierpinski();
        let view = sys.fit_view(800, 600);
        assert_eq!(
            outline_points(&mapping(), &view, sys.map_outline(0)).map(|p| p.len()),
            Some(4)
        );
    }

    #[test]
    fn bookmarked_view_survives_first_sizing() {
        let mut s = IfsState::new(&egui::Context::default(), 1);
        s.set_view(Complex64::new(0.1, 7.0), 0.5);
        s.resize(800, 600);
        assert_eq!(
            (s.view.center, s.view.span),
            (Complex64::new(0.1, 7.0), 0.5)
        );
        s.resize(0, 0); // minimized
        s.resize(800, 600); // restored
        assert_eq!(
            (s.view.center, s.view.span),
            (Complex64::new(0.1, 7.0), 0.5)
        );
    }

    #[test]
    fn fresh_state_fits_on_first_sizing() {
        let mut s = IfsState::new(&egui::Context::default(), 1);
        s.resize(800, 600);
        assert_eq!(s.view, s.system.fit_view(800, 600));
    }
}
