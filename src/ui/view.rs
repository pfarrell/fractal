//! The interactive fractal view: draws the rendered image and turns mouse
//! input into zoom, pan, hover, and click.

use crate::engine::deep::HiComplex;
use crate::engine::viewport::Viewport;
use eframe::egui::{self, Color32, Pos2, Rect, Sense, TextureHandle};
use num_complex::Complex64;

/// What happened in the view this frame.
pub struct ViewResponse {
    /// Complex point under the cursor.
    pub hovered: Option<Complex64>,
    /// Complex point under the cursor, at the view's precision.
    pub hovered_hi: Option<HiComplex>,
    /// Complex point clicked (without dragging).
    pub clicked: Option<Complex64>,
    /// Complex point right-clicked (where a context menu opens), at the
    /// view's precision.
    pub secondary_clicked: Option<HiComplex>,
    /// The view's own response, for attaching a context menu.
    pub response: egui::Response,
    /// The view was zoomed or panned.
    pub view_changed: bool,
    /// Where the view sits on screen, for overlays.
    pub mapping: ScreenMapping,
}

/// Converts between the view's physical-pixel coordinates and egui's
/// logical screen points.
#[derive(Clone, Copy)]
pub struct ScreenMapping {
    pub rect: Rect,
    /// Physical pixels per logical point.
    pub ppp: f32,
}

impl ScreenMapping {
    /// Physical pixel coordinate of a screen position.
    pub fn to_pixel(&self, pos: Pos2) -> (f64, f64) {
        (
            ((pos.x - self.rect.min.x) * self.ppp) as f64,
            ((pos.y - self.rect.min.y) * self.ppp) as f64,
        )
    }

    /// Screen position of a complex point in `view`.
    pub fn to_screen(&self, view: &Viewport, z: Complex64) -> Pos2 {
        let (x, y) = view.complex_to_pixel(z);
        self.pixel_to_screen(x, y)
    }

    /// Screen position of physical pixel coordinate (x, y).
    pub fn pixel_to_screen(&self, x: f64, y: f64) -> Pos2 {
        Pos2::new(
            self.rect.min.x + x as f32 / self.ppp,
            self.rect.min.y + y as f32 / self.ppp,
        )
    }

    /// View size in physical pixels.
    pub fn pixel_size(&self) -> (usize, usize) {
        (
            (self.rect.width() * self.ppp).round().max(0.0) as usize,
            (self.rect.height() * self.ppp).round().max(0.0) as usize,
        )
    }
}

/// Scroll distance (in points) for one 1.2× zoom step.
const SCROLL_PER_STEP: f64 = 50.0;
const ZOOM_STEP: f64 = 1.2;

/// Show the view filling the available space. `image` is the latest
/// texture plus the viewport it was rendered for; if that differs from
/// `view` (mid-pan, mid-zoom, or a low-res preview) it is drawn stretched
/// to where that region now sits, so the old image slides with the mouse.
/// `floor` is the smallest pixel spacing zooming may reach (see
/// [`crate::engine::viewport::pixel_floor`]).
pub fn show_view(
    ui: &mut egui::Ui,
    image: Option<(&TextureHandle, &Viewport)>,
    view: &mut Viewport,
    floor: f64,
) -> ViewResponse {
    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    let mapping = ScreenMapping {
        rect,
        ppp: ui.ctx().pixels_per_point(),
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::BLACK);

    let mut view_changed = false;
    if let Some(pos) = response.hover_pos() {
        let (px, py) = mapping.to_pixel(pos);
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let mut factor = ZOOM_STEP.powf(scroll as f64 / SCROLL_PER_STEP);
        factor *= pinch as f64;
        if factor != 1.0 {
            view.zoom_about(px, py, factor, floor);
            view_changed = true;
        }
    }
    if response.dragged_by(egui::PointerButton::Primary) {
        let d = response.drag_delta();
        if d != egui::Vec2::ZERO {
            view.pan_pixels((d.x * mapping.ppp) as f64, (d.y * mapping.ppp) as f64);
            view_changed = true;
        }
    }

    if let Some((texture, shown)) = image {
        // Where the rendered region's corners land in the current view.
        // Worked in offsets from the current center, so it holds at depth.
        let shift = shown.center.diff_c64(&view.center);
        let corner = |x: f64, y: f64| {
            let (px, py) = view.offset_to_pixel(shift + shown.pixel_offset(x, y));
            mapping.pixel_to_screen(px, py)
        };
        let target = Rect::from_min_max(
            corner(0.0, 0.0),
            corner(shown.width as f64, shown.height as f64),
        );
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        painter.image(texture.id(), target, uv, Color32::WHITE);
    }

    let to_complex = |pos: Pos2| {
        let (x, y) = mapping.to_pixel(pos);
        view.pixel_to_complex(x, y)
    };
    let hovered = response.hover_pos().map(to_complex);
    let to_hi = |pos: Pos2| {
        let (x, y) = mapping.to_pixel(pos);
        view.pixel_to_hi(x, y)
    };
    let hovered_hi = response.hover_pos().map(to_hi);
    let clicked = if response.clicked() {
        response.interact_pointer_pos().map(to_complex)
    } else {
        None
    };
    let secondary_clicked = if response.secondary_clicked() {
        response.interact_pointer_pos().map(to_hi)
    } else {
        None
    };
    ViewResponse {
        hovered,
        hovered_hi,
        clicked,
        secondary_clicked,
        response,
        view_changed,
        mapping,
    }
}

/// Draw an orbit z₀, z₁, … as a polyline with dots, on top of the view.
pub fn draw_orbit(ui: &egui::Ui, mapping: &ScreenMapping, view: &Viewport, path: &[Complex64]) {
    let painter = ui.painter_at(mapping.rect);
    // Points far outside the view (escaping orbits) are skipped so huge
    // coordinates never reach the tessellator.
    let limit = mapping
        .rect
        .expand(mapping.rect.width().max(mapping.rect.height()) * 4.0);
    let pts: Vec<Option<Pos2>> = path
        .iter()
        .map(|z| Some(mapping.to_screen(view, *z)).filter(|p| limit.contains(*p)))
        .collect();
    let line = egui::Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 200));
    for w in pts.windows(2) {
        if let [Some(a), Some(b)] = w {
            painter.line_segment([*a, *b], line);
        }
    }
    for (i, p) in pts.iter().enumerate() {
        if let Some(p) = p {
            let color = if i == 0 {
                Color32::from_rgb(255, 80, 80)
            } else {
                Color32::from_rgb(255, 230, 120)
            };
            painter.circle_filled(*p, if i == 0 { 4.0 } else { 2.5 }, color);
        }
    }
}
