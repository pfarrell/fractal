//! Mapping between screen pixels and the complex plane, plus zoom/pan.
//!
//! Convention: pixel (0, 0) is the top-left corner and y grows downward,
//! while the imaginary axis grows upward. Renderers sample pixel *centers*,
//! i.e. they call `pixel_to_complex(x + 0.5, y + 0.5)`.

use crate::engine::deep::{HiComplex, bits_for};
use crate::engine::escape::FractalKind;
use crate::engine::perturb::SWITCH;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// The span (visible height in the complex plane) of the default view.
pub const DEFAULT_SPAN: f64 = 3.0;

/// The deepest pixel spacing a Mandelbrot view may have: perturbation
/// keeps pixel offsets as normal f64 values down to here.
pub const DEEP_PIXEL_FLOOR: f64 = 1e-300;

/// A rectangular window onto the complex plane, sized in pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Viewport {
    /// Complex value at the middle of the view.
    pub center: HiComplex,
    /// Height of the visible region in the complex plane.
    pub span: f64,
    pub width: usize,
    pub height: usize,
}

/// The smallest distance between neighboring pixels that f64 can still
/// represent meaningfully near `center`. Zooming further would make
/// neighboring pixels round to the same complex number (blocky garbage).
pub fn min_pixel_scale(center: Complex64) -> f64 {
    8.0 * f64::EPSILON * center.re.abs().max(center.im.abs()).max(1.0)
}

/// The smallest pixel spacing a view of `kind` may have at `center`:
/// Mandelbrot renders by perturbation down to [`DEEP_PIXEL_FLOOR`]; every
/// other kind iterates in f64 and stops at [`min_pixel_scale`].
pub fn pixel_floor(kind: &FractalKind, center: &HiComplex) -> f64 {
    match kind {
        FractalKind::Mandelbrot => DEEP_PIXEL_FLOOR,
        _ => min_pixel_scale(center.to_c64()),
    }
}

/// Iteration budget that grows with zoom depth: deeper views need more
/// iterations to resolve the boundary.
pub fn auto_max_iter(magnification: f64) -> u32 {
    let n = 100.0 + 150.0 * magnification.log10().max(0.0);
    n.clamp(100.0, 50_000.0) as u32
}

/// How the maximum iteration count is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxIter {
    Fixed(u32),
    /// Scale with magnification via [`auto_max_iter`].
    Auto,
}

impl MaxIter {
    pub fn resolve(&self, magnification: f64) -> u32 {
        match self {
            MaxIter::Fixed(n) => *n,
            MaxIter::Auto => auto_max_iter(magnification),
        }
    }
}

impl Viewport {
    pub fn new(center: impl Into<HiComplex>, span: f64, width: usize, height: usize) -> Self {
        Self {
            center: center.into(),
            span,
            width,
            height,
        }
    }

    /// The classic whole-Mandelbrot view.
    pub fn default_for(width: usize, height: usize) -> Self {
        Self::new(Complex64::new(-0.5, 0.0), DEFAULT_SPAN, width, height)
    }

    /// Distance in the complex plane between neighboring pixels.
    pub fn pixel_scale(&self) -> f64 {
        self.span / self.height.max(1) as f64
    }

    /// Offset of pixel position (x, y) from the center, in f64.
    pub fn pixel_offset(&self, x: f64, y: f64) -> Complex64 {
        let s = self.pixel_scale();
        Complex64::new(
            (x - self.width as f64 / 2.0) * s,
            -(y - self.height as f64 / 2.0) * s,
        )
    }

    /// The point at pixel position (x, y), rounded to f64. Exact for
    /// shallow views; only an approximation past the f64 limit.
    pub fn pixel_to_complex(&self, x: f64, y: f64) -> Complex64 {
        let c = self.center.to_c64();
        let d = self.pixel_offset(x, y);
        Complex64::new(c.re + d.re, c.im + d.im)
    }

    /// The point at pixel position (x, y), at the view's precision.
    pub fn pixel_to_hi(&self, x: f64, y: f64) -> HiComplex {
        self.center
            .add_c64(self.pixel_offset(x, y), bits_for(self.pixel_scale()))
    }

    /// Pixel position of the point `offset` away from the center.
    pub fn offset_to_pixel(&self, offset: Complex64) -> (f64, f64) {
        let s = self.pixel_scale();
        (
            offset.re / s + self.width as f64 / 2.0,
            -offset.im / s + self.height as f64 / 2.0,
        )
    }

    /// Pixel position of `z` (an f64 approximation; see
    /// [`Viewport::offset_to_pixel`] for deep views).
    pub fn complex_to_pixel(&self, z: Complex64) -> (f64, f64) {
        self.offset_to_pixel(z - self.center.to_c64())
    }

    /// Zoom by `factor` (> 1 zooms in) keeping the point under pixel
    /// (x, y) fixed. `floor` is the smallest allowed pixel spacing (see
    /// [`pixel_floor`]). Returns `false` if the zoom was limited by it.
    pub fn zoom_about(&mut self, x: f64, y: f64, factor: f64, floor: f64) -> bool {
        let floor_span = floor * self.height.max(1) as f64;
        let wanted = self.span / factor;
        let clamped = wanted < floor_span;
        let new_span = wanted.max(floor_span);
        let ratio = new_span / self.span;
        // anchor − (anchor − center)·ratio = center + offset·(1 − ratio)
        let offset = self.pixel_offset(x, y);
        self.move_center(offset * (1.0 - ratio), new_span / self.height.max(1) as f64);
        self.span = new_span;
        !clamped
    }

    /// Move the view so the content follows a drag of (dx, dy) pixels.
    pub fn pan_pixels(&mut self, dx: f64, dy: f64) {
        let s = self.pixel_scale();
        self.move_center(Complex64::new(-dx * s, dy * s), s);
    }

    /// Move the center by `delta` for a view whose pixels end up
    /// `pixel_scale` apart. Where f64 is still plenty (shallower than the
    /// perturbation switch-over) the center stays an exact f64, so shallow
    /// work behaves and saves exactly as it did before deep zoom.
    fn move_center(&mut self, delta: Complex64, pixel_scale: f64) {
        let c = self.center.to_c64();
        self.center = if pixel_scale >= SWITCH * min_pixel_scale(c) {
            HiComplex::from_c64(c + delta)
        } else {
            self.center.add_c64(delta, bits_for(pixel_scale))
        };
    }

    /// Same region of the plane at a different pixel size.
    pub fn with_size(&self, width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            ..self.clone()
        }
    }

    /// Zoom relative to the default view.
    pub fn magnification(&self) -> f64 {
        DEFAULT_SPAN / self.span
    }

    /// The view sits at `floor` (see [`pixel_floor`]).
    pub fn at_precision_floor(&self, floor: f64) -> bool {
        self.pixel_scale() <= floor * (1.0 + 1e-6)
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn view() -> Viewport {
        Viewport::new(Complex64::new(-0.75, 0.1), 0.5, 800, 600)
    }

    #[test]
    fn pixel_round_trip() {
        let v = view();
        let (x, y) = v.complex_to_pixel(v.pixel_to_complex(123.0, 456.0));
        assert!((x - 123.0).abs() < 1e-9 && (y - 456.0).abs() < 1e-9);
    }

    #[test]
    fn center_pixel_maps_to_center() {
        let v = view();
        assert_eq!(v.pixel_to_complex(400.0, 300.0), v.center.to_c64());
    }

    #[test]
    fn y_down_is_imag_down() {
        let v = view();
        assert!(v.pixel_to_complex(400.0, 0.0).im > v.center.to_c64().im);
    }

    #[test]
    fn zoom_keeps_cursor_point_fixed() {
        let mut v = view();
        let before = v.pixel_to_complex(100.0, 200.0);
        assert!(v.zoom_about(100.0, 200.0, 2.0, min_pixel_scale(v.center.to_c64())));
        let after = v.pixel_to_complex(100.0, 200.0);
        assert!((after - before).norm() <= 1e-12 * before.norm());
        assert!((v.span - 0.25).abs() < 1e-15);
    }

    #[test]
    fn zoom_clamps_at_precision_floor() {
        let mut v = view();
        let floor = pixel_floor(&FractalKind::BurningShip, &v.center);
        let mut last = true;
        for _ in 0..40 {
            last = v.zoom_about(100.0, 200.0, 10.0, floor);
        }
        assert!(!last);
        assert!(v.pixel_scale() >= floor);
        assert!(v.at_precision_floor(floor));
    }

    #[test]
    fn other_kinds_keep_f64_floor() {
        let c = view().center;
        assert_eq!(
            pixel_floor(&FractalKind::BurningShip, &c),
            min_pixel_scale(c.to_c64())
        );
        assert_eq!(pixel_floor(&FractalKind::Mandelbrot, &c), DEEP_PIXEL_FLOOR);
    }

    #[test]
    fn mandelbrot_floor_is_deep() {
        let mut v = view();
        let mut last = true;
        for _ in 0..400 {
            last = v.zoom_about(100.0, 200.0, 10.0, DEEP_PIXEL_FLOOR);
        }
        assert!(!last);
        assert!(v.pixel_scale() >= DEEP_PIXEL_FLOOR);
        assert!(v.at_precision_floor(DEEP_PIXEL_FLOOR));
    }

    #[test]
    fn deep_zoom_keeps_cursor_point() {
        let mut v = view();
        for _ in 0..40 {
            v.zoom_about(100.0, 200.0, 10.0, DEEP_PIXEL_FLOOR);
        }
        let before = v.pixel_to_hi(100.0, 200.0);
        v.zoom_about(100.0, 200.0, 2.0, DEEP_PIXEL_FLOOR);
        let after = v.pixel_to_hi(100.0, 200.0);
        assert!(after.diff_c64(&before).norm() < 0.01 * v.pixel_scale());
    }

    #[test]
    fn pan_moves_by_pixels() {
        let mut v = view();
        let s = v.pixel_scale();
        let before = v.center.clone();
        v.pan_pixels(10.0, 0.0);
        let d = v.center.diff_c64(&before);
        assert!((d.re + 10.0 * s).abs() < 1e-15);
        assert_eq!(d.im, 0.0);
    }

    #[test]
    fn deep_pan_moves_by_pixels() {
        let mut v = Viewport::new(Complex64::new(-0.75, 0.1), 1e-60, 800, 600);
        let s = v.pixel_scale();
        let before = v.center.clone();
        v.pan_pixels(10.0, 0.0);
        let d = v.center.diff_c64(&before);
        assert!((d.re + 10.0 * s).abs() <= 1e-9 * 10.0 * s, "{d}");
        assert_eq!(d.im, 0.0);
    }

    #[test]
    fn shallow_points_unchanged() {
        let v = view();
        let c = Complex64::new(-0.75, 0.1);
        let s = v.pixel_scale();
        for (x, y) in [(0.5, 0.5), (123.5, 456.5), (799.5, 599.5)] {
            let old = Complex64::new(c.re + (x - 400.0) * s, c.im - (y - 300.0) * s);
            assert_eq!(v.pixel_to_complex(x, y), old);
        }
    }

    #[test]
    fn empty_view_is_safe() {
        let v = Viewport::new(Complex64::new(0.0, 0.0), 3.0, 0, 0);
        assert!(v.is_empty());
        assert!(v.pixel_scale().is_finite());
    }

    #[test]
    fn auto_iter_values() {
        assert_eq!(auto_max_iter(1.0), 100);
        assert_eq!(auto_max_iter(1e4), 700);
        assert_eq!(auto_max_iter(1e300), 45_100);
    }

    #[test]
    fn max_iter_resolve() {
        assert_eq!(MaxIter::Fixed(500).resolve(1e9), 500);
        assert_eq!(MaxIter::Auto.resolve(1e4), 700);
    }

    #[test]
    fn shallow_zoom_and_pan_stay_f64_exact() {
        let mut v = Viewport::default_for(800, 600);
        v.zoom_about(123.0, 77.0, 1.2, DEEP_PIXEL_FLOOR);
        v.pan_pixels(3.0, -5.0);
        let json = serde_json::to_string(&v.center).unwrap();
        assert!(json.starts_with('['), "{json}");
        for _ in 0..40 {
            v.zoom_about(100.0, 100.0, 10.0, DEEP_PIXEL_FLOOR);
        }
        for _ in 0..40 {
            v.zoom_about(100.0, 100.0, 0.1, DEEP_PIXEL_FLOOR);
        }
        let json = serde_json::to_string(&v.center).unwrap();
        assert!(json.starts_with('['), "back out: {json}");
    }
}
