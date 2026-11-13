//! Mapping between screen pixels and the complex plane, plus zoom/pan.
//!
//! Convention: pixel (0, 0) is the top-left corner and y grows downward,
//! while the imaginary axis grows upward. Renderers sample pixel *centers*,
//! i.e. they call `pixel_to_complex(x + 0.5, y + 0.5)`.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// The span (visible height in the complex plane) of the default view.
pub const DEFAULT_SPAN: f64 = 3.0;

/// A rectangular window onto the complex plane, sized in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Viewport {
    /// Complex value at the middle of the view.
    pub center: Complex64,
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

/// Iteration budget that grows with zoom depth: deeper views need more
/// iterations to resolve the boundary.
pub fn auto_max_iter(magnification: f64) -> u32 {
    let n = 100.0 + 150.0 * magnification.log10().max(0.0);
    n.clamp(100.0, 20_000.0) as u32
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
    pub fn new(center: Complex64, span: f64, width: usize, height: usize) -> Self {
        Self {
            center,
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

    pub fn pixel_to_complex(&self, x: f64, y: f64) -> Complex64 {
        let s = self.pixel_scale();
        Complex64::new(
            self.center.re + (x - self.width as f64 / 2.0) * s,
            self.center.im - (y - self.height as f64 / 2.0) * s,
        )
    }

    pub fn complex_to_pixel(&self, z: Complex64) -> (f64, f64) {
        let s = self.pixel_scale();
        (
            (z.re - self.center.re) / s + self.width as f64 / 2.0,
            -(z.im - self.center.im) / s + self.height as f64 / 2.0,
        )
    }

    /// Zoom by `factor` (> 1 zooms in) keeping the complex point under
    /// pixel (x, y) fixed. Returns `false` if the zoom was limited by the
    /// f64 precision floor.
    pub fn zoom_about(&mut self, x: f64, y: f64, factor: f64) -> bool {
        let anchor = self.pixel_to_complex(x, y);
        let floor_span = min_pixel_scale(anchor) * self.height.max(1) as f64;
        let wanted = self.span / factor;
        let clamped = wanted < floor_span;
        let new_span = wanted.max(floor_span);
        let ratio = new_span / self.span;
        self.center = anchor - (anchor - self.center) * ratio;
        self.span = new_span;
        !clamped
    }

    /// Move the view so the content follows a drag of (dx, dy) pixels.
    pub fn pan_pixels(&mut self, dx: f64, dy: f64) {
        let s = self.pixel_scale();
        self.center.re -= dx * s;
        self.center.im += dy * s;
    }

    /// Same region of the plane at a different pixel size.
    pub fn with_size(&self, width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            ..*self
        }
    }

    /// Zoom relative to the default view.
    pub fn magnification(&self) -> f64 {
        DEFAULT_SPAN / self.span
    }

    pub fn at_precision_floor(&self) -> bool {
        self.pixel_scale() <= min_pixel_scale(self.center) * (1.0 + 1e-6)
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
        assert_eq!(v.pixel_to_complex(400.0, 300.0), v.center);
    }

    #[test]
    fn y_down_is_imag_down() {
        let v = view();
        assert!(v.pixel_to_complex(400.0, 0.0).im > v.center.im);
    }

    #[test]
    fn zoom_keeps_cursor_point_fixed() {
        let mut v = view();
        let before = v.pixel_to_complex(100.0, 200.0);
        assert!(v.zoom_about(100.0, 200.0, 2.0));
        let after = v.pixel_to_complex(100.0, 200.0);
        assert!((after - before).norm() <= 1e-12 * before.norm());
        assert!((v.span - 0.25).abs() < 1e-15);
    }

    #[test]
    fn zoom_clamps_at_precision_floor() {
        let mut v = view();
        let mut last = true;
        for _ in 0..40 {
            last = v.zoom_about(100.0, 200.0, 10.0);
        }
        assert!(!last);
        assert!(v.pixel_scale() >= min_pixel_scale(v.center));
        assert!(v.at_precision_floor());
    }

    #[test]
    fn pan_moves_by_pixels() {
        let mut v = view();
        let s = v.pixel_scale();
        let before = v.center;
        v.pan_pixels(10.0, 0.0);
        assert!((v.center.re - (before.re - 10.0 * s)).abs() < 1e-15);
        assert_eq!(v.center.im, before.im);
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
        assert_eq!(auto_max_iter(1e300), 20_000);
    }

    #[test]
    fn max_iter_resolve() {
        assert_eq!(MaxIter::Fixed(500).resolve(1e9), 500);
        assert_eq!(MaxIter::Auto.resolve(1e4), 700);
    }
}
