//! Escape-time fractals: iterate a formula per point and count how long it
//! takes to fly off to infinity.
//!
//! For the "parameter plane" fractals (Mandelbrot, Burning Ship, Tricorn,
//! Multibrot) the pixel is `c` and every orbit starts at z₀ = 0. For Julia
//! sets `c` is fixed and the pixel is the starting point z₀.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Escape radius. Anything this far out is certain to diverge; a large
/// radius (rather than the minimal 2) makes smooth coloring smoother.
pub const BAILOUT: f64 = 256.0;

/// Which formula to iterate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FractalKind {
    /// z → z² + c
    Mandelbrot,
    /// z → z² + c with c fixed; the pixel is z₀.
    Julia { c: Complex64 },
    /// z → (|Re z| + i|Im z|)² + c
    BurningShip,
    /// z → conj(z)² + c (a.k.a. Mandelbar)
    Tricorn,
    /// z → z^power + c, power in 2..=8
    Multibrot { power: u32 },
}

/// The outcome of iterating one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sample {
    /// Stayed bounded for the whole iteration budget.
    Inside,
    /// Escaped after `iter` steps; `smooth` is the continuous version of
    /// `iter` used for band-free coloring.
    Escaped { iter: u32, smooth: f32 },
    /// Newton's method reached root number `root` after `iter` steps.
    Converged { root: u8, iter: u32 },
}

impl FractalKind {
    pub fn label(&self) -> &'static str {
        match self {
            FractalKind::Mandelbrot => "Mandelbrot",
            FractalKind::Julia { .. } => "Julia",
            FractalKind::BurningShip => "Burning Ship",
            FractalKind::Tricorn => "Tricorn",
            FractalKind::Multibrot { .. } => "Multibrot",
        }
    }

    /// The exponent of the formula (used by smooth coloring).
    pub fn power(&self) -> u32 {
        match self {
            FractalKind::Multibrot { power } => *power,
            _ => 2,
        }
    }

    /// Starting z and the constant c for a given pixel value.
    fn start(&self, point: Complex64) -> (Complex64, Complex64) {
        match self {
            FractalKind::Julia { c } => (point, *c),
            _ => (Complex64::new(0.0, 0.0), point),
        }
    }

    /// One application of the formula. `iterate` and `orbit` both use
    /// this, so they can never disagree.
    fn step(&self, z: Complex64, c: Complex64) -> Complex64 {
        match self {
            FractalKind::Mandelbrot | FractalKind::Julia { .. } => z * z + c,
            FractalKind::BurningShip => {
                let w = Complex64::new(z.re.abs(), z.im.abs());
                w * w + c
            }
            FractalKind::Tricorn => {
                let w = z.conj();
                w * w + c
            }
            FractalKind::Multibrot { power } => z.powu(*power) + c,
        }
    }
}

fn escaped(z: Complex64) -> bool {
    z.norm_sqr() > BAILOUT * BAILOUT
}

/// Continuous iteration count: `iter + 1 - log_power(ln|z| / ln BAILOUT)`.
/// It removes the visible bands between integer escape counts.
pub fn smooth_value(iter: u32, z_abs: f64, power: u32) -> f32 {
    let nu = (z_abs.ln() / BAILOUT.ln()).ln() / (power as f64).ln();
    (iter as f64 + 1.0 - nu) as f32
}

/// Iterate one point and classify it.
pub fn iterate(kind: &FractalKind, point: Complex64, max_iter: u32) -> Sample {
    let (mut z, c) = kind.start(point);
    for n in 1..=max_iter {
        z = kind.step(z, c);
        if escaped(z) {
            return Sample::Escaped {
                iter: n,
                smooth: smooth_value(n, z.norm(), kind.power()),
            };
        }
    }
    Sample::Inside
}

/// The sequence z₀, z₁, … for one point, ending at the first escaping
/// value or after `max_iter` steps.
pub fn orbit(kind: &FractalKind, point: Complex64, max_iter: u32) -> Vec<Complex64> {
    let (mut z, c) = kind.start(point);
    let mut path = vec![z];
    for _ in 0..max_iter {
        z = kind.step(z, c);
        path.push(z);
        if escaped(z) {
            break;
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn mandelbrot_known_inside() {
        for p in [
            c(0.0, 0.0),
            c(-1.0, 0.0),
            c(0.0, 1.0),
            c(0.25, 0.0),
            c(-1.75, 0.0),
        ] {
            assert_eq!(
                iterate(&FractalKind::Mandelbrot, p, 1000),
                Sample::Inside,
                "{p}"
            );
        }
    }

    #[test]
    fn mandelbrot_escape_counts() {
        assert!(matches!(
            iterate(&FractalKind::Mandelbrot, c(2.0, 0.0), 1000),
            Sample::Escaped { iter: 4, .. }
        ));
        assert!(matches!(
            iterate(&FractalKind::Mandelbrot, c(1.0, 0.0), 1000),
            Sample::Escaped { iter: 5, .. }
        ));
    }

    #[test]
    fn smooth_values() {
        assert!((smooth_value(4, 1446.0, 2) - 4.60798).abs() < 1e-4);
        assert!((smooth_value(5, 677.0, 2) - 5.76688).abs() < 1e-4);
        match iterate(&FractalKind::Mandelbrot, c(2.0, 0.0), 1000) {
            Sample::Escaped { smooth, .. } => assert!((smooth - 4.60798).abs() < 1e-4),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn julia_c0_is_unit_disk() {
        let k = FractalKind::Julia { c: c(0.0, 0.0) };
        assert_eq!(iterate(&k, c(0.5, 0.0), 1000), Sample::Inside);
        assert!(matches!(
            iterate(&k, c(1.5, 0.0), 1000),
            Sample::Escaped { .. }
        ));
    }

    #[test]
    fn orbit_second_step_per_kind() {
        let p = c(-1.0, 1.0);
        let cases = [
            (FractalKind::Mandelbrot, c(-1.0, -1.0)),
            (FractalKind::BurningShip, c(-1.0, 3.0)),
            (FractalKind::Tricorn, c(-1.0, 3.0)),
        ];
        for (k, want) in cases {
            let o = orbit(&k, p, 3);
            assert_eq!(o[0], c(0.0, 0.0), "{k:?}");
            assert_eq!(o[2], want, "{k:?}");
        }
    }

    #[test]
    fn burning_ship_matches_mandelbrot_on_real_axis_for_positive_c() {
        assert!(matches!(
            iterate(&FractalKind::BurningShip, c(1.0, 0.0), 1000),
            Sample::Escaped { iter: 5, .. }
        ));
    }

    #[test]
    fn multibrot_power3() {
        let k = FractalKind::Multibrot { power: 3 };
        assert_eq!(orbit(&k, c(0.5, 0.0), 5)[2], c(0.625, 0.0));
        assert!(matches!(
            iterate(&k, c(1.0, 0.0), 1000),
            Sample::Escaped { .. }
        ));
    }

    #[test]
    fn orbit_stops_at_escape() {
        let o = orbit(&FractalKind::Mandelbrot, c(2.0, 0.0), 1000);
        assert_eq!(o.len(), 5);
        assert!(o[4].norm() > BAILOUT);
    }

    #[test]
    fn orbit_caps_at_max_iter() {
        assert_eq!(orbit(&FractalKind::Mandelbrot, c(0.0, 0.0), 50).len(), 51);
    }
}
