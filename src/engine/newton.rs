//! Newton's method fractal: color each starting point by which root of a
//! polynomial Newton's method converges to.
//!
//! Newton's method repeats z → z − p(z)/p'(z). Near a root it converges
//! fast; the fractal lives on the boundaries between the basins.

use crate::engine::escape::Sample;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Distance to a root that counts as "converged".
pub const NEWTON_TOL: f64 = 1e-6;

/// The polynomials on offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NewtonPoly {
    /// z³ − 1
    CubeRootsOfUnity,
    /// z⁴ − 1
    FourthRootsOfUnity,
    /// z³ − 2z + 2: has a region where Newton gets stuck in the cycle 0 → 1 → 0.
    CycleCubic,
}

impl NewtonPoly {
    pub const ALL: [NewtonPoly; 3] = [
        NewtonPoly::CubeRootsOfUnity,
        NewtonPoly::FourthRootsOfUnity,
        NewtonPoly::CycleCubic,
    ];

    pub fn roots(&self) -> Vec<Complex64> {
        let c = Complex64::new;
        match self {
            NewtonPoly::CubeRootsOfUnity => {
                let h = 3f64.sqrt() / 2.0;
                vec![c(1.0, 0.0), c(-0.5, h), c(-0.5, -h)]
            }
            NewtonPoly::FourthRootsOfUnity => {
                vec![c(1.0, 0.0), c(0.0, 1.0), c(-1.0, 0.0), c(0.0, -1.0)]
            }
            NewtonPoly::CycleCubic => vec![
                c(-1.769_292_354_238_631_4, 0.0),
                c(0.884_646_177_119_315_7, 0.589_742_805_022_205_6),
                c(0.884_646_177_119_315_7, -0.589_742_805_022_205_6),
            ],
        }
    }

    /// (p(z), p'(z))
    pub fn eval(&self, z: Complex64) -> (Complex64, Complex64) {
        let one = Complex64::new(1.0, 0.0);
        match self {
            NewtonPoly::CubeRootsOfUnity => (z * z * z - one, 3.0 * z * z),
            NewtonPoly::FourthRootsOfUnity => (z.powu(4) - one, 4.0 * z * z * z),
            NewtonPoly::CycleCubic => (z * z * z - 2.0 * z + 2.0, 3.0 * z * z - 2.0),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            NewtonPoly::CubeRootsOfUnity => "z³ − 1",
            NewtonPoly::FourthRootsOfUnity => "z⁴ − 1",
            NewtonPoly::CycleCubic => "z³ − 2z + 2",
        }
    }

    fn root_near(&self, z: Complex64) -> Option<u8> {
        self.roots()
            .iter()
            .position(|r| (z - r).norm() < NEWTON_TOL)
            .map(|i| i as u8)
    }
}

/// One Newton step, or `None` where it is undefined (flat spot, p'(z) ≈ 0)
/// or has blown up to a non-finite value.
pub fn newton_step(poly: NewtonPoly, z: Complex64) -> Option<Complex64> {
    let (p, dp) = poly.eval(z);
    let slope = dp.norm();
    if slope.is_nan() || slope < 1e-14 {
        return None;
    }
    let next = z - p / dp;
    next.is_finite().then_some(next)
}

/// Run Newton's method from `z0` and report which root it reaches.
pub fn newton_iterate(poly: NewtonPoly, z0: Complex64, max_iter: u32) -> Sample {
    let mut z = z0;
    for n in 0..=max_iter {
        if let Some(root) = poly.root_near(z) {
            return Sample::Converged { root, iter: n };
        }
        match newton_step(poly, z) {
            Some(next) => z = next,
            None => break,
        }
    }
    Sample::Inside
}

/// Newton iterates z₀, z₁, … up to convergence, failure, or `max_iter`.
pub fn newton_orbit(poly: NewtonPoly, z0: Complex64, max_iter: u32) -> Vec<Complex64> {
    let mut z = z0;
    let mut path = vec![z];
    for _ in 0..max_iter {
        if poly.root_near(z).is_some() {
            break;
        }
        match newton_step(poly, z) {
            Some(next) => {
                z = next;
                path.push(z);
            }
            None => break,
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::escape::Sample;
    use num_complex::Complex64;

    const ALL: [NewtonPoly; 3] = NewtonPoly::ALL;

    #[test]
    fn roots_are_roots() {
        for p in ALL {
            for r in p.roots() {
                assert!(p.eval(r).0.norm() < 1e-10, "{p:?} {r}");
            }
        }
    }

    #[test]
    fn near_root_converges_to_it() {
        let p = NewtonPoly::CubeRootsOfUnity;
        match newton_iterate(p, Complex64::new(1.1, 0.0), 100) {
            Sample::Converged { root, .. } => {
                assert!((p.roots()[root as usize] - Complex64::new(1.0, 0.0)).norm() < 1e-9)
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn zero_derivative_is_inside() {
        let s = newton_iterate(NewtonPoly::CubeRootsOfUnity, Complex64::new(0.0, 0.0), 100);
        assert_eq!(s, Sample::Inside);
    }

    #[test]
    fn cycle_region_is_inside() {
        let s = newton_iterate(NewtonPoly::CycleCubic, Complex64::new(0.0, 0.0), 100);
        assert_eq!(s, Sample::Inside);
    }

    #[test]
    fn nan_input_is_inside() {
        let s = newton_iterate(
            NewtonPoly::CubeRootsOfUnity,
            Complex64::new(f64::NAN, 0.0),
            100,
        );
        assert_eq!(s, Sample::Inside);
    }
}
