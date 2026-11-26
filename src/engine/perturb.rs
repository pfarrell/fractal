//! Deep Mandelbrot rendering by perturbation.
//!
//! One reference orbit Z₀, Z₁, … is iterated in high precision at the
//! view's centre C and stored as f64. Each pixel c = C + dc then follows
//! only its small difference δ from that orbit, in plain f64:
//!
//! δₙ₊₁ = 2·Zₙ·δₙ + δₙ² + dc,   zₙ = Zₙ + δₙ.
//!
//! When the pixel's orbit comes closer to 0 than its difference from the
//! reference (|z| < |δ|), or the reference runs out (it escaped first),
//! the pixel *rebases*: it restarts from the reference's start with
//! δ = z (Z₀ = 0, so that is the same point). That keeps δ small without
//! glitch detection or extra references.

use crate::engine::deep::{HiComplex, bits_for};
use crate::engine::escape::{BAILOUT, FractalKind, Sample, smooth_value};
use crate::engine::movie::{RotatedSampler, rotated_offset};
use crate::engine::render::{F64Sampler, Sampler};
use crate::engine::viewport::{Viewport, min_pixel_scale};
use num_complex::Complex64;
use std::sync::{Arc, OnceLock};

/// Perturbation takes over this far (in pixel spacing) before f64 runs
/// out. It costs about twice f64's time per pixel, so f64 keeps every view
/// it can still place: the switch sits near ×10¹¹ at 1080 px. (f64 already
/// misplaces a few percent of boundary pixels there, as it always has.)
pub const SWITCH: f64 = 10.0;

/// `view` of `kind` is rendered by perturbation.
pub fn uses_perturbation(kind: &FractalKind, view: &Viewport) -> bool {
    matches!(kind, FractalKind::Mandelbrot)
        && view.pixel_scale() < SWITCH * min_pixel_scale(view.center.to_c64())
}

/// The reference orbit Z₀ = 0, Z₁, …, Z_k as f64: Z_k is the first value
/// past the bailout (kept so even a centre that escapes at once leaves a
/// step to perturb from), or k = max_iter.
#[derive(Clone, Debug)]
pub struct ReferenceOrbit {
    pub z: Arc<[Complex64]>,
}

impl ReferenceOrbit {
    pub fn compute(c: &HiComplex, max_iter: u32, bits: usize) -> ReferenceOrbit {
        let c = c.with_bits(bits);
        let mut z = HiComplex::from_c64(Complex64::new(0.0, 0.0)).with_bits(bits);
        let mut path = Vec::with_capacity(max_iter as usize + 1);
        path.push(Complex64::new(0.0, 0.0));
        for _ in 0..max_iter {
            z = z.mul(&z).add(&c);
            let zf = z.to_c64();
            path.push(zf);
            // NaN (overflowed) counts as escaped too.
            if zf.norm_sqr() > BAILOUT * BAILOUT || zf.norm_sqr().is_nan() {
                break;
            }
        }
        ReferenceOrbit { z: path.into() }
    }
}

/// Mandelbrot samples by perturbation about the centre of the view it
/// was built for; it must only be used with views of that centre and span.
///
/// The reference orbit is computed on first use, so building a sampler is
/// cheap on the UI thread and the work lands on the render worker.
#[derive(Debug)]
pub struct PerturbSampler {
    center: HiComplex,
    bits: usize,
    orbit: OnceLock<ReferenceOrbit>,
    max_iter: u32,
    angle: f64,
    span: f64,
}

impl PerturbSampler {
    /// Reference at `view`'s centre; the picture turned by `angle`.
    pub fn new(view: &Viewport, max_iter: u32, angle: f64) -> Self {
        PerturbSampler {
            center: view.center.clone(),
            bits: bits_for(view.pixel_scale()),
            orbit: OnceLock::new(),
            max_iter,
            angle,
            span: view.span,
        }
    }

    fn orbit(&self) -> &ReferenceOrbit {
        self.orbit
            .get_or_init(|| ReferenceOrbit::compute(&self.center, self.max_iter, self.bits))
    }

    /// How many reference values there are (Z₀ included).
    pub fn reference_len(&self) -> usize {
        self.orbit().z.len()
    }

    /// One pixel, `dc` away from the reference point.
    fn iterate(&self, dc: Complex64) -> Sample {
        let zs = &self.orbit().z;
        let last = zs.len() - 1;
        let bail = BAILOUT * BAILOUT;
        let mut d = Complex64::new(0.0, 0.0);
        let mut m = 0;
        for n in 1..=self.max_iter {
            d = 2.0 * zs[m] * d + d * d + dc;
            m += 1;
            let z = zs[m] + d;
            let r = z.norm_sqr();
            if r > bail {
                return Sample::Escaped {
                    iter: n,
                    smooth: smooth_value(n, r.sqrt(), 2),
                };
            }
            if r < d.norm_sqr() || m == last {
                d = z;
                m = 0;
            }
        }
        Sample::Inside
    }
}

impl Sampler for PerturbSampler {
    fn sample(&self, view: &Viewport, x: usize, y: usize) -> Sample {
        // Built for this view's centre and span; a lower-resolution pass
        // of the same view (same centre and span) is fine.
        debug_assert_eq!(view.span, self.span);
        self.iterate(rotated_offset(
            view,
            self.angle,
            x as f64 + 0.5,
            y as f64 + 0.5,
        ))
    }
}

/// The sampler for `view` of `kind`, turned by `angle`: perturbation past
/// the switch-over, the plain f64 samplers otherwise.
pub fn sampler_for(
    kind: &FractalKind,
    view: &Viewport,
    max_iter: u32,
    angle: f64,
) -> Arc<dyn Sampler> {
    if uses_perturbation(kind, view) {
        Arc::new(PerturbSampler::new(view, max_iter, angle))
    } else if angle == 0.0 {
        Arc::new(F64Sampler {
            kind: *kind,
            max_iter,
        })
    } else {
        Arc::new(RotatedSampler {
            kind: *kind,
            max_iter,
            angle,
        })
    }
}
