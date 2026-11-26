use fractal_explorer::engine::deep::{HiComplex, bits_for};
use fractal_explorer::engine::escape::{BAILOUT, FractalKind, Sample};
use fractal_explorer::engine::movie::RotatedSampler;
use fractal_explorer::engine::perturb::{PerturbSampler, uses_perturbation};
use fractal_explorer::engine::render::{F64Sampler, Sampler, render_serial};
use fractal_explorer::engine::viewport::{Viewport, auto_max_iter};
use num_complex::Complex64;

const SEAHORSE: Complex64 = Complex64::new(-0.743643887, 0.131825904);

fn iter_of(s: &Sample) -> Option<u32> {
    match s {
        Sample::Escaped { iter, .. } => Some(*iter),
        _ => None,
    }
}

/// Fraction of pixels where `a` and `b` agree: same variant, escape
/// counts within ±1.
fn agreement(a: &[Sample], b: &[Sample]) -> f64 {
    assert_eq!(a.len(), b.len());
    let ok = a
        .iter()
        .zip(b)
        .filter(|(x, y)| match (iter_of(x), iter_of(y)) {
            (Some(p), Some(q)) => p.abs_diff(q) <= 1,
            (None, None) => true,
            _ => false,
        })
        .count();
    ok as f64 / a.len() as f64
}

/// Every pixel iterated in high precision: slow but exact.
fn slow_render(view: &Viewport, max_iter: u32) -> Vec<Sample> {
    let bits = bits_for(view.pixel_scale());
    let mut out = Vec::new();
    for y in 0..view.height {
        for x in 0..view.width {
            let c = view
                .pixel_to_hi(x as f64 + 0.5, y as f64 + 0.5)
                .with_bits(bits);
            let mut z = HiComplex::from_c64(Complex64::new(0.0, 0.0)).with_bits(bits);
            let mut s = Sample::Inside;
            for n in 1..=max_iter {
                z = z.mul(&z).add(&c);
                if z.norm_sqr_f64() > BAILOUT * BAILOUT {
                    s = Sample::Escaped {
                        iter: n,
                        smooth: 0.0,
                    };
                    break;
                }
            }
            out.push(s);
        }
    }
    out
}

fn perturbed(view: &Viewport, max_iter: u32, angle: f64) -> Vec<Sample> {
    render_serial(&PerturbSampler::new(view, max_iter, angle), view)
}

/// Just past the switch-over: perturbation matches exact high-precision
/// iteration, and f64 as closely as f64 is right itself (chaotic boundary
/// pixels differ by a few iterations from rounding alone).
#[test]
fn agrees_with_f64_past_switch() {
    let view = Viewport::new(SEAHORSE, 3e-8, 64, 48);
    assert!(uses_perturbation(&FractalKind::Mandelbrot, &view));
    let max_iter = auto_max_iter(view.magnification());
    let f = render_serial(
        &F64Sampler {
            kind: FractalKind::Mandelbrot,
            max_iter,
        },
        &view,
    );
    let p = perturbed(&view, max_iter, 0.0);
    let slow = slow_render(&view, max_iter);
    let exact = agreement(&p, &slow);
    assert!(exact >= 0.999, "perturbation vs exact {exact}");
    let a = agreement(&p, &f);
    assert!(a >= 0.99, "perturbation vs f64 {a}");
}

#[test]
fn rotation_matches_rotated_sampler() {
    let view = Viewport::new(SEAHORSE, 3e-8, 64, 48);
    let max_iter = auto_max_iter(view.magnification());
    let f = render_serial(
        &RotatedSampler {
            kind: FractalKind::Mandelbrot,
            max_iter,
            angle: 0.7,
        },
        &view,
    );
    let a = agreement(&perturbed(&view, max_iter, 0.7), &f);
    assert!(a >= 0.99, "agreement {a}");
}

#[test]
fn deep_matches_slow_reference() {
    // c = i is a Misiurewicz point: structure at every depth.
    let view = Viewport::new(Complex64::new(0.0, 1.0), 3e-30, 32, 24);
    assert!(uses_perturbation(&FractalKind::Mandelbrot, &view));
    let max_iter = auto_max_iter(1e30);
    let slow = slow_render(&view, max_iter);
    let mut counts: Vec<u32> = slow.iter().filter_map(iter_of).collect();
    counts.sort();
    counts.dedup();
    assert!(counts.len() >= 5, "too plain to test: {counts:?}");
    let a = agreement(&perturbed(&view, max_iter, 0.0), &slow);
    assert!(a >= 0.99, "agreement {a}");
}

#[test]
fn rebasing_carries_pixels_past_reference() {
    let center = HiComplex::from_c64(Complex64::new(0.0, 1.0))
        .add_c64(Complex64::new(4e-11, 0.0), bits_for(1e-13));
    let view = Viewport::new(center, 3e-10, 48, 36);
    let max_iter = 2000;
    // The stored orbit ends with the escaping value: Z₀ … Z_k, k = escape.
    let reference = PerturbSampler::new(&view, max_iter, 0.0).reference_len() - 1;
    assert!(reference < 100, "centre escapes after {reference}");
    let slow = slow_render(&view, max_iter);
    let longer = slow
        .iter()
        .filter(|s| iter_of(s).is_none_or(|n| n as usize > reference))
        .count();
    assert!(
        longer * 10 >= slow.len(),
        "only {longer} pixels outlive the reference"
    );
    let a = agreement(&perturbed(&view, max_iter, 0.0), &slow);
    assert!(a >= 0.99, "agreement {a}");
}

#[test]
fn shallow_and_other_kinds_use_f64() {
    let shallow = Viewport::new(Complex64::new(-0.5, 0.0), 3.0, 160, 120);
    assert!(!uses_perturbation(&FractalKind::Mandelbrot, &shallow));
    let deep = Viewport::new(SEAHORSE, 1e-14, 160, 120);
    let before = Viewport::new(SEAHORSE, 3e-3, 160, 120);
    assert!(!uses_perturbation(&FractalKind::Mandelbrot, &before));
    assert!(!uses_perturbation(&FractalKind::BurningShip, &deep));
    assert!(uses_perturbation(&FractalKind::Mandelbrot, &deep));
}

#[test]
fn sampler_for_picks_the_f64_path_when_shallow() {
    use fractal_explorer::engine::perturb::sampler_for;
    let view = Viewport::new(SEAHORSE, 1e-3, 40, 30);
    let s = sampler_for(&FractalKind::Mandelbrot, &view, 500, 0.0);
    let plain = F64Sampler {
        kind: FractalKind::Mandelbrot,
        max_iter: 500,
    };
    assert!(render_serial(&*s, &view) == render_serial(&plain, &view));
    let _: &dyn Sampler = &*s;
}

#[test]
fn centre_outside_bailout_does_not_panic() {
    let view = Viewport::new(Complex64::new(300.0, 0.0), 1e-20, 16, 12);
    let out = perturbed(&view, 500, 0.0);
    assert!(out.iter().all(|s| iter_of(s) == Some(1)), "{:?}", &out[..3]);
}

#[test]
fn building_a_sampler_defers_the_orbit() {
    // c = -1 is inside the set: the reference never escapes.
    let view = Viewport::new(Complex64::new(-1.0, 0.0), 3e-250, 16, 12);
    let t = std::time::Instant::now();
    let s = PerturbSampler::new(&view, 2_000_000, 0.0);
    assert!(t.elapsed().as_millis() < 50, "{:?}", t.elapsed());
    drop(s);
}
