use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::newton::NewtonPoly;
use fractal_explorer::engine::render::{CancelToken, F64Sampler, render, render_serial};
use fractal_explorer::engine::viewport::Viewport;
use num_complex::Complex64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

fn check_equal(kind: FractalKind) {
    let view = Viewport::default_for(320, 200);
    let sampler = F64Sampler {
        kind,
        max_iter: 200,
    };
    let serial = render_serial(&sampler, &view);
    for n in [1, 4, 12] {
        let par = render(&sampler, &view, &pool(n), &CancelToken::never()).unwrap();
        assert!(par == serial, "{kind:?} with {n} threads differs");
    }
}

#[test]
fn parallel_equals_serial() {
    check_equal(FractalKind::Mandelbrot);
}

#[test]
fn parallel_equals_serial_newton_and_ship() {
    check_equal(FractalKind::Newton {
        poly: NewtonPoly::CubeRootsOfUnity,
    });
    check_equal(FractalKind::BurningShip);
}

#[test]
fn cancelled_returns_none() {
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    generation.store(1, Ordering::Relaxed);
    let sampler = F64Sampler {
        kind: FractalKind::Mandelbrot,
        max_iter: 100,
    };
    assert!(render(&sampler, &Viewport::default_for(64, 64), &pool(2), &token).is_none());
}

#[test]
fn cancel_mid_render_is_prompt() {
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    // Every pixel is inside the set: the worst case, all 20k iterations each.
    let view = Viewport::new(Complex64::new(0.0, 0.0), 0.1, 1920, 1080);
    let sampler = F64Sampler {
        kind: FractalKind::Mandelbrot,
        max_iter: 20_000,
    };
    let bumped_at = Arc::new(std::sync::Mutex::new(None));
    let (g, b) = (generation.clone(), bumped_at.clone());
    let bumper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        *b.lock().unwrap() = Some(Instant::now());
        g.store(1, Ordering::Relaxed);
    });
    let result = render(&sampler, &view, &pool(12), &token);
    let done = Instant::now();
    bumper.join().unwrap();
    assert!(result.is_none());
    let bumped = bumped_at.lock().unwrap().expect("bumper ran");
    assert!(
        done.duration_since(bumped) < Duration::from_millis(500),
        "took {:?}",
        done.duration_since(bumped)
    );
}

#[test]
fn empty_view_renders_empty() {
    let sampler = F64Sampler {
        kind: FractalKind::Mandelbrot,
        max_iter: 100,
    };
    let out = render(
        &sampler,
        &Viewport::default_for(0, 0),
        &pool(2),
        &CancelToken::never(),
    );
    assert_eq!(out.map(|v| v.len()), Some(0));
}
