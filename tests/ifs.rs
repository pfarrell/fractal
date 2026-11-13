use fractal_explorer::engine::ifs::{IfsSystem, chaos_game};
use fractal_explorer::engine::render::CancelToken;
use num_complex::Complex64;

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

fn count_at(sys: &IfsSystem, points: u64, x: f64, y: f64) -> u32 {
    let view = sys.fit_view(400, 400);
    let d = chaos_game(sys, &view, points, 7, &pool(4), &CancelToken::never()).unwrap();
    let (px, py) = view.complex_to_pixel(Complex64::new(x, y));
    d.counts[py as usize * d.width + px as usize]
}

#[test]
fn sierpinski_hole_is_empty() {
    let s = IfsSystem::sierpinski();
    assert_eq!(count_at(&s, 2_000_000, 0.5, 0.333), 0);
    assert!(count_at(&s, 2_000_000, 0.25, 0.001) > 0);
}

#[test]
fn deterministic_for_seed_and_threads() {
    let s = IfsSystem::barnsley_fern();
    let view = s.fit_view(200, 200);
    let a = chaos_game(&s, &view, 200_000, 3, &pool(4), &CancelToken::never()).unwrap();
    let b = chaos_game(&s, &view, 200_000, 3, &pool(4), &CancelToken::never()).unwrap();
    assert_eq!(a.counts, b.counts);
}

#[test]
fn counts_bounded_by_points() {
    let s = IfsSystem::heighway_dragon();
    let view = s.fit_view(200, 200);
    let d = chaos_game(&s, &view, 100_000, 1, &pool(3), &CancelToken::never()).unwrap();
    let total: u64 = d.counts.iter().map(|&c| c as u64).sum();
    assert!(total <= 100_000 && total > 0);
}

#[test]
fn probabilities_normalized() {
    let mut s = IfsSystem::heighway_dragon();
    s.maps[0].p = 0.2;
    s.maps[1].p = 0.2;
    assert_eq!(s.normalized_probabilities(), vec![0.5, 0.5]);

    let mut s = IfsSystem::sierpinski();
    s.maps[0].p = -1.0;
    s.maps[1].p = f64::NAN;
    s.maps[2].p = 2.0;
    assert_eq!(s.normalized_probabilities(), vec![0.0, 0.0, 1.0]);

    for m in &mut s.maps {
        m.p = 0.0;
    }
    let third = 1.0 / 3.0;
    assert_eq!(s.normalized_probabilities(), vec![third, third, third]);
}

#[test]
fn bad_probabilities_still_render() {
    let mut s = IfsSystem::barnsley_fern();
    for m in &mut s.maps {
        m.p = 0.0;
    }
    let view = s.fit_view(100, 100);
    let d = chaos_game(&s, &view, 50_000, 1, &pool(2), &CancelToken::never()).unwrap();
    assert!(d.counts.iter().any(|&c| c > 0));
}

#[test]
fn fern_bounds_roughly_known() {
    let (x0, y0, x1, y1) = IfsSystem::barnsley_fern().bounds();
    assert!(
        (x0 - -2.18).abs() < 0.3 && (x1 - 2.66).abs() < 0.3,
        "x {x0}..{x1}"
    );
    assert!(y0.abs() < 0.3 && (y1 - 10.0).abs() < 0.3, "y {y0}..{y1}");
}

#[test]
fn outline_of_sierpinski_map0() {
    let s = IfsSystem::sierpinski();
    let (x0, y0, x1, y1) = s.bounds();
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let outline = s.map_outline(0);
    for (o, c) in outline.iter().zip(corners) {
        assert!((o.0 - 0.5 * c.0).abs() < 1e-12 && (o.1 - 0.5 * c.1).abs() < 1e-12);
    }
}

#[test]
fn presets_and_single_map_render() {
    assert_eq!(IfsSystem::presets().len(), 4);
    let mut s = IfsSystem::sierpinski();
    s.maps.truncate(1);
    let view = s.fit_view(50, 50);
    assert!(chaos_game(&s, &view, 10_000, 1, &pool(2), &CancelToken::never()).is_some());
}
