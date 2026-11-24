use fractal_explorer::engine::route::{
    JOINT_HALF_WIDTH_S, Route, Stop, outside_previous, step_length,
};
use num_complex::Complex64;

fn st(re: f64, im: f64, span: f64, rotation: f64) -> Stop {
    Stop {
        center: Complex64::new(re, im),
        span,
        rotation,
    }
}

fn nested() -> Vec<Stop> {
    vec![
        st(-0.5, 0.0, 3.0, 0.0),
        st(-0.75, 0.1, 0.03, 0.0),
        st(-0.745, 0.105, 3e-5, 0.0),
    ]
}

const SECONDS: f64 = 12.0;
const FPS: f64 = 30.0;

fn frames(route: &Route, seconds: f64, n: usize) -> Vec<(Complex64, f64)> {
    (0..n)
        .map(|i| route.view_at(i as f64 / (n - 1) as f64 * seconds))
        .collect()
}

#[test]
fn step_length_of_pure_zoom_is_log_ratio() {
    let d = step_length(&st(0.0, 0.0, 1.0, 0.0), &st(0.0, 0.0, 1e-3, 0.0));
    assert!((d - 1e3f64.ln()).abs() < 1e-9, "{d}");
}

#[test]
fn longer_pan_gets_more_time() {
    let r = Route::new(
        &[
            st(0.0, 0.0, 1.0, 0.0),
            st(0.1, 0.0, 1.0, 0.0),
            st(10.1, 0.0, 1.0, 0.0),
        ],
        10.0,
        JOINT_HALF_WIDTH_S,
    );
    let t = r.step_times();
    assert_eq!(t.len(), 3);
    assert!(t[2] - t[1] > 3.0 * (t[1] - t[0]), "{t:?}");
}

#[test]
fn two_stops_match_the_straight_zoom() {
    let (c0, s0) = (Complex64::new(-0.5, 0.0), 3.0);
    let (c1, s1) = (Complex64::new(-1.25, 0.4), 1e-3);
    let r = Route::new(
        &[st(c0.re, c0.im, s0, 0.0), st(c1.re, c1.im, s1, 0.0)],
        10.0,
        JOINT_HALF_WIDTH_S,
    );
    let ratio = s1 / s0;
    let fixed = (c1 - c0 * ratio) / (1.0 - ratio);
    for i in 0..=300 {
        let u = i as f64 / 300.0;
        let span = s0 * ratio.powf(u);
        let center = fixed + (c0 - fixed) * (span / s0);
        let (c, s) = r.view_at(u * 10.0);
        assert!((s / span - 1.0).abs() < 1e-12, "frame {i}: {s} vs {span}");
        assert!((c - center).norm() < 1e-12 * span, "frame {i}");
    }
}

#[test]
fn first_and_last_views_are_exact() {
    let stops = nested();
    let r = Route::new(&stops, SECONDS, JOINT_HALF_WIDTH_S);
    assert_eq!(r.view_at(0.0), (stops[0].center, stops[0].span));
    assert_eq!(r.view_at(SECONDS), (stops[2].center, stops[2].span));
}

#[test]
fn passes_near_interior_stops() {
    let stops = nested();
    let r = Route::new(&stops, SECONDS, JOINT_HALF_WIDTH_S);
    let (c, s) = r.view_at(r.step_times()[1]);
    assert!((c - stops[1].center).norm() < 0.05 * stops[1].span);
    assert!((s / stops[1].span - 1.0).abs() < 0.05, "{s}");
}

/// Largest frame-to-frame change in velocity, in views per frame.
fn kink(route: &Route) -> f64 {
    let f = frames(route, SECONDS, (SECONDS * FPS) as usize + 1);
    let v: Vec<Complex64> = f.windows(2).map(|w| (w[1].0 - w[0].0) / w[0].1).collect();
    v.windows(2)
        .map(|w| (w[1] - w[0]).norm())
        .fold(0.0, f64::max)
}

#[test]
fn blending_removes_the_kink() {
    let stops = nested();
    let blended = kink(&Route::new(&stops, SECONDS, JOINT_HALF_WIDTH_S));
    let unblended = kink(&Route::new(&stops, SECONDS, 0.0));
    assert!(blended <= unblended / 4.0, "{blended} vs {unblended}");
}

#[test]
fn steady_zoom_rate_on_nested_route() {
    let stops = nested();
    let r = Route::new(&stops, SECONDS, JOINT_HALF_WIDTH_S);
    let n = (SECONDS * FPS) as usize + 1;
    let f = frames(&r, SECONDS, n);
    let joint = r.step_times()[1];
    let ratios: Vec<f64> = (0..n - 1)
        .filter(|&i| {
            let t = i as f64 / (n - 1) as f64 * SECONDS;
            (t - joint).abs() >= 1.0 && (t + 1.0 / FPS - joint).abs() >= 1.0
        })
        .map(|i| f[i + 1].1 / f[i].1)
        .collect();
    let max = ratios.iter().cloned().fold(f64::MIN, f64::max);
    let min = ratios.iter().cloned().fold(f64::MAX, f64::min);
    assert!(max / min < 1.01, "{min} .. {max}");
}

#[test]
fn rotation_rate_mid_step_and_continuity() {
    let mut stops = nested();
    stops[1].rotation = 10.0;
    stops[2].rotation = -20.0;
    let r = Route::new(&stops, SECONDS, JOINT_HALF_WIDTH_S);
    let t = r.step_times().to_vec();
    let rate = |at: f64| (r.angle_at(at + 1e-4) - r.angle_at(at - 1e-4)) / 2e-4;
    assert!((rate((t[0] + t[1]) / 2.0) - 10f64.to_radians()).abs() < 1e-6);
    assert!((rate((t[1] + t[2]) / 2.0) - (-20f64).to_radians()).abs() < 1e-6);
    assert!((r.angle_at(t[1] + 1e-9) - r.angle_at(t[1] - 1e-9)).abs() < 1e-6);

    let two = Route::new(
        &[st(-0.5, 0.0, 3.0, 0.0), st(-0.7, 0.1, 0.01, 6.0)],
        10.0,
        JOINT_HALF_WIDTH_S,
    );
    for i in 0..=10 {
        let at = i as f64;
        assert!((two.angle_at(at) - 6f64.to_radians() * at).abs() < 1e-12);
    }
}

#[test]
fn many_stops_in_a_short_movie_stay_finite() {
    let stops = [
        st(-0.5, 0.0, 3.0, 5.0),
        st(-0.75, 0.1, 0.3, -5.0),
        st(-0.745, 0.11, 0.03, 30.0),
        st(-0.7455, 0.112, 0.003, 0.0),
        st(-0.74555, 0.1121, 3e-4, -30.0),
    ];
    let r = Route::new(&stops, 1.0, JOINT_HALF_WIDTH_S);
    for i in 0..24 {
        let t = i as f64 / 23.0;
        let (c, s) = r.view_at(t);
        assert!(
            c.re.is_finite() && c.im.is_finite() && s.is_finite() && s > 0.0,
            "{t}"
        );
        assert!(r.angle_at(t).is_finite());
    }
    assert_eq!(r.view_at(0.0), (stops[0].center, stops[0].span));
    assert_eq!(r.view_at(1.0), (stops[4].center, stops[4].span));
}

#[test]
fn outside_previous_detects_cross_branch_steps() {
    assert!(outside_previous(
        &st(-0.75, 0.1, 0.01, 0.0),
        &st(0.3, 0.5, 1e-4, 0.0),
        64,
        48
    ));
    let n = nested();
    assert!(!outside_previous(&n[0], &n[1], 64, 48));
    assert!(!outside_previous(&n[1], &n[2], 64, 48));
}
