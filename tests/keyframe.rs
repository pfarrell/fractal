use fractal_explorer::engine::color::{ColorSettings, Palette, colorize};
use fractal_explorer::engine::escape::{FractalKind, Sample};
use fractal_explorer::engine::keyframe::{
    Block, KEYFRAME_MAX_PIXELS, Keyframe, Quality, Stretch, TAP_COST, estimated_cost, frame_work,
    keyframe_work, linear_to_srgb, pack, plan, plan_range, planned_work, render_keyframe, resample,
    srgb_to_linear, tap_color,
};
use fractal_explorer::engine::movie::{MovieSpec, RotatedSampler};
use fractal_explorer::engine::render::{CancelToken, F64Sampler, render};
use fractal_explorer::engine::route::Stop;
use fractal_explorer::engine::viewport::{MaxIter, Viewport, min_pixel_scale};
use num_complex::Complex64;
use std::path::PathBuf;

const TARGET: Complex64 = Complex64::new(-0.743643887, 0.131825904);

/// A two-stop route: start view to end view, turning at `rotation` °/s.
fn two(c0: Complex64, s0: f64, c1: Complex64, s1: f64, rotation: f64) -> Vec<Stop> {
    vec![
        Stop {
            center: c0,
            span: s0,
            rotation: 0.0,
        },
        Stop {
            center: c1,
            span: s1,
            rotation,
        },
    ]
}

fn base(quality: Quality) -> MovieSpec {
    MovieSpec {
        kind: FractalKind::Mandelbrot,
        stops: two(Complex64::new(-0.5, 0.0), 3.0, TARGET, 3e-9, 0.0),
        fps: 30,
        seconds: 30.0,
        width: 160,
        height: 90,
        max_iter: MaxIter::Auto,
        color: ColorSettings::default(),
        output: PathBuf::from("/tmp/unused.mp4"),
        quality,
        music: Vec::new(),
    }
}

/// The motions every planner property is checked over.
fn specs(quality: Quality) -> Vec<(&'static str, MovieSpec)> {
    let zoom_in = base(quality);
    let zoom_out = MovieSpec {
        stops: two(TARGET, 3e-9, Complex64::new(-0.5, 0.0), 3.0, 0.0),
        ..base(quality)
    };
    let pan = MovieSpec {
        stops: two(
            Complex64::new(-1.5, -0.8),
            0.01,
            Complex64::new(0.4, 0.9),
            0.01,
            0.0,
        ),
        ..base(quality)
    };
    let turning = MovieSpec {
        stops: two(Complex64::new(-0.5, 0.0), 3.0, TARGET, 3e-9, 12.0),
        ..base(quality)
    };
    let two_frames = MovieSpec {
        fps: 1,
        seconds: 1.0,
        ..base(quality)
    };
    assert_eq!(two_frames.frame_count(), 2);
    vec![
        ("zoom in", zoom_in),
        ("zoom out", zoom_out),
        ("pan", pan),
        ("turning", turning),
        ("two frames", two_frames),
    ]
}

fn stretches(blocks: &[Block]) -> impl Iterator<Item = &Stretch> {
    blocks.iter().filter_map(|b| match b {
        Block::Keyframe(s) => Some(s),
        Block::Exact(_) => None,
    })
}

fn range(b: &Block) -> std::ops::Range<usize> {
    match b {
        Block::Keyframe(s) => s.frames.clone(),
        Block::Exact(r) => r.clone(),
    }
}

fn pixels(v: &Viewport) -> usize {
    v.width * v.height
}

#[test]
fn blocks_cover_every_frame_once() {
    for q in [1, 2] {
        for (name, spec) in specs(Quality::Smooth) {
            let blocks = plan(&spec, q, KEYFRAME_MAX_PIXELS);
            let mut next = 0;
            for b in &blocks {
                let r = range(b);
                assert_eq!(r.start, next, "{name}: gap or overlap at {next}");
                assert!(r.end > r.start, "{name}: empty block");
                next = r.end;
            }
            assert_eq!(next, spec.frame_count(), "{name}: frames left over");
        }
    }
}

#[test]
fn stretches_obey_span_ratio_and_size() {
    for q in [1, 2] {
        for (name, spec) in specs(Quality::Smooth) {
            for s in stretches(&plan(&spec, q, KEYFRAME_MAX_PIXELS)) {
                let spans: Vec<f64> = s.frames.clone().map(|i| spec.frame_view(i).span).collect();
                let max = spans.iter().cloned().fold(f64::MIN, f64::max);
                let min = spans.iter().cloned().fold(f64::MAX, f64::min);
                assert!(max / min <= 2.0 + 1e-12, "{name}: span ratio {}", max / min);
                assert!(
                    pixels(&s.view) <= KEYFRAME_MAX_PIXELS,
                    "{name}: keyframe too big"
                );
            }
        }
    }
}

#[test]
fn every_frame_lies_inside_its_keyframe() {
    for q in [1, 2] {
        for (name, spec) in specs(Quality::Smooth) {
            for s in stretches(&plan(&spec, q, KEYFRAME_MAX_PIXELS)) {
                let v = &s.view;
                let k = v.pixel_scale();
                let half_w = v.width as f64 * k / 2.0;
                let half_h = v.height as f64 * k / 2.0;
                let (left, right) = (v.center.re - half_w + k, v.center.re + half_w - k);
                let (bottom, top) = (v.center.im - half_h + k, v.center.im + half_h - k);
                for i in s.frames.clone() {
                    for c in spec.frame_corners(i) {
                        assert!(
                            (left..=right).contains(&c.re) && (bottom..=top).contains(&c.im),
                            "{name} q={q}: frame {i} corner {c} outside keyframe {v:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn frame_corners_follow_rotation() {
    let spec = MovieSpec {
        stops: two(Complex64::new(-0.5, 0.0), 3.0, TARGET, 3e-9, 90.0),
        seconds: 1.0,
        ..base(Quality::Exact)
    };
    let last = spec.frame_count() - 1;
    let v = spec.frame_view(last);
    let [tl, tr, br, bl] = spec.frame_corners(last);
    // A quarter turn counterclockwise: the top-left corner swings to
    // bottom-left, so it sits below and left of the center.
    let (hw, hh) = (
        v.span * v.width as f64 / v.height as f64 / 2.0,
        v.span / 2.0,
    );
    let near = |a: Complex64, b: Complex64| (a - b).norm() < 1e-9;
    assert!(near(tl, v.center + Complex64::new(-hh, -hw)), "{tl}");
    assert!(near(tr, v.center + Complex64::new(-hh, hw)), "{tr}");
    assert!(near(br, v.center + Complex64::new(hh, hw)), "{br}");
    assert!(near(bl, v.center + Complex64::new(hh, -hw)), "{bl}");
}

#[test]
fn deepest_frame_gets_q_samples_per_axis() {
    for q in [1, 2] {
        for (name, spec) in specs(Quality::Smooth) {
            for s in stretches(&plan(&spec, q, KEYFRAME_MAX_PIXELS)) {
                let deepest = s
                    .frames
                    .clone()
                    .map(|i| spec.frame_view(i).pixel_scale())
                    .fold(f64::MAX, f64::min);
                let ratio = deepest / s.view.pixel_scale();
                assert!(ratio >= q as f64 - 1e-9, "{name} q={q}: {ratio}");
                assert_eq!(s.q, q);
            }
        }
    }
}

#[test]
fn stretch_iterations_are_the_deepest_frames() {
    for (name, spec) in specs(Quality::Smooth) {
        for s in stretches(&plan(&spec, 2, KEYFRAME_MAX_PIXELS)) {
            let deepest = s
                .frames
                .clone()
                .map(|i| spec.max_iter.resolve(spec.frame_view(i).magnification()))
                .max()
                .unwrap();
            assert_eq!(s.max_iter, deepest, "{name}");
        }
    }
}

#[test]
fn density_clamped_at_precision_floor() {
    let spec = MovieSpec {
        stops: two(
            Complex64::new(-0.5, 0.0),
            3.0,
            TARGET,
            min_pixel_scale(TARGET) * 90.0,
            0.0,
        ),
        ..base(Quality::Smooth)
    };
    let blocks = plan(&spec, 2, KEYFRAME_MAX_PIXELS);
    for s in stretches(&blocks) {
        let floor = min_pixel_scale(s.view.center);
        assert!(
            s.view.pixel_scale() >= floor * (1.0 - 1e-9),
            "{:?}: {} < floor {floor}",
            s.frames,
            s.view.pixel_scale()
        );
    }
    assert!(stretches(&blocks).count() > 0);
}

#[test]
fn oversized_frame_lowers_q_then_goes_exact() {
    for (name, spec) in specs(Quality::Smooth) {
        let spec = MovieSpec {
            width: 7680,
            height: 4320,
            fps: 5,
            seconds: spec.seconds.min(10.0),
            ..spec
        };
        let blocks = plan(&spec, 2, KEYFRAME_MAX_PIXELS);
        assert!(!blocks.is_empty());
        for s in stretches(&blocks) {
            assert_eq!(s.q, 1, "{name}: q not lowered");
            assert!(pixels(&s.view) <= KEYFRAME_MAX_PIXELS, "{name}: over cap");
        }

        let tiny = spec.width * spec.height - 1;
        let blocks = plan(&spec, 1, tiny);
        assert_eq!(blocks.len(), 1, "{name}: Exact blocks not merged");
        assert_eq!(range(&blocks[0]), 0..spec.frame_count(), "{name}");
        assert!(matches!(blocks[0], Block::Exact(_)), "{name}");
    }
}

fn hd(seconds: f64, fps: u32) -> MovieSpec {
    // A 1e13 zoom, but no deeper than the precision floor allows at 1080p.
    let end_span = (3.0 / 1e13_f64).max(min_pixel_scale(TARGET) * 1080.0);
    MovieSpec {
        width: 1920,
        height: 1080,
        seconds,
        fps,
        stops: two(Complex64::new(-0.5, 0.0), 3.0, TARGET, end_span, 0.0),
        ..base(Quality::Smooth)
    }
}

#[test]
fn cost_is_low_for_slow_zooms_and_high_for_fast() {
    // Resampling alone is ~0.15 of Exact (about 12 taps a pixel), so even
    // a slow zoom is only several times faster.
    let slow = hd(600.0, 60);
    let cost = estimated_cost(&slow);
    assert!(cost < 0.25, "slow zoom costs {cost}");

    let fast = hd(3.0, 30);
    let cost = estimated_cost(&fast);
    assert!(cost > 1.0, "fast zoom costs {cost}");

    let exact = MovieSpec {
        quality: Quality::Exact,
        ..hd(3.0, 30)
    };
    assert_eq!(estimated_cost(&exact), 1.0);
}

#[test]
fn cost_counts_resampling() {
    // However many frames share a keyframe, each still costs its taps: a
    // Smooth frame takes at least 2×2 per pixel.
    for sp in [hd(600.0, 60), hd(3600.0, 120)] {
        let cost = estimated_cost(&sp);
        assert!(cost >= 4.0 * TAP_COST, "{} s costs {cost}", sp.seconds);
    }
}

#[test]
fn planned_work_adds_up_to_the_estimate() {
    for (name, sp) in specs(Quality::Smooth) {
        let blocks = plan(&sp, 2, KEYFRAME_MAX_PIXELS);
        let frames: f64 = blocks
            .iter()
            .flat_map(|b| range(b).map(|i| frame_work(&sp, b, i)).collect::<Vec<_>>())
            .sum();
        let keyframes: f64 = stretches(&blocks).map(keyframe_work).sum();
        let total = planned_work(&sp, &blocks);
        assert!((frames + keyframes - total).abs() <= 1e-9 * total, "{name}");
        let exact = (sp.width * sp.height * sp.frame_count()) as f64;
        assert!(
            (total / exact - estimated_cost(&sp)).abs() < 1e-12,
            "{name}"
        );
    }
}

#[test]
fn quality_q_and_labels() {
    assert_eq!(Quality::default(), Quality::Exact);
    assert_eq!(Quality::Exact.q(), None);
    assert_eq!(Quality::Fast.q(), Some(1));
    assert_eq!(Quality::Smooth.q(), Some(2));
    let labels: Vec<_> = Quality::ALL.iter().map(Quality::label).collect();
    assert_eq!(labels, ["Exact", "Fast", "Smooth"]);
}

#[test]
fn old_specs_load_as_exact() {
    let mut json = serde_json::to_value(base(Quality::Smooth)).unwrap();
    json.as_object_mut().unwrap().remove("quality");
    let spec: MovieSpec = serde_json::from_value(json).unwrap();
    assert_eq!(spec.quality, Quality::Exact);
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().build().unwrap()
}

fn plain_render(kind: FractalKind, view: &Viewport, max_iter: u32) -> Vec<Sample> {
    let sampler = F64Sampler { kind, max_iter };
    render(&sampler, view, &pool(), &CancelToken::never()).unwrap()
}

fn resampled(kf: &Keyframe, frame: &Viewport, angle: f64, max_iter: u32) -> Vec<[u8; 4]> {
    let mut out = vec![[0u8; 4]; frame.width * frame.height];
    let color = ColorSettings::default();
    resample(kf, frame, angle, max_iter, &color, &pool(), &mut out);
    out
}

/// Mean absolute difference per color channel (alpha ignored).
fn mean_abs_diff(a: &[[u8; 4]], b: &[[u8; 4]]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: u64 = a
        .iter()
        .zip(b)
        .flat_map(|(p, q)| (0..3).map(move |c| p[c].abs_diff(q[c]) as u64))
        .sum();
    sum as f64 / (a.len() * 3) as f64
}

const SEAHORSE: Complex64 = Complex64::new(-0.7453, 0.1127);

#[test]
fn keyframe_matches_a_plain_render() {
    // Two full bands and a partial one, at 256 rows a band and (wide
    // enough that a band is held to a 1080p frame's samples) at 172.
    for (width, height) in [(173, 600), (12_000, 400)] {
        let view = Viewport::new(Complex64::new(-0.74, 0.12), 0.05, width, height);
        let s = Stretch {
            frames: 0..1,
            view,
            q: 2,
            max_iter: 300,
        };
        let kind = FractalKind::Mandelbrot;
        let kf = render_keyframe(kind, &s, &pool(), &CancelToken::never()).unwrap();
        assert_eq!(kf.view, view);
        let want: Vec<_> = plain_render(kind, &view, 300)
            .into_iter()
            .map(pack)
            .collect();
        assert!(
            kf.data == want,
            "{width}×{height}: banded keyframe differs from a plain render"
        );
    }
}

#[test]
fn same_grid_resample_is_exact() {
    let frame = Viewport::new(SEAHORSE, 0.02, 96, 54);
    let s = Stretch {
        frames: 0..1,
        view: frame,
        q: 1,
        max_iter: 400,
    };
    let kind = FractalKind::Mandelbrot;
    let kf = render_keyframe(kind, &s, &pool(), &CancelToken::never()).unwrap();
    let got = resampled(&kf, &frame, 0.0, 400);
    let samples = plain_render(kind, &frame, 400);
    let color = ColorSettings::default();
    // Packing keeps the smooth fraction to a 256th of an iteration, which
    // can move a channel by one; the resampling itself adds nothing.
    let mut want = vec![[0u8; 4]; 96 * 54];
    colorize(&samples, &color, &mut want);
    for (g, w) in got.iter().zip(&want) {
        assert!((0..4).all(|c| g[c].abs_diff(w[c]) <= 1), "{g:?} vs {w:?}");
    }
    let packed: Vec<_> = samples.iter().map(|&s| pack(s).cut(400)).collect();
    colorize(&packed, &color, &mut want);
    assert!(got == want);
}

fn uniform_keyframe(view: Viewport, f: impl Fn(usize, usize) -> Sample) -> Keyframe {
    let data = (0..view.height)
        .flat_map(|y| (0..view.width).map(move |x| (x, y)))
        .map(|(x, y)| pack(f(x, y)))
        .collect();
    Keyframe { view, data }
}

#[test]
fn constant_keyframe_stays_constant() {
    let escaped = Sample::Escaped {
        iter: 37,
        smooth: 37.25,
    };
    let kf = uniform_keyframe(
        Viewport::new(Complex64::new(0.0, 0.0), 2.0, 120, 100),
        |_, _| escaped,
    );
    let color = ColorSettings::default();
    let want = tap_color(pack(escaped), 1000, &color, &Palette::get(color.palette));
    for span in [0.3, 1.0, 1.9] {
        for angle in [0.0, 0.4, std::f64::consts::FRAC_PI_2, 2.5] {
            let frame = Viewport::new(Complex64::new(0.1, -0.05), span, 40, 30);
            let out = resampled(&kf, &frame, angle, 1000);
            assert!(
                out.iter().all(|&p| p == want),
                "span {span} angle {angle}: not constant"
            );
        }
    }
}

#[test]
fn quarter_turn_rotates_the_picture() {
    let view = Viewport::new(Complex64::new(0.0, 0.0), 2.0, 100, 100);
    let kf = uniform_keyframe(view, |x, _| {
        if x < 50 {
            Sample::Inside
        } else {
            Sample::Escaped {
                iter: 5,
                smooth: 5.5,
            }
        }
    });
    let out = resampled(&kf, &view, std::f64::consts::FRAC_PI_2, 1000);
    let black = [0, 0, 0, 255];
    // Turned a quarter counterclockwise (as `rotated_point` and the Exact
    // path do), a frame's top row looks at the plane's left side: the
    // Inside half ends up on top, the colored half below.
    for y in (0..45).chain(55..100) {
        for x in 0..100 {
            let p = out[y * 100 + x];
            if y < 45 {
                assert_eq!(p, black, "({x}, {y}) should be black");
            } else {
                assert_ne!(p, black, "({x}, {y}) should be colored");
            }
        }
    }
}

#[test]
fn frame_cutoff_applies() {
    let view = Viewport::new(Complex64::new(0.0, 0.0), 2.0, 64, 64);
    let kf = uniform_keyframe(view, |_, _| Sample::Escaped {
        iter: 800,
        smooth: 800.5,
    });
    let frame = Viewport::new(Complex64::new(0.0, 0.0), 1.0, 32, 18);
    let out = resampled(&kf, &frame, 0.3, 500);
    assert!(out.iter().all(|&p| p == [0, 0, 0, 255]));
    let out = resampled(&kf, &frame, 0.3, 800);
    assert!(out.iter().all(|&p| p != [0, 0, 0, 255]));
}

/// A short zoom into seahorse valley at 96×54.
fn seahorse(quality: Quality, rotation: f64) -> MovieSpec {
    MovieSpec {
        stops: two(SEAHORSE, 0.05, SEAHORSE, 0.002, rotation),
        fps: 10,
        seconds: 2.0,
        width: 96,
        height: 54,
        ..base(quality)
    }
}

/// Frame `i` rendered Exact at 8×8 samples per pixel, averaged in linear light.
fn supersampled(spec: &MovieSpec, i: usize) -> Vec<[u8; 4]> {
    supersampled_by(spec, i, 8)
}

/// Frame `i` rendered Exact at `N`×`N` samples per pixel, averaged in
/// linear light. `N` = 1 is the Exact movie frame itself.
#[allow(non_snake_case)]
fn supersampled_by(spec: &MovieSpec, i: usize, N: usize) -> Vec<[u8; 4]> {
    let view = spec.frame_view(i);
    let big = view.with_size(view.width * N, view.height * N);
    let sampler = RotatedSampler {
        kind: spec.kind,
        max_iter: spec.max_iter.resolve(view.magnification()),
        angle: spec.frame_angle(i),
    };
    let samples = render(&sampler, &big, &pool(), &CancelToken::never()).unwrap();
    let mut colors = vec![[0u8; 4]; samples.len()];
    colorize(&samples, &spec.color, &mut colors);
    let mut out = vec![[0u8; 4]; view.width * view.height];
    for (k, px) in out.iter_mut().enumerate() {
        let (x, y) = (k % view.width, k / view.width);
        let mut sum = [0f32; 3];
        for b in 0..N {
            for a in 0..N {
                let c = colors[(y * N + b) * big.width + x * N + a];
                for ch in 0..3 {
                    sum[ch] += srgb_to_linear(c[ch]);
                }
            }
        }
        let n2 = (N * N) as f32;
        *px = [
            linear_to_srgb(sum[0] / n2),
            linear_to_srgb(sum[1] / n2),
            linear_to_srgb(sum[2] / n2),
            255,
        ];
    }
    out
}

/// Frame `i` resampled from stretch `s`'s keyframe.
fn from_keyframe(spec: &MovieSpec, s: &Stretch, i: usize) -> Vec<[u8; 4]> {
    let kf = render_keyframe(spec.kind, s, &pool(), &CancelToken::never()).unwrap();
    let view = spec.frame_view(i);
    let max_iter = spec.max_iter.resolve(view.magnification());
    resampled(&kf, &view, spec.frame_angle(i), max_iter)
}

#[test]
fn smooth_agrees_with_supersampled_exact() {
    let spec = seahorse(Quality::Smooth, 20.0);
    let blocks = plan(&spec, 2, KEYFRAME_MAX_PIXELS);
    let s = stretches(&blocks).next().unwrap();
    for i in [s.frames.start, s.frames.end - 1] {
        let err = mean_abs_diff(&from_keyframe(&spec, s, i), &supersampled(&spec, i));
        assert!(err < 0.03 * 255.0, "frame {i}: mean abs error {err}");
    }
}

#[test]
fn no_pulse_between_keyframes() {
    // At every stretch boundary, switching keyframes must jump less than
    // the shimmer Exact already has on that frame: its own 1×1 frame's
    // error against the 8×8 reference. Fast (1 tap per keyframe sample at
    // the deepest frame) measured 1.02–1.08× Exact's error, so it gets 1.5×.
    for (quality, slack) in [(Quality::Smooth, 1.0), (Quality::Fast, 1.5)] {
        let spec = seahorse(quality, 0.0);
        let blocks = plan(&spec, quality.q().unwrap(), KEYFRAME_MAX_PIXELS);
        let all: Vec<_> = stretches(&blocks).collect();
        assert!(all.len() >= 4, "{quality:?}: only {} stretches", all.len());
        for pair in all.windows(2) {
            let (s0, s1) = (pair[0], pair[1]);
            assert_eq!(s0.frames.end, s1.frames.start);
            // The first frame of the later stretch, from the keyframe of the
            // stretch that ends there and of the one that starts there.
            let i = s1.frames.start;
            let jump = mean_abs_diff(&from_keyframe(&spec, s0, i), &from_keyframe(&spec, s1, i));
            let exact = mean_abs_diff(&supersampled_by(&spec, i, 1), &supersampled(&spec, i));
            assert!(
                jump < slack * exact,
                "{quality:?} frame {i}: switch {jump:.2} vs Exact error {exact:.2}"
            );
        }
    }
}

#[test]
fn plan_range_over_all_frames_is_plan() {
    for (name, s) in specs(Quality::Smooth) {
        let all = plan_range(&s, 0..s.frame_count(), 2, KEYFRAME_MAX_PIXELS);
        assert_eq!(all, plan(&s, 2, KEYFRAME_MAX_PIXELS), "{name}");
    }
}

#[test]
fn plan_range_covers_exactly_the_range() {
    for (name, mut s) in specs(Quality::Smooth) {
        (s.fps, s.seconds) = (30, 4.0); // 120 frames
        assert_eq!(s.frame_count(), 120);
        let blocks = plan_range(&s, 40..95, 2, KEYFRAME_MAX_PIXELS);
        let covered: Vec<usize> = blocks.iter().flat_map(range).collect();
        assert_eq!(covered, (40..95).collect::<Vec<_>>(), "{name}");
        for st in stretches(&blocks) {
            let (lo, hi) = (
                st.view.pixel_to_complex(0.0, st.view.height as f64),
                st.view.pixel_to_complex(st.view.width as f64, 0.0),
            );
            for i in st.frames.clone() {
                for c in s.frame_corners(i) {
                    assert!(
                        c.re >= lo.re.min(hi.re)
                            && c.re <= lo.re.max(hi.re)
                            && c.im >= lo.im.min(hi.im)
                            && c.im <= lo.im.max(hi.im),
                        "{name}: frame {i} corner outside its keyframe"
                    );
                }
            }
        }
    }
}

#[test]
fn plan_range_of_empty_range_is_empty() {
    let s = base(Quality::Smooth);
    assert!(plan_range(&s, 7..7, 2, KEYFRAME_MAX_PIXELS).is_empty());
}
