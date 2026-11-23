use fractal_explorer::engine::color::ColorSettings;
use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::keyframe::{
    Block, KEYFRAME_MAX_PIXELS, Quality, Stretch, estimated_cost, plan,
};
use fractal_explorer::engine::movie::{CameraPath, MovieSpec};
use fractal_explorer::engine::viewport::{MaxIter, Viewport, min_pixel_scale};
use num_complex::Complex64;
use std::path::PathBuf;

const TARGET: Complex64 = Complex64::new(-0.743643887, 0.131825904);

fn base(quality: Quality) -> MovieSpec {
    MovieSpec {
        kind: FractalKind::Mandelbrot,
        start_center: Complex64::new(-0.5, 0.0),
        start_span: 3.0,
        target: TARGET,
        end_span: 3e-9,
        fps: 30,
        seconds: 30.0,
        width: 160,
        height: 90,
        max_iter: MaxIter::Auto,
        color: ColorSettings::default(),
        output: PathBuf::from("/tmp/unused.mp4"),
        path: CameraPath::Straight,
        rotation: 0.0,
        quality,
    }
}

/// The motions every planner property is checked over.
fn specs(quality: Quality) -> Vec<(&'static str, MovieSpec)> {
    let zoom_in = base(quality);
    let zoom_out = MovieSpec {
        start_center: TARGET,
        target: Complex64::new(-0.5, 0.0),
        start_span: 3e-9,
        end_span: 3.0,
        ..base(quality)
    };
    let pan = MovieSpec {
        start_center: Complex64::new(-1.5, -0.8),
        target: Complex64::new(0.4, 0.9),
        start_span: 0.01,
        end_span: 0.01,
        ..base(quality)
    };
    let circle = MovieSpec {
        path: CameraPath::Circle,
        rotation: 12.0,
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
        ("circle", circle),
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
        rotation: 90.0,
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
        end_span: min_pixel_scale(TARGET) * 90.0,
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
        end_span,
        ..base(Quality::Smooth)
    }
}

#[test]
fn cost_is_low_for_slow_zooms_and_high_for_fast() {
    let slow = hd(600.0, 60);
    let cost = estimated_cost(&slow);
    assert!(cost < 0.1, "slow zoom costs {cost}");

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
