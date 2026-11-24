#![allow(clippy::single_range_in_vec_init)] // runs of frames are lists of ranges
use fractal_explorer::engine::color::ColorSettings;
use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::keyframe::Quality;
use fractal_explorer::engine::movie::MovieError;
use fractal_explorer::engine::movie::MovieSpec;
use fractal_explorer::engine::preview::{
    PREVIEW_FPS, PreviewMsg, fingerprint, frame_at_time, preview_size, preview_spec,
    runs_to_render, spawn_preview, window_at,
};
use fractal_explorer::engine::render::CancelToken;
use fractal_explorer::engine::route::{JOINT_HALF_WIDTH_S, Stop};
use fractal_explorer::engine::viewport::MaxIter;
use num_complex::Complex64;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc::channel;

fn st(re: f64, im: f64, span: f64, rotation: f64) -> Stop {
    Stop {
        center: Complex64::new(re, im),
        span,
        rotation,
    }
}

/// A three-stop folder path at final settings.
fn folder(seconds: f64) -> MovieSpec {
    MovieSpec {
        kind: FractalKind::Mandelbrot,
        stops: vec![
            st(-0.5, 0.0, 3.0, 0.0),
            st(-0.75, 0.1, 0.03, 10.0),
            st(-0.745, 0.105, 3e-5, -5.0),
        ],
        fps: 30,
        seconds,
        width: 1920,
        height: 1080,
        max_iter: MaxIter::Auto,
        color: ColorSettings::default(),
        output: PathBuf::from("/no/such/dir/x.mp4"),
        quality: Quality::Smooth,
    }
}

#[test]
fn preview_size_fits_the_box() {
    for ((w, h), want) in [
        ((1920, 1080), (320, 180)),
        ((1280, 720), (320, 180)),
        ((800, 600), (240, 180)),
        ((2560, 1080), (320, 136)),
        ((7680, 16), (320, 16)),
        ((16, 4320), (16, 180)),
    ] {
        let got = preview_size(w, h);
        assert_eq!(got, want, "{w}×{h}");
        assert!(got.0.is_multiple_of(2) && got.1.is_multiple_of(2));
    }
}

#[test]
fn preview_spec_keeps_the_path() {
    let f = folder(20.0);
    let p = preview_spec(&f);
    assert_eq!(
        (p.width, p.height, p.fps, p.quality),
        (320, 180, PREVIEW_FPS, Quality::Fast)
    );
    assert_eq!(p.stops, f.stops);
    assert_eq!(p.seconds, f.seconds);
    assert_eq!(p.max_iter, f.max_iter);
    assert_eq!(p.color, f.color);
    assert_eq!(p.kind, f.kind);
    assert!(p.validate_frames().is_ok());
}

#[test]
fn windows_clip_to_the_movie() {
    let long = preview_spec(&folder(1200.0));
    assert_eq!(long.frame_count(), 14_400);
    assert_eq!(window_at(&long, 0), 0..3600);
    assert_eq!(window_at(&long, 3600), 3600..7200);
    assert_eq!(window_at(&long, 14_000), 10_800..14_400);
    assert_eq!(window_at(&long, 20_000), 10_800..14_400);
    let short = preview_spec(&folder(90.0));
    assert_eq!(window_at(&short, 0), 0..1080);
    assert_eq!(window_at(&short, 500), 0..1080);
}

#[test]
fn turn_change_keeps_earlier_fingerprints() {
    let a = preview_spec(&folder(20.0));
    let mut b = a.clone();
    b.stops[2].rotation = 25.0;
    let joint = a.route().step_times()[1];
    let mut kept = 0;
    let mut changed = 0;
    for i in 0..a.frame_count() {
        let t = a.frame_time(i);
        if t < joint - JOINT_HALF_WIDTH_S {
            assert_eq!(fingerprint(&a, i), fingerprint(&b, i), "frame {i}");
            kept += 1;
        } else if t > joint + JOINT_HALF_WIDTH_S {
            assert_ne!(fingerprint(&a, i), fingerprint(&b, i), "frame {i}");
            changed += 1;
        }
    }
    assert!(kept > 10 && changed > 10, "{kept} {changed}");
}

#[test]
fn duration_change_changes_fingerprints() {
    let a = preview_spec(&folder(20.0));
    let b = preview_spec(&folder(21.0));
    assert_eq!(fingerprint(&a, 0), fingerprint(&b, 0));
    for i in 1..a.frame_count() {
        assert_ne!(fingerprint(&a, i), fingerprint(&b, i), "frame {i}");
    }
}

#[test]
fn runs_split_around_kept_frames() {
    let have = |i: usize| [2, 3, 7].contains(&i);
    assert_eq!(runs_to_render(0..10, have), vec![0..2, 4..7, 8..10]);
    assert!(runs_to_render(0..10, |_| true).is_empty());
    assert_eq!(runs_to_render(5..8, |_| false), vec![5..8]);
}

#[test]
fn frame_at_time_inverts_frame_time() {
    let p = preview_spec(&folder(20.0));
    for i in [0, 1, 17, p.frame_count() - 1] {
        assert_eq!(frame_at_time(&p, p.frame_time(i)), i);
    }
    assert_eq!(frame_at_time(&p, -3.0), 0);
    assert_eq!(frame_at_time(&p, 99.0), p.frame_count() - 1);
}

/// A tiny, short preview: 64×36, 2 s at 12 fps.
fn tiny() -> MovieSpec {
    let mut f = folder(2.0);
    (f.width, f.height) = (64, 36);
    preview_spec(&f)
}

fn collect(
    spec: MovieSpec,
    runs: Vec<std::ops::Range<usize>>,
    cancel: CancelToken,
) -> Vec<PreviewMsg> {
    let (tx, rx) = channel();
    let job = spawn_preview(spec, runs, 2, cancel, tx, || {});
    let mut msgs = Vec::new();
    for m in rx {
        let done = matches!(m, PreviewMsg::Finished { .. });
        msgs.push(m);
        if done {
            break;
        }
    }
    job.join().unwrap();
    msgs
}

#[test]
fn preview_job_renders_only_the_runs() {
    let spec = tiny();
    let (w, h) = (spec.width, spec.height);
    let cancel = CancelToken::new(Arc::new(AtomicU64::new(5)), 5);
    let msgs = collect(spec, vec![0..4, 9..12], cancel);
    let mut frames = Vec::new();
    let mut last_progress = None;
    let mut finished = None;
    for m in msgs {
        match m {
            PreviewMsg::Frame {
                generation,
                index,
                rgb,
            } => {
                assert_eq!(generation, 5);
                assert_eq!(rgb.len(), w * h * 3);
                frames.push(index);
            }
            PreviewMsg::Progress {
                generation,
                done,
                total,
            } => {
                assert_eq!(generation, 5);
                last_progress = Some((done, total));
            }
            PreviewMsg::Finished { generation, result } => {
                assert_eq!(generation, 5);
                finished = Some(result.is_ok());
            }
        }
    }
    assert_eq!(frames, vec![0, 1, 2, 3, 9, 10, 11]);
    assert_eq!(last_progress, Some((7, 7)));
    assert_eq!(finished, Some(true));
}

#[test]
fn preview_job_stops_on_cancel() {
    let cancel = CancelToken::new(Arc::new(AtomicU64::new(1)), 0);
    let msgs = collect(tiny(), vec![0..10], cancel);
    assert!(!msgs.iter().any(|m| matches!(m, PreviewMsg::Frame { .. })));
    assert!(matches!(
        msgs.last(),
        Some(PreviewMsg::Finished {
            result: Err(MovieError::Cancelled { .. }),
            ..
        })
    ));
}
