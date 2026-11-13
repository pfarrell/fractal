use fractal_explorer::engine::color::ColorSettings;
use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::movie::{
    MovieError, MovieProgress, MovieSpec, ffmpeg_args, render_movie, render_movie_in,
};
use fractal_explorer::engine::render::CancelToken;
use fractal_explorer::engine::viewport::MaxIter;
use num_complex::Complex64;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn spec(out: PathBuf) -> MovieSpec {
    MovieSpec {
        kind: FractalKind::Mandelbrot,
        start_center: Complex64::new(-0.5, 0.0),
        start_span: 3.0,
        target: Complex64::new(-0.743643887, 0.131825904),
        end_span: 3.0e-6,
        fps: 30,
        seconds: 10.0,
        width: 64,
        height: 48,
        max_iter: MaxIter::Auto,
        color: ColorSettings::default(),
        output: out,
    }
}

fn s() -> MovieSpec {
    spec(PathBuf::from("/tmp/unused.mp4"))
}

#[test]
fn first_frame_is_start() {
    let v = s().frame_view(0);
    assert_eq!(v.center, s().start_center);
    assert_eq!(v.span, 3.0);
    assert_eq!((v.width, v.height), (64, 48));
}

#[test]
fn last_frame_span_is_end() {
    let sp = s();
    let v = sp.frame_view(sp.frame_count() - 1);
    assert!((v.span - 3.0e-6).abs() < 1e-18);
    assert_eq!(sp.frame_count(), 300);
}

#[test]
fn constant_zoom_ratio() {
    let sp = s();
    let r0 = sp.frame_view(1).span / sp.frame_view(0).span;
    for i in 1..sp.frame_count() - 1 {
        let r = sp.frame_view(i + 1).span / sp.frame_view(i).span;
        assert!((r - r0).abs() < 1e-9 * r0, "frame {i}");
    }
}

#[test]
fn ends_near_target() {
    let sp = s();
    let last = sp.frame_view(sp.frame_count() - 1);
    let bound = (sp.start_center - sp.target).norm() * sp.end_span / sp.start_span + 1e-15;
    assert!((last.center - sp.target).norm() <= bound);
}

#[test]
fn validate_rejects_odd_size_and_bad_fps() {
    assert!(s().validate().is_ok());
    let mut odd = s();
    odd.width = 65;
    assert!(matches!(odd.validate(), Err(MovieError::Invalid(_))));
    let mut fps = s();
    fps.fps = 0;
    assert!(matches!(fps.validate(), Err(MovieError::Invalid(_))));
    let mut deep = s();
    deep.end_span = 1e-20;
    assert!(matches!(deep.validate(), Err(MovieError::Invalid(_))));
}

#[test]
fn ffmpeg_args_exact() {
    let got = ffmpeg_args(30, Path::new("/tmp/x"), Path::new("/tmp/o.mp4"));
    let want: Vec<OsString> = [
        "-y",
        "-framerate",
        "30",
        "-i",
        "/tmp/x/frame_%05d.png",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "/tmp/o.mp4",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    assert_eq!(got, want);
}

fn short(out: PathBuf, frames: u32) -> MovieSpec {
    let mut sp = spec(out);
    sp.fps = frames;
    sp.seconds = 1.0;
    sp.width = 32;
    sp.height = 32;
    sp
}

#[test]
fn missing_ffmpeg_errors_and_cleans_up() {
    let parent = tempfile::tempdir().unwrap();
    let out = parent.path().join("o.mp4");
    let r = render_movie_in(
        parent.path(),
        &short(out, 2),
        2,
        Path::new("/nonexistent/ffmpeg"),
        &CancelToken::never(),
        &mut |_| {},
    );
    assert!(matches!(r, Err(MovieError::FfmpegMissing)), "{r:?}");
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
}

#[test]
fn unwritable_output_errors() {
    let r = render_movie(
        &short(PathBuf::from("/proc/forbidden.mp4"), 2),
        2,
        Path::new("ffmpeg"),
        &CancelToken::never(),
        &mut |_| {},
    );
    assert!(
        matches!(
            r,
            Err(MovieError::Ffmpeg { .. })
                | Err(MovieError::Io(_))
                | Err(MovieError::FfmpegMissing)
        ),
        "{r:?}"
    );
}

#[test]
fn tiny_movie_end_to_end() {
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("ffmpeg not installed; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("tiny.mp4");
    let mut sp = short(out.clone(), 12);
    sp.width = 64;
    sp.height = 48;
    let mut frames = 0;
    let r = render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        &CancelToken::never(),
        &mut |p| {
            if let MovieProgress::Frame { .. } = p {
                frames += 1
            }
        },
    );
    assert_eq!(r.unwrap(), out);
    assert_eq!(frames, 12);
    assert!(std::fs::metadata(&out).unwrap().len() > 0);
}

#[test]
fn cancel_stops_early() {
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    let dir = tempfile::tempdir().unwrap();
    let r = render_movie(
        &short(dir.path().join("c.mp4"), 10),
        2,
        Path::new("ffmpeg"),
        &token,
        &mut |p| {
            if let MovieProgress::Frame { done: 1, .. } = p {
                generation.fetch_add(1, Ordering::Relaxed);
            }
        },
    );
    assert!(matches!(r, Err(MovieError::Cancelled)), "{r:?}");
}
