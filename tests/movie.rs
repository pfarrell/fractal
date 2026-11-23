use fractal_explorer::engine::color::ColorSettings;
use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::movie::{
    MovieError, MovieProgress, MovieSpec, ffmpeg_args, ffmpeg_install_hint, find_program,
    render_movie,
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

/// Distance between two points in units of the frame's pixel size.
fn pixels_apart(
    v: &fractal_explorer::engine::viewport::Viewport,
    a: Complex64,
    b: Complex64,
) -> f64 {
    (a - b).norm() / (v.span / v.height as f64)
}

#[test]
fn last_frame_is_centered_on_target() {
    let sp = s();
    let last = sp.frame_view(sp.frame_count() - 1);
    assert!(pixels_apart(&last, last.center, sp.target) < 0.01);
}

#[test]
fn off_center_bookmark_end_view_is_hit_exactly() {
    // End view far from the start view's center, as with a bookmark.
    let mut sp = s();
    sp.start_center = Complex64::new(0.0, 0.0);
    sp.target = Complex64::new(-1.25, 0.4);
    sp.end_span = 1e-9;
    let last = sp.frame_view(sp.frame_count() - 1);
    assert!(pixels_apart(&last, last.center, sp.target) < 0.01);
    assert!((last.span - 1e-9).abs() < 1e-21);
}

#[test]
fn zoom_follows_one_straight_zoom_line() {
    // Every frame is the start view scaled about one fixed point, so that
    // point sits on the same pixel in every frame.
    let mut sp = s();
    sp.start_center = Complex64::new(0.0, 0.0);
    sp.target = Complex64::new(-1.25, 0.4);
    sp.end_span = 1e-3;
    let r = sp.end_span / sp.start_span;
    let fixed = (sp.target - sp.start_center * r) / (1.0 - r);
    let first = sp.frame_view(0).complex_to_pixel(fixed);
    for i in 1..sp.frame_count() {
        let (x, y) = sp.frame_view(i).complex_to_pixel(fixed);
        assert!(
            (x - first.0).abs() < 1e-6 && (y - first.1).abs() < 1e-6,
            "frame {i}"
        );
    }
}

#[test]
fn equal_spans_pan_straight_to_target() {
    let mut sp = s();
    sp.end_span = sp.start_span;
    let n = sp.frame_count();
    let mid = sp.frame_view((n - 1) / 2);
    let last = sp.frame_view(n - 1);
    assert!(mid.center.re.is_finite() && mid.center.im.is_finite());
    assert!((last.center - sp.target).norm() < 1e-12);
    assert_eq!(last.span, sp.start_span);
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
    let got = ffmpeg_args(30, 64, 48, Path::new("/tmp/o.mp4"));
    let want: Vec<OsString> = [
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-video_size",
        "64x48",
        "-framerate",
        "30",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "/tmp/o.mp4",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    assert_eq!(got, want);
}

#[test]
fn mkv_gets_no_mp4_flags() {
    let got = ffmpeg_args(30, 64, 48, Path::new("/tmp/o.mkv"));
    assert!(!got.iter().any(|a| a == "-movflags"), "{got:?}");
    assert_eq!(got.last().unwrap(), "/tmp/o.mkv");
    let mov = ffmpeg_args(30, 64, 48, Path::new("/tmp/O.MOV"));
    assert!(mov.iter().any(|a| a == "-movflags"), "{mov:?}");
}

#[test]
fn long_durations_are_valid() {
    let mut sp = s();
    sp.seconds = 3.0 * 3600.0;
    assert!(sp.validate().is_ok());
    assert_eq!(sp.frame_count(), 30 * 3 * 3600);
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
fn missing_ffmpeg_errors_before_rendering() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("o.mp4");
    let mut frames = 0;
    let r = render_movie(
        &short(out.clone(), 2),
        2,
        Path::new("/nonexistent/ffmpeg"),
        &CancelToken::never(),
        &mut |p| {
            if let MovieProgress::Frame { .. } = p {
                frames += 1
            }
        },
    );
    assert!(matches!(r, Err(MovieError::FfmpegMissing)), "{r:?}");
    assert_eq!(frames, 0, "no frames rendered without an encoder");
    assert!(!out.exists());
}

#[test]
fn encoder_dying_mid_stream_is_reported() {
    // `false` exits at once with status 1 and never reads its input.
    let dir = tempfile::tempdir().unwrap();
    let r = render_movie(
        &short(dir.path().join("o.mp4"), 30),
        2,
        Path::new("false"),
        &CancelToken::never(),
        &mut |_| {},
    );
    assert!(
        matches!(
            r,
            Err(MovieError::Ffmpeg {
                status: Some(1),
                ..
            })
        ),
        "{r:?}"
    );
}

fn have_ffmpeg() -> bool {
    let ok = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok();
    if !ok {
        eprintln!("ffmpeg not installed; skipping");
    }
    ok
}

#[test]
fn unwritable_output_errors() {
    // The output name is taken by a folder, so ffmpeg can't write it.
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("taken.mp4");
    std::fs::create_dir(&out).unwrap();
    let r = render_movie(
        &short(out, 2),
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
    if !have_ffmpeg() {
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
    // The encoded file really holds 12 frames of 64×48.
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(&out)
        .output();
    if let Ok(probe) = probe {
        assert_eq!(String::from_utf8_lossy(&probe.stdout).trim(), "64,48,12");
    }
}

/// Frames ffprobe can decode from `path` (None if ffprobe isn't installed).
fn decodable_frames(path: &Path) -> Option<usize> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    Some(
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .unwrap_or(0),
    )
}

#[test]
fn unfinished_mp4_is_already_watchable() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("long.mp4");
    // Long enough for several keyframe fragments (x264 keyint is 250).
    let mut sp = short(out.clone(), 60);
    sp.seconds = 10.0;
    sp.width = 64;
    sp.height = 48;
    render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        &CancelToken::never(),
        &mut |_| {},
    )
    .unwrap();
    // A file still being written (or cut off by a crash) is a prefix of the
    // finished one. Half of it must already play.
    let bytes = std::fs::read(&out).unwrap();
    let partial = dir.path().join("partial.mp4");
    std::fs::write(&partial, &bytes[..bytes.len() / 2]).unwrap();
    let Some(n) = decodable_frames(&partial) else {
        return;
    };
    assert!(
        n >= 250,
        "only {n} frames decodable from a half-written file"
    );
}

#[test]
fn cancel_stops_early_and_removes_partial_file() {
    if !have_ffmpeg() {
        return;
    }
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("c.mp4");
    let r = render_movie(
        &short(out.clone(), 10),
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
    assert!(!out.exists(), "partial movie left behind");
}

#[test]
fn validate_rejects_bad_output_paths() {
    for out in ["/nonexistent-dir-xyz/o.mp4", "/tmp/o.txt", ""] {
        let sp = spec(PathBuf::from(out));
        assert!(
            matches!(sp.validate(), Err(MovieError::Invalid(_))),
            "{out:?}"
        );
    }
}

#[test]
fn slider_max_end_zoom_is_accepted() {
    use fractal_explorer::engine::viewport::{DEFAULT_SPAN, min_pixel_scale};
    // Same computation as the dialog's slider maximum, over many targets.
    for i in 0..6000 {
        let mut sp = s();
        sp.height = 1080;
        sp.width = 1920;
        sp.target = Complex64::new(-2.0 + i as f64 * 0.000_7, 0.3 - i as f64 * 0.000_11);
        let floor_mag = DEFAULT_SPAN / (min_pixel_scale(sp.target) * sp.height as f64);
        sp.end_span = DEFAULT_SPAN / floor_mag;
        assert!(sp.validate().is_ok(), "target {}", sp.target);
    }
}

#[cfg(unix)]
fn fake_program(dir: &Path, name: &str, mode: u32) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join(name);
    std::fs::write(&p, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
    p
}

#[cfg(unix)]
#[test]
fn find_program_prefers_path_then_fallbacks() {
    let (a, b, fb) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let in_b = fake_program(b.path(), "ffmpeg", 0o755);
    let in_fb = fake_program(fb.path(), "ffmpeg", 0o755);
    let path = std::env::join_paths([a.path(), b.path()]).unwrap();
    assert_eq!(
        find_program("ffmpeg", Some(&path), [fb.path()]),
        Some(in_b),
        "first PATH hit wins"
    );
    let only_a = std::env::join_paths([a.path()]).unwrap();
    assert_eq!(
        find_program("ffmpeg", Some(&only_a), [fb.path()]),
        Some(in_fb.clone()),
        "falls back when PATH lacks it"
    );
    assert_eq!(find_program("ffmpeg", None, [fb.path()]), Some(in_fb));
    assert_eq!(find_program("ffmpeg", Some(&only_a), [a.path()]), None);
}

#[cfg(unix)]
#[test]
fn find_program_skips_non_executables_and_folders() {
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    fake_program(a.path(), "ffmpeg", 0o644);
    std::fs::create_dir(b.path().join("ffmpeg")).unwrap();
    assert_eq!(find_program("ffmpeg", None, [a.path(), b.path()]), None);
}

#[test]
fn missing_ffmpeg_message_says_how_to_fix_it() {
    let msg = MovieError::FfmpegMissing.to_string();
    assert!(msg.contains("ffmpeg not found"), "{msg}");
    assert!(msg.contains(ffmpeg_install_hint()), "{msg}");
    assert!(ffmpeg_install_hint().contains("install ffmpeg"));
}
