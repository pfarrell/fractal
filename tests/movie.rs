use fractal_explorer::engine::color::ColorSettings;
use fractal_explorer::engine::color::colorize;
use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::keyframe::{Block, KEYFRAME_MAX_PIXELS, Quality, plan};
use fractal_explorer::engine::movie::{
    CIRCLE_PERIOD_S, CIRCLE_RADIUS, CameraPath, Kept, MovieError, MovieProgress, MovieSpec,
    RotatedSampler, ffmpeg_args, ffmpeg_install_hint, find_program, render_exact_frame,
    render_movie, rotated_point,
};
use fractal_explorer::engine::render::{CancelToken, F64Sampler, Sampler};
use fractal_explorer::engine::viewport::{MaxIter, Viewport};
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
        path: CameraPath::Straight,
        rotation: 0.0,
        quality: Quality::Exact,
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

/// A 60 s movie with the circle path, and its straight twin.
fn circling(seconds: f64) -> (MovieSpec, MovieSpec) {
    let mut c = s();
    c.seconds = seconds;
    c.path = CameraPath::Circle;
    let mut st = s();
    st.seconds = seconds;
    (c, st)
}

/// How far frame `i` of the circle drifts from the straight zoom, in spans.
fn drift(c: &MovieSpec, st: &MovieSpec, i: usize) -> Complex64 {
    let v = c.frame_view(i);
    (v.center - st.frame_view(i).center) / v.span
}

#[test]
fn circle_keeps_first_and_last_frames() {
    for secs in [1.0, 60.0] {
        let (c, st) = circling(secs);
        let n = c.frame_count();
        assert_eq!(c.frame_view(0).center, st.frame_view(0).center, "{secs}");
        assert_eq!(
            c.frame_view(n - 1).center,
            st.frame_view(n - 1).center,
            "{secs}"
        );
    }
}

#[test]
fn circle_drift_stays_within_radius() {
    let (c, st) = circling(60.0);
    let max = (0..c.frame_count())
        .map(|i| drift(&c, &st, i).norm())
        .fold(0.0, f64::max);
    assert!(max <= CIRCLE_RADIUS * (1.0 + 1e-9), "{max}");
    assert!(max > 0.1, "barely drifts: {max}");
}

#[test]
fn circle_period_is_fifteen_seconds() {
    assert_eq!(CIRCLE_PERIOD_S, 15.0);
    let (c, st) = circling(60.0);
    let time = |i: usize| i as f64 / (c.frame_count() - 1) as f64 * c.seconds;
    // Mid-movie (fully eased in), the drift turns 2π every 15 s.
    let base = 600; // 20 s in
    for j in [700, 900, 1050, 1200] {
        let turned = (drift(&c, &st, j) / drift(&c, &st, base)).arg();
        let want = std::f64::consts::TAU * (time(j) - time(base)) / CIRCLE_PERIOD_S;
        let want =
            (want + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        assert!(
            (turned - want).abs() < 1e-6,
            "frame {j}: {turned} vs {want}"
        );
    }
}

#[test]
fn quarter_turn_moves_right_of_center_to_above() {
    let v = Viewport::new(Complex64::new(1.0, 2.0), 2.0, 100, 100);
    // Half-way to the right edge, on the middle row.
    let p = rotated_point(&v, std::f64::consts::FRAC_PI_2, 75.0, 50.0);
    assert!((p - Complex64::new(1.0, 2.5)).norm() < 1e-12, "{p}");
}

#[test]
fn zero_rotation_matches_plain_sampler() {
    let v = Viewport::new(Complex64::new(-0.743643887, 0.131825904), 3e-6, 37, 23);
    let kind = FractalKind::Mandelbrot;
    let plain = F64Sampler {
        kind,
        max_iter: 2000,
    };
    let turned = RotatedSampler {
        kind,
        max_iter: 2000,
        angle: 0.0,
    };
    for y in 0..v.height {
        for x in 0..v.width {
            assert_eq!(turned.sample(&v, x, y), plain.sample(&v, x, y), "({x},{y})");
        }
    }
}

#[test]
fn frame_angle_is_rate_times_time() {
    let mut sp = s();
    sp.rotation = 6.0;
    assert_eq!(sp.frame_angle(0), 0.0);
    let last = sp.frame_angle(sp.frame_count() - 1);
    assert!((last - 60f64.to_radians()).abs() < 1e-12, "{last}");
}

#[test]
fn validate_rejects_non_finite_rotation() {
    let dir = tempfile::tempdir().unwrap();
    for r in [f64::NAN, f64::INFINITY] {
        let mut sp = spec(dir.path().join("o.mp4"));
        sp.rotation = r;
        assert!(matches!(sp.validate(), Err(MovieError::Invalid(_))), "{r}");
    }
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

/// Render `sp`, cancelling once `after` frames are done.
fn cancel_after(sp: &MovieSpec, after: usize) -> Result<PathBuf, MovieError> {
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    if after == 0 {
        generation.fetch_add(1, Ordering::Relaxed);
    }
    render_movie(sp, 2, Path::new("ffmpeg"), &token, &mut |p| {
        if let MovieProgress::Frame { done, .. } = p
            && done == after
        {
            generation.fetch_add(1, Ordering::Relaxed);
        }
    })
}

#[test]
fn cancel_keeps_the_frames_rendered_so_far() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    for ext in ["mp4", "mkv", "mov"] {
        let out = dir.path().join(format!("c.{ext}"));
        let r = cancel_after(&short(out.clone(), 30), 5);
        let Err(MovieError::Cancelled {
            kept:
                Some(Kept {
                    path,
                    frames: n,
                    finished: true,
                }),
        }) = r
        else {
            panic!("{ext}: {r:?}");
        };
        assert_eq!(path, out);
        assert!((5..30).contains(&n), "{ext}: kept {n}");
        // ffmpeg was let finish, so every kept frame plays.
        if let Some(playable) = decodable_frames(&out) {
            assert_eq!(playable, n, "{ext}");
        }
    }
}

#[cfg(unix)]
#[test]
fn cancel_where_the_encoder_fails_to_finish_says_so() {
    // A stand-in encoder that writes its output file, reads every frame,
    // and then fails instead of closing the movie properly. A separate
    // shell writes it: a file this process held open for writing could be
    // inherited by a process another test spawns meanwhile, and running it
    // would then fail with "Text file busy".
    let dir = tempfile::tempdir().unwrap();
    let enc = dir.path().join("bad-encoder");
    let made = std::process::Command::new("sh")
        .args([
            "-c",
            r#"printf '#!/bin/sh\nfor a; do out=$a; done\ncat > "$out"\nexit 1\n' > "$1" && chmod 755 "$1""#,
            "sh",
        ])
        .arg(&enc)
        .status()
        .unwrap();
    assert!(made.success());
    let out = dir.path().join("c.mp4");
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    let r = render_movie(&short(out.clone(), 30), 2, &enc, &token, &mut |p| {
        if let MovieProgress::Frame { done: 3, .. } = p {
            generation.fetch_add(1, Ordering::Relaxed);
        }
    });
    let Err(
        e @ MovieError::Cancelled {
            kept: Some(Kept {
                finished: false, ..
            }),
        },
    ) = &r
    else {
        panic!("{r:?}");
    };
    assert!(e.to_string().contains("may hold fewer"), "{e}");
}

#[test]
fn cancel_before_any_frame_leaves_no_file() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("c.mp4");
    let r = cancel_after(&short(out.clone(), 10), 0);
    assert!(
        matches!(r, Err(MovieError::Cancelled { kept: None })),
        "{r:?}"
    );
    assert!(!out.exists(), "empty movie left behind");
}

#[test]
fn cancelled_message_says_where_the_frames_are() {
    let kept = |frames, finished| MovieError::Cancelled {
        kept: Some(Kept {
            path: PathBuf::from("/m/z.mp4"),
            frames,
            finished,
        }),
    };
    assert_eq!(
        kept(42, true).to_string(),
        "cancelled; the 42 frames rendered are in /m/z.mp4"
    );
    assert_eq!(
        kept(1, true).to_string(),
        "cancelled; the 1 frame rendered is in /m/z.mp4"
    );
    assert_eq!(
        kept(42, false).to_string(),
        "cancelled; ffmpeg didn't finish the file, so /m/z.mp4 may hold fewer \
         than the 42 frames rendered"
    );
    let e = MovieError::Cancelled { kept: None };
    assert_eq!(e.to_string(), "cancelled");
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

/// 64×48, 2 s at 30 fps, circling and turning.
fn keyframe_spec(out: PathBuf, quality: Quality) -> MovieSpec {
    let mut sp = spec(out);
    sp.seconds = 2.0;
    sp.path = CameraPath::Circle;
    sp.rotation = 10.0;
    sp.quality = quality;
    sp
}

#[test]
fn fast_and_smooth_movies_decode_every_frame() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    for q in [Quality::Fast, Quality::Smooth] {
        let out = dir.path().join(format!("{q:?}.mp4"));
        let sp = keyframe_spec(out.clone(), q);
        let r = render_movie(
            &sp,
            2,
            Path::new("ffmpeg"),
            &CancelToken::never(),
            &mut |_| {},
        );
        assert_eq!(r.unwrap(), out);
        if let Some(n) = decodable_frames(&out) {
            assert_eq!(n, 60, "{q:?}");
        }
    }
}

#[test]
fn keyframe_progress_is_reported() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let sp = keyframe_spec(dir.path().join("k.mp4"), Quality::Smooth);
    let blocks = plan(&sp, 2, KEYFRAME_MAX_PIXELS);
    let total = blocks
        .iter()
        .filter(|b| matches!(b, Block::Keyframe(_)))
        .count();
    assert!(total > 1, "want several keyframes: {total}");
    let (mut seen, mut frames) = (Vec::new(), 0);
    render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        &CancelToken::never(),
        &mut |p| match p {
            MovieProgress::Keyframe { done, total } => seen.push((done, total)),
            MovieProgress::Frame { .. } => frames += 1,
            _ => {}
        },
    )
    .unwrap();
    let want: Vec<_> = (0..total).map(|d| (d, total)).collect();
    assert_eq!(seen, want);
    assert_eq!(frames, 60);
}

#[test]
fn cancel_during_keyframe_is_prompt() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut sp = keyframe_spec(dir.path().join("p.mp4"), Quality::Smooth);
    // Big frames make the first keyframe take seconds uncancelled.
    sp.width = 1920;
    sp.height = 1080;
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    let mut cancelled_at = None;
    let r = render_movie(&sp, 2, Path::new("ffmpeg"), &token, &mut |p| {
        if let MovieProgress::Keyframe { .. } = p
            && cancelled_at.is_none()
        {
            generation.fetch_add(1, Ordering::Relaxed);
            cancelled_at = Some(std::time::Instant::now());
        }
    });
    let waited = cancelled_at.expect("a keyframe started").elapsed();
    assert!(
        matches!(r, Err(MovieError::Cancelled { kept: None })),
        "{r:?}"
    );
    assert!(waited < std::time::Duration::from_secs(1), "{waited:?}");
}

#[test]
fn exact_quality_is_unchanged() {
    let sp = keyframe_spec(PathBuf::from("/tmp/unused.mp4"), Quality::Exact);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let n = sp.width * sp.height;
    for i in (0..sp.frame_count()).step_by(7) {
        let view = sp.frame_view(i);
        let sampler = RotatedSampler {
            kind: sp.kind,
            max_iter: sp.max_iter.resolve(view.magnification()),
            angle: sp.frame_angle(i),
        };
        let samples =
            fractal_explorer::engine::render::render(&sampler, &view, &pool, &CancelToken::never())
                .unwrap();
        let mut want = vec![[0u8; 4]; n];
        colorize(&samples, &sp.color, &mut want);
        let mut got = vec![[1u8; 4]; n];
        render_exact_frame(&sp, i, &pool, &CancelToken::never(), &mut got).unwrap();
        assert!(got == want, "frame {i}");
    }
}
