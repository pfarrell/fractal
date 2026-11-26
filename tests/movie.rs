mod common;

use fractal_explorer::engine::color::ColorSettings;
use fractal_explorer::engine::color::colorize;
use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::keyframe::{Block, KEYFRAME_MAX_PIXELS, Quality, plan};
use fractal_explorer::engine::movie::{
    Kept, MovieError, MovieProgress, MovieSpec, RotatedSampler, Saved, ffmpeg_args,
    ffmpeg_install_hint, find_program, precision_floor, render_exact_frame, render_movie,
    render_range, rotated_point,
};
use fractal_explorer::engine::render::{CancelToken, F64Sampler, Sampler};
use fractal_explorer::engine::route::{RotationUnit, Stop};
use fractal_explorer::engine::viewport::{MaxIter, Viewport, min_pixel_scale};
use num_complex::Complex64;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn spec(out: PathBuf) -> MovieSpec {
    MovieSpec {
        kind: FractalKind::Mandelbrot,
        stops: vec![
            Stop {
                center: Complex64::new(-0.5, 0.0).into(),
                span: 3.0,
                rotation: 0.0,
                rotation_unit: RotationUnit::PerSecond,
            },
            Stop {
                center: Complex64::new(-0.743643887, 0.131825904).into(),
                span: 3.0e-6,
                rotation: 0.0,
                rotation_unit: RotationUnit::PerSecond,
            },
        ],
        fps: 30,
        seconds: 10.0,
        width: 64,
        height: 48,
        max_iter: MaxIter::Auto,
        color: ColorSettings::default(),
        output: out,
        quality: Quality::Exact,
        music: Vec::new(),
    }
}

fn s() -> MovieSpec {
    spec(PathBuf::from("/tmp/unused.mp4"))
}

#[test]
fn first_frame_is_start() {
    let v = s().frame_view(0);
    assert_eq!(v.center, s().stops[0].center);
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
    assert!(pixels_apart(&last, last.center.to_c64(), sp.stops[1].center.to_c64()) < 0.01);
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
fn frame_angle_ends_at_rate_times_time() {
    let mut sp = s();
    sp.stops[1].rotation = 6.0;
    assert_eq!(sp.frame_angle(0), 0.0);
    let last = sp.frame_angle(sp.frame_count() - 1);
    assert!((last - 60f64.to_radians()).abs() < 1e-12, "{last}");
}

#[test]
fn validate_rejects_non_finite_rotation() {
    let dir = tempfile::tempdir().unwrap();
    for r in [f64::NAN, f64::INFINITY] {
        let mut sp = spec(dir.path().join("o.mp4"));
        sp.stops[1].rotation = r;
        assert!(matches!(sp.validate(), Err(MovieError::Invalid(_))), "{r}");
    }
}

#[test]
fn off_center_bookmark_end_view_is_hit_exactly() {
    // End view far from the start view's center, as with a bookmark.
    let mut sp = s();
    sp.stops[0].center = Complex64::new(0.0, 0.0).into();
    sp.stops[1].center = Complex64::new(-1.25, 0.4).into();
    sp.stops[1].span = 1e-9;
    let last = sp.frame_view(sp.frame_count() - 1);
    assert!(pixels_apart(&last, last.center.to_c64(), sp.stops[1].center.to_c64()) < 0.01);
    assert!((last.span - 1e-9).abs() < 1e-21);
}

#[test]
fn zoom_follows_one_straight_zoom_line() {
    // Every frame is the start view scaled about one fixed point, so that
    // point sits on the same pixel in every frame.
    let mut sp = s();
    sp.stops[0].center = Complex64::new(0.0, 0.0).into();
    sp.stops[1].center = Complex64::new(-1.25, 0.4).into();
    sp.stops[1].span = 1e-3;
    let r = sp.stops[1].span / sp.stops[0].span;
    let fixed = (sp.stops[1].center.to_c64() - sp.stops[0].center.to_c64() * r) / (1.0 - r);
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
    sp.stops[1].span = sp.stops[0].span;
    let n = sp.frame_count();
    let mid = sp.frame_view((n - 1) / 2);
    let last = sp.frame_view(n - 1);
    assert!(mid.center.to_c64().re.is_finite() && mid.center.to_c64().im.is_finite());
    assert!(last.center.diff_c64(&sp.stops[1].center).norm() < 1e-12);
    assert_eq!(last.span, sp.stops[0].span);
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
    deep.stops[1].span = 1e-301;
    assert!(matches!(deep.validate(), Err(MovieError::Invalid(_))));
    deep.stops[1].span = 1e-20;
    assert!(deep.validate().is_ok(), "{:?}", deep.validate());
    deep.kind = FractalKind::BurningShip;
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
        None,
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
        None,
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
        None,
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
        None,
        &CancelToken::never(),
        &mut |p| {
            if let MovieProgress::Frame { .. } = p {
                frames += 1
            }
        },
    );
    assert_eq!(r.unwrap().path, out);
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
        None,
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
fn cancel_after(sp: &MovieSpec, after: usize) -> Result<Saved, MovieError> {
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    if after == 0 {
        generation.fetch_add(1, Ordering::Relaxed);
    }
    render_movie(sp, 2, Path::new("ffmpeg"), None, &token, &mut |p| {
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
                    ..
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
    let r = render_movie(&short(out.clone(), 30), 2, &enc, None, &token, &mut |p| {
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
            music_error: None,
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
    use fractal_explorer::engine::viewport::DEFAULT_SPAN;
    // Same computation as the dialog's slider maximum, over many targets.
    for i in 0..6000 {
        let mut sp = s();
        sp.height = 1080;
        sp.width = 1920;
        sp.stops[1].center =
            Complex64::new(-2.0 + i as f64 * 0.000_7, 0.3 - i as f64 * 0.000_11).into();
        let floor_mag =
            DEFAULT_SPAN / (min_pixel_scale(sp.stops[1].center.to_c64()) * sp.height as f64);
        sp.stops[1].span = DEFAULT_SPAN / floor_mag;
        assert!(sp.validate().is_ok(), "target {:?}", sp.stops[1].center);
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

/// 64×48, 2 s at 30 fps, turning.
fn keyframe_spec(out: PathBuf, quality: Quality) -> MovieSpec {
    let mut sp = spec(out);
    sp.seconds = 2.0;
    sp.stops[1].rotation = 10.0;
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
            None,
            &CancelToken::never(),
            &mut |_| {},
        );
        assert_eq!(r.unwrap().path, out);
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
        None,
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
    let r = render_movie(&sp, 2, Path::new("ffmpeg"), None, &token, &mut |p| {
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

#[test]
fn cancel_during_a_later_keyframe_keeps_the_earlier_stretch() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("k.mp4");
    let sp = keyframe_spec(out.clone(), Quality::Smooth);
    let blocks = plan(&sp, 2, KEYFRAME_MAX_PIXELS);
    let Some(Block::Keyframe(first)) = blocks.first() else {
        panic!("want a keyframe first: {blocks:?}");
    };
    assert!(matches!(blocks.get(1), Some(Block::Keyframe(_))));
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    // Cancel as the second keyframe starts: the first stretch is all sent.
    let r = render_movie(&sp, 2, Path::new("ffmpeg"), None, &token, &mut |p| {
        if let MovieProgress::Keyframe { done: 1, .. } = p {
            generation.fetch_add(1, Ordering::Relaxed);
        }
    });
    let Err(MovieError::Cancelled {
        kept:
            Some(Kept {
                path,
                frames,
                finished: true,
                ..
            }),
    }) = r
    else {
        panic!("{r:?}");
    };
    assert_eq!(path, out);
    assert_eq!(frames, first.frames.len());
    if let Some(playable) = decodable_frames(&out) {
        assert_eq!(playable, frames);
    }
}

fn stop(re: f64, im: f64, span: f64, rotation: f64) -> Stop {
    Stop {
        center: Complex64::new(re, im).into(),
        span,
        rotation,
        rotation_unit: RotationUnit::PerSecond,
    }
}

/// Three stops along one dive, with a turn on each step.
fn three_stops(out: PathBuf) -> MovieSpec {
    let mut sp = spec(out);
    sp.stops = vec![
        stop(-0.5, 0.0, 3.0, 0.0),
        stop(-0.75, 0.1, 0.03, 10.0),
        stop(-0.745, 0.105, 3e-5, -20.0),
    ];
    sp
}

#[test]
fn three_stop_spec_hits_first_and_last() {
    let sp = three_stops(PathBuf::from("/tmp/unused.mp4"));
    let first = sp.frame_view(0);
    let last = sp.frame_view(sp.frame_count() - 1);
    assert_eq!(
        (first.center.to_c64(), first.span),
        (sp.stops[0].center.to_c64(), 3.0)
    );
    assert_eq!(
        (last.center.to_c64(), last.span),
        (sp.stops[2].center.to_c64(), 3e-5)
    );
}

#[test]
fn validate_rejects_single_stop() {
    let dir = tempfile::tempdir().unwrap();
    let mut sp = spec(dir.path().join("o.mp4"));
    sp.stops.truncate(1);
    match sp.validate() {
        Err(MovieError::Invalid(m)) => assert!(m.contains("at least two stops"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn validate_rejects_zero_length_step() {
    let dir = tempfile::tempdir().unwrap();
    let mut sp = spec(dir.path().join("o.mp4"));
    let again = sp.stops[1].clone();
    sp.stops.push(again);
    match sp.validate() {
        Err(MovieError::Invalid(m)) => assert!(m.contains("same view"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn validate_checks_every_stop_against_precision() {
    let dir = tempfile::tempdir().unwrap();
    let mut sp = three_stops(dir.path().join("o.mp4"));
    // Mandelbrot goes far deeper (perturbation); test an f64-only kind.
    sp.kind = FractalKind::BurningShip;
    let c1 = sp.stops[1].center.clone();
    sp.stops[1].span = min_pixel_scale(c1.to_c64()) * sp.height as f64 * 0.5;
    sp.stops[2] = stop(-0.75, 0.1001, 1e-3, 0.0);
    match sp.validate() {
        Err(MovieError::Invalid(m)) => assert!(m.contains("stop 2"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn fast_and_smooth_three_stop_movie_decodes() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    for q in [Quality::Fast, Quality::Smooth] {
        let out = dir.path().join(format!("{q:?}.mp4"));
        let mut sp = three_stops(out.clone());
        sp.seconds = 2.0;
        sp.quality = q;
        let r = render_movie(
            &sp,
            2,
            Path::new("ffmpeg"),
            None,
            &CancelToken::never(),
            &mut |_| {},
        );
        assert_eq!(r.unwrap().path, out);
        if let Some(n) = decodable_frames(&out) {
            assert_eq!(n, sp.frame_count(), "{q:?}");
        }
    }
}

#[test]
fn spin_in_place_is_a_valid_movie() {
    let dir = tempfile::tempdir().unwrap();
    let mut sp = spec(dir.path().join("o.mp4"));
    sp.stops[1] = Stop {
        rotation: 10.0,
        ..sp.stops[0].clone()
    };
    assert!(sp.validate().is_ok(), "{:?}", sp.validate());
    for i in [0, sp.frame_count() / 2, sp.frame_count() - 1] {
        let v = sp.frame_view(i);
        assert_eq!(
            (v.center.to_c64(), v.span),
            (sp.stops[0].center.to_c64(), sp.stops[0].span)
        );
    }
    let last = sp.frame_angle(sp.frame_count() - 1);
    assert!((last - 100f64.to_radians()).abs() < 1e-9, "{last}");

    sp.stops[1].rotation = 360.0;
    sp.stops[1].rotation_unit = RotationUnit::Total;
    assert!(sp.validate().is_ok(), "{:?}", sp.validate());
    let last = sp.frame_angle(sp.frame_count() - 1);
    assert!((last - std::f64::consts::TAU).abs() < 1e-9, "{last}");
}

fn range_pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
}

fn small(quality: Quality, frames: u32) -> MovieSpec {
    let mut sp = spec(PathBuf::from("/tmp/unused.mp4"));
    (sp.fps, sp.seconds, sp.width, sp.height, sp.quality) = (frames, 1.0, 64, 48, quality);
    sp
}

#[test]
fn render_range_exact_matches_single_frames() {
    let sp = small(Quality::Exact, 12);
    let pool = range_pool();
    let mut got = Vec::new();
    render_range(
        &sp,
        5..9,
        &pool,
        &CancelToken::never(),
        &mut |i, px| {
            got.push((i, px.to_vec()));
            true
        },
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        got.iter().map(|g| g.0).collect::<Vec<_>>(),
        vec![5, 6, 7, 8]
    );
    for (i, px) in got {
        let mut want = vec![[0u8; 4]; 64 * 48];
        render_exact_frame(&sp, i, &pool, &CancelToken::never(), &mut want).unwrap();
        assert_eq!(px, want, "frame {i}");
    }
}

#[test]
fn render_range_fast_visits_each_frame_once() {
    let sp = small(Quality::Fast, 60);
    let mut seen = Vec::new();
    let mut last = None;
    render_range(
        &sp,
        10..50,
        &range_pool(),
        &CancelToken::never(),
        &mut |i, _| {
            seen.push(i);
            true
        },
        &mut |p| {
            if let MovieProgress::Frame { done, total, .. } = p {
                last = Some((done, total));
            }
        },
    )
    .unwrap();
    assert_eq!(seen, (10..50).collect::<Vec<_>>());
    assert_eq!(last, Some((40, 40)));
}

#[test]
fn render_range_stops_when_sink_says_so() {
    let sp = small(Quality::Fast, 30);
    let mut calls = 0;
    let r = render_range(
        &sp,
        0..30,
        &range_pool(),
        &CancelToken::never(),
        &mut |_, _| {
            calls += 1;
            calls < 3
        },
        &mut |_| {},
    );
    assert!(r.is_ok());
    assert_eq!(calls, 3);
}

#[test]
fn validate_frames_ignores_output() {
    let mut sp = small(Quality::Fast, 12);
    sp.output = PathBuf::from("/no/such/dir/x.mp4");
    assert!(sp.validate_frames().is_ok());
    assert!(sp.validate().is_err());
}

/// `short(out, frames)` with one `secs`-long tone as its music.
fn with_music(dir: &Path, out: PathBuf, frames: u32, secs: f64) -> MovieSpec {
    let mut sp = short(out, frames);
    sp.music = vec![common::sine(dir, "t.mp3", secs)];
    sp
}

#[test]
fn music_without_ffprobe_fails_before_rendering() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("o.mp4");
    let mut sp = short(out.clone(), 2);
    sp.music = vec![dir.path().join("t.mp3")];
    let mut frames = 0;
    let r = render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        None,
        &CancelToken::never(),
        &mut |p| {
            if let MovieProgress::Frame { .. } = p {
                frames += 1
            }
        },
    );
    assert!(matches!(r, Err(MovieError::FfprobeMissing)), "{r:?}");
    assert_eq!(frames, 0);
    assert!(!out.exists());
}

#[test]
fn unreadable_track_fails_before_rendering() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("o.mp4");
    let mut sp = short(out.clone(), 2);
    sp.music = vec![dir.path().join("gone.mp3")];
    let mut frames = 0;
    let r = render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        Some(Path::new("ffprobe")),
        &CancelToken::never(),
        &mut |p| {
            if let MovieProgress::Frame { .. } = p {
                frames += 1
            }
        },
    );
    let Err(MovieError::Invalid(m)) = &r else {
        panic!("{r:?}");
    };
    assert!(m.contains("gone.mp3"), "{m}");
    assert_eq!(frames, 0);
    assert!(!out.exists());
}

#[test]
fn movie_with_music_end_to_end() {
    if !common::have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("m.mp4");
    let sp = with_music(dir.path(), out.clone(), 12, 0.5);
    let mut seen = Vec::new();
    let r = render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        Some(Path::new("ffprobe")),
        &CancelToken::never(),
        &mut |p| match p {
            MovieProgress::Encoding => seen.push("encoding"),
            MovieProgress::AddingMusic => seen.push("music"),
            MovieProgress::Done(_) => seen.push("done"),
            _ => {}
        },
    )
    .unwrap();
    assert_eq!(r.path, out);
    assert_eq!(r.music_error, None);
    assert_eq!(seen, ["encoding", "music", "done"]);
    assert_eq!(common::stream_types(&out), ["video", "audio"]);
}

#[test]
fn movie_without_music_skips_the_music_pass() {
    if !common::have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("m.mp4");
    let mut music = false;
    let r = render_movie(
        &short(out.clone(), 4),
        2,
        Path::new("ffmpeg"),
        None,
        &CancelToken::never(),
        &mut |p| music |= matches!(p, MovieProgress::AddingMusic),
    )
    .unwrap();
    assert_eq!(r.music_error, None);
    assert!(!music);
    assert_eq!(common::stream_types(&out), ["video"]);
}

#[test]
fn music_failure_still_saves_the_movie() {
    if !common::have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("m.mp4");
    let sp = with_music(dir.path(), out.clone(), 12, 0.5);
    let track = sp.music[0].clone();
    let r = render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        Some(Path::new("ffprobe")),
        &CancelToken::never(),
        &mut |p| {
            if let MovieProgress::AddingMusic = p {
                std::fs::remove_file(&track).unwrap();
            }
        },
    )
    .unwrap();
    assert_eq!(r.path, out);
    assert!(r.music_error.is_some());
    assert_eq!(common::stream_types(&out), ["video"]);
}

#[test]
fn cancel_during_music_keeps_silent_movie() {
    if !common::have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("m.mp4");
    let sp = with_music(dir.path(), out.clone(), 12, 0.5);
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    let r = render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        Some(Path::new("ffprobe")),
        &token,
        &mut |p| {
            if let MovieProgress::AddingMusic = p {
                generation.fetch_add(1, Ordering::Relaxed);
            }
        },
    );
    let Err(MovieError::Cancelled {
        kept:
            Some(Kept {
                path,
                finished: true,
                music_error: Some(m),
                ..
            }),
    }) = &r
    else {
        panic!("{r:?}");
    };
    assert_eq!(path, &out);
    assert_eq!(m, "cancelled");
    assert_eq!(common::stream_types(&out), ["video"]);
}

#[test]
fn cancelled_render_still_gets_music() {
    if !common::have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("m.mp4");
    let sp = with_music(dir.path(), out.clone(), 30, 0.5);
    let generation = Arc::new(AtomicU64::new(0));
    let token = CancelToken::new(generation.clone(), 0);
    let r = render_movie(
        &sp,
        2,
        Path::new("ffmpeg"),
        Some(Path::new("ffprobe")),
        &token,
        &mut |p| {
            if let MovieProgress::Frame { done: 5, .. } = p {
                generation.fetch_add(1, Ordering::Relaxed);
            }
        },
    );
    let Err(MovieError::Cancelled {
        kept: Some(Kept {
            music_error: None, ..
        }),
    }) = &r
    else {
        panic!("{r:?}");
    };
    assert_eq!(common::stream_types(&out), ["video", "audio"]);
    let (a, v) = (
        common::stream_secs(&out, "a"),
        common::stream_secs(&out, "v"),
    );
    assert!((a - v).abs() < 0.1, "audio {a} video {v}");
}

#[test]
fn kept_message_mentions_missing_music() {
    let e = MovieError::Cancelled {
        kept: Some(Kept {
            path: PathBuf::from("/m/z.mp4"),
            frames: 5,
            finished: true,
            music_error: Some("x".into()),
        }),
    };
    assert_eq!(
        e.to_string(),
        "cancelled; the 5 frames rendered are in /m/z.mp4; music not added: x"
    );
}

#[test]
fn ffprobe_missing_message_has_the_install_hint() {
    let m = MovieError::FfprobeMissing.to_string();
    assert_eq!(
        m,
        format!(
            "ffprobe not found — adding music needs it. {}, then try again.",
            ffmpeg_install_hint()
        )
    );
}

#[test]
fn clamp_depth_raises_too_deep_stops_to_the_floor() {
    let mut sp = s();
    sp.width = 1920;
    sp.height = 1080;
    let floor = precision_floor(&sp.kind, &sp.stops[1].center, sp.height);
    sp.stops[1].span = floor / 2.0;
    assert!(sp.validate_frames().is_err());
    assert_eq!(sp.clamp_depth(), vec![1]);
    assert_eq!(sp.stops[1].span, floor);
    assert!(sp.validate_frames().is_ok(), "{:?}", sp.validate_frames());
}

#[test]
fn clamp_depth_leaves_resolvable_stops() {
    let mut sp = s();
    let before = sp.stops.clone();
    assert!(sp.clamp_depth().is_empty());
    assert_eq!(sp.stops, before);
}

fn deep_end() -> fractal_explorer::engine::deep::HiComplex {
    use fractal_explorer::engine::deep::{HiComplex, bits_for};
    HiComplex::from_c64(Complex64::new(-0.743643887, 0.131825904))
        .add_c64(Complex64::new(3e-45, -1e-45), bits_for(1e-60))
}

#[test]
fn zoom_line_holds_at_depth() {
    use fractal_explorer::engine::deep::{HiComplex, bits_for};
    let mut sp = s();
    sp.stops[1].center = deep_end();
    sp.stops[1].span = 3e-50;
    sp.seconds = 2.0;
    let (c0, c1) = (sp.stops[0].center.clone(), sp.stops[1].center.clone());
    let r = 1e-50;
    // Q = (c₁ − c₀·r)/(1 − r) = c₁ + (c₁ − c₀)·r/(1 − r).
    let q = c1
        .with_bits(bits_for(1e-60))
        .add(&c1.sub(&c0).mul_f64(r / (1.0 - r)));
    let dir0 = c0.diff_c64(&q);
    let n = sp.frame_count();
    for i in 0..n {
        let v = sp.frame_view(i);
        let d = v.center.diff_c64(&q);
        let cross = d.re * dir0.im - d.im * dir0.re;
        assert!(
            cross.abs() <= 1e-6 * d.norm() * dir0.norm() + 1e-300,
            "frame {i} off the zoom line"
        );
    }
    let last = sp.frame_view(n - 1);
    assert!(last.center.diff_c64(&c1).norm() < 1e-3 * 3e-50);
    let _ = HiComplex::from_c64(Complex64::new(0.0, 0.0));
}

#[test]
fn clamp_depth_per_kind() {
    let mut sp = s();
    sp.height = 1080;
    sp.width = 1920;
    sp.stops[1].span = 1e-50;
    assert!(sp.clamp_depth().is_empty());
    assert_eq!(sp.stops[1].span, 1e-50);
    sp.kind = FractalKind::Julia {
        c: Complex64::new(-0.8, 0.156),
    };
    sp.stops[1].span = 1e-14;
    assert_eq!(sp.clamp_depth(), vec![1]);
    assert_eq!(
        sp.stops[1].span,
        precision_floor(&sp.kind, &sp.stops[1].center, 1080)
    );
}

#[test]
fn old_spec_json_loads() {
    let mut j = serde_json::to_value(s()).unwrap();
    j["stops"][1]["center"] = serde_json::json!([-0.75, 0.1]);
    let back: MovieSpec = serde_json::from_value(j).unwrap();
    assert_eq!(
        back.stops[1].center,
        fractal_explorer::engine::deep::HiComplex::from_c64(Complex64::new(-0.75, 0.1))
    );
}

fn movie_pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().build().unwrap()
}

/// Fraction of escape samples that agree within ±1 iteration.
fn sample_agreement(
    a: &[fractal_explorer::engine::escape::Sample],
    b: &[fractal_explorer::engine::escape::Sample],
) -> f64 {
    use fractal_explorer::engine::escape::Sample::Escaped;
    let ok = a
        .iter()
        .zip(b)
        .filter(|(x, y)| match (x, y) {
            (Escaped { iter: p, .. }, Escaped { iter: q, .. }) => p.abs_diff(*q) <= 1,
            (Escaped { .. }, _) | (_, Escaped { .. }) => false,
            _ => true,
        })
        .count();
    ok as f64 / a.len() as f64
}

#[test]
fn switch_over_has_no_jump() {
    use fractal_explorer::engine::perturb::{PerturbSampler, uses_perturbation};
    use fractal_explorer::engine::render::render_serial;
    let mut sp = s();
    sp.stops[0].center = Complex64::new(-0.743643887, 0.131825904).into();
    sp.stops[0].span = 3e-6;
    sp.stops[1].span = 3e-12;
    sp.seconds = 40.0 / 30.0;
    sp.fps = 30;
    let flip = (0..sp.frame_count())
        .find(|&i| uses_perturbation(&sp.kind, &sp.frame_view(i)))
        .expect("route crosses the switch-over");
    assert!(flip > 0);
    for i in [flip - 1, flip] {
        let v = sp.frame_view(i);
        let max_iter = sp.max_iter.resolve(v.magnification());
        let f = render_serial(
            &F64Sampler {
                kind: sp.kind,
                max_iter,
            },
            &v,
        );
        let p = render_serial(&PerturbSampler::new(&v, max_iter, 0.0), &v);
        let a = sample_agreement(&f, &p);
        assert!(a >= 0.99, "frame {i}: {a}");
    }
}

#[test]
fn deep_exact_frame_matches_keyframe_frame() {
    let mut sp = s();
    sp.stops[1].center = Complex64::new(0.0, 1.0).into();
    sp.stops[1].span = 3e-40;
    sp.width = 48;
    sp.height = 32;
    sp.seconds = 1.0;
    let last = sp.frame_count() - 1;
    let pool = movie_pool();
    let mut exact = vec![[0u8; 4]; 48 * 32];
    render_exact_frame(&sp, last, &pool, &CancelToken::never(), &mut exact).unwrap();
    sp.quality = Quality::Fast;
    let mut fast = Vec::new();
    render_range(
        &sp,
        last..last + 1,
        &pool,
        &CancelToken::never(),
        &mut |_, px| {
            fast = px.to_vec();
            true
        },
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(fast.len(), exact.len());
    let distinct: std::collections::HashSet<_> = exact.iter().collect();
    assert!(distinct.len() > 5, "plain frame");
    let off = exact
        .iter()
        .zip(&fast)
        .filter(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 8))
        .count();
    assert!(
        off * 20 < exact.len(),
        "{off} of {} pixels differ",
        exact.len()
    );
}
