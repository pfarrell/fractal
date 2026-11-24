mod common;

use common::*;
use fractal_explorer::engine::render::CancelToken;
use fractal_explorer::engine::soundtrack::{
    add_music, filter_graph, find_ffprobe, mux_args, plan_soundtrack, probe_duration, probe_track,
};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn long_track_plays_once() {
    let p = plan_soundtrack(&[600.0], 300.0).unwrap();
    assert_eq!(p.sequence, vec![0]);
    assert_eq!(
        (p.crossfade, p.fade_in, p.fade_out, p.length),
        (4.0, 3.0, 3.0, 300.0)
    );
}

#[test]
fn short_track_loops_onto_itself() {
    // 100, then +100-4 = 196 < 250, then +100-4 = 292 >= 250.
    assert_eq!(
        plan_soundtrack(&[100.0], 250.0).unwrap().sequence,
        vec![0, 0, 0]
    );
}

#[test]
fn playlist_wraps_in_order() {
    // 60, then +50-4 = 106, then +60-4 = 162 >= 150.
    assert_eq!(
        plan_soundtrack(&[60.0, 50.0], 150.0).unwrap().sequence,
        vec![0, 1, 0]
    );
}

#[test]
fn short_tracks_shrink_the_crossfade() {
    assert_eq!(plan_soundtrack(&[60.0, 5.0], 30.0).unwrap().crossfade, 2.5);
}

#[test]
fn one_second_movie_caps_fades() {
    let p = plan_soundtrack(&[60.0], 1.0).unwrap();
    assert_eq!(p.sequence, vec![0]);
    assert!((p.fade_in - 1.0 / 3.0).abs() < 1e-12);
    assert!((p.fade_out - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn too_many_plays_is_refused() {
    let e = plan_soundtrack(&[10.0], 3600.0).unwrap_err();
    assert!(
        e.starts_with("tracks too short for a movie this long (needs "),
        "{e}"
    );
    assert!(e.ends_with(" plays, limit 500)"), "{e}");
    // 10, then +6 per play: 10 + 6*(n-1) >= 3600 → n = 600.
    assert!(e.contains("needs 600 plays"), "{e}");
}

#[test]
fn empty_or_bad_lengths_are_refused() {
    assert!(plan_soundtrack(&[], 10.0).is_err());
    assert!(plan_soundtrack(&[0.0], 10.0).is_err());
    assert!(plan_soundtrack(&[f64::NAN], 10.0).is_err());
    assert!(plan_soundtrack(&[10.0], 0.0).is_err());
}

#[test]
fn graph_for_one_input() {
    let p = plan_soundtrack(&[600.0], 300.0).unwrap();
    assert_eq!(
        filter_graph(&p),
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a1];\
         [a1]atrim=0:300.000,asetpts=N/SR/TB,afade=t=in:d=3.000,afade=t=out:st=297.000:d=3.000[aout]"
    );
}

#[test]
fn graph_chains_crossfades() {
    let p = plan_soundtrack(&[60.0, 50.0], 150.0).unwrap();
    assert_eq!(
        filter_graph(&p),
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a1];\
         [2:a]aformat=sample_rates=48000:channel_layouts=stereo[a2];\
         [3:a]aformat=sample_rates=48000:channel_layouts=stereo[a3];\
         [a1][a2]acrossfade=d=4.000:c1=tri:c2=tri[x2];\
         [x2][a3]acrossfade=d=4.000:c1=tri:c2=tri[x3];\
         [x3]atrim=0:150.000,asetpts=N/SR/TB,afade=t=in:d=3.000,afade=t=out:st=147.000:d=3.000[aout]"
    );
}

#[test]
fn mux_args_exact() {
    let p = plan_soundtrack(&[60.0, 50.0], 150.0).unwrap();
    let tracks = [PathBuf::from("/m/a.mp3"), PathBuf::from("/m/b.mp3")];
    let got = mux_args(
        Path::new("/v/in.mp4"),
        &tracks,
        &p,
        Path::new("/t/g.txt"),
        Path::new("/v/out.mp4"),
    );
    let want: Vec<OsString> = [
        "-y",
        "-v",
        "error",
        "-i",
        "/v/in.mp4",
        "-i",
        "/m/a.mp3",
        "-i",
        "/m/b.mp3",
        "-i",
        "/m/a.mp3",
        "-filter_complex_script",
        "/t/g.txt",
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "/v/out.mp4",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    assert_eq!(got, want);
}

#[test]
fn mkv_mux_has_no_faststart() {
    let p = plan_soundtrack(&[60.0], 10.0).unwrap();
    let a = [PathBuf::from("a.mp3")];
    let mkv = mux_args(
        Path::new("i.mkv"),
        &a,
        &p,
        Path::new("g"),
        Path::new("o.MKV"),
    );
    assert!(!mkv.iter().any(|x| x == "-movflags"));
    let mov = mux_args(
        Path::new("i.mov"),
        &a,
        &p,
        Path::new("g"),
        Path::new("o.MOV"),
    );
    assert!(mov.iter().any(|x| x == "+faststart"));
}

#[test]
fn probe_reads_a_tracks_length() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let t = sine(dir.path(), "t.mp3", 2.0);
    let d = probe_duration(Path::new("ffprobe"), &t).unwrap();
    assert!((d - 2.0).abs() < 0.1, "{d}");
}

#[test]
fn probe_names_a_missing_file() {
    let e = probe_duration(Path::new("ffprobe"), Path::new("/nope/gone.mp3")).unwrap_err();
    assert!(e.starts_with("gone.mp3"), "{e}");
}

#[test]
fn probe_rejects_file_without_audio() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("notes.mp3");
    std::fs::write(&p, "hello").unwrap();
    let e = probe_duration(Path::new("ffprobe"), &p).unwrap_err();
    assert!(e.starts_with("notes.mp3"), "{e}");
}

#[cfg(unix)]
#[test]
fn find_ffprobe_prefers_the_sibling() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    for name in ["ffmpeg", "ffprobe"] {
        let p = dir.path().join(name);
        std::fs::write(&p, "").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert_eq!(
        find_ffprobe(&dir.path().join("ffmpeg")),
        Some(dir.path().join("ffprobe"))
    );
}

#[test]
fn renewed_token_ignores_old_cancel() {
    let g = Arc::new(AtomicU64::new(0));
    let t = CancelToken::new(g.clone(), 0);
    g.fetch_add(1, Ordering::Relaxed);
    assert!(t.is_cancelled());
    let r = t.renewed();
    assert!(!r.is_cancelled());
    g.fetch_add(1, Ordering::Relaxed);
    assert!(r.is_cancelled());
}

#[test]
fn music_is_added_with_video_untouched() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let m = video(d, "m.mp4", 3.0);
    // Two short tracks: the playlist wraps and the crossfade shrinks.
    let tracks = vec![sine(d, "it's a: track, 1.mp3", 1.5), sine(d, "b.mp3", 1.0)];
    let secs: Vec<f64> = tracks
        .iter()
        .map(|t| probe_duration(Path::new("ffprobe"), t).unwrap())
        .collect();
    let before = video_md5(&m);
    add_music(
        Path::new("ffmpeg"),
        Path::new("ffprobe"),
        &m,
        &tracks,
        &secs,
        &CancelToken::never(),
    )
    .unwrap();
    assert_eq!(stream_types(&m), ["video", "audio"]);
    let a = stream_secs(&m, "a");
    assert!((a - 3.0).abs() < 0.1, "audio {a}");
    assert_eq!(video_md5(&m), before);
    assert_eq!(names(d), ["b.mp3", "it's a: track, 1.mp3", "m.mp4"]);
}

#[test]
fn failed_mux_keeps_silent_movie() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let m = video(d, "m.mp4", 1.0);
    let bad = d.join("notes.mp3");
    std::fs::write(&bad, "hello").unwrap();
    let bytes = std::fs::read(&m).unwrap();
    let r = add_music(
        Path::new("ffmpeg"),
        Path::new("ffprobe"),
        &m,
        &[bad],
        &[5.0],
        &CancelToken::never(),
    );
    assert!(r.is_err());
    assert_eq!(std::fs::read(&m).unwrap(), bytes);
    assert_eq!(names(d), ["m.mp4", "notes.mp3"]);
}

#[test]
fn cancelled_mux_keeps_silent_movie() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let m = video(d, "m.mp4", 1.0);
    let t = sine(d, "t.mp3", 2.0);
    let bytes = std::fs::read(&m).unwrap();
    let g = Arc::new(AtomicU64::new(1));
    let cancelled = CancelToken::new(g, 0);
    let r = add_music(
        Path::new("ffmpeg"),
        Path::new("ffprobe"),
        &m,
        &[t],
        &[2.0],
        &cancelled,
    );
    assert_eq!(r, Err("cancelled".to_string()));
    assert_eq!(std::fs::read(&m).unwrap(), bytes);
    assert_eq!(names(d), ["m.mp4", "t.mp3"]);
}

#[cfg(unix)]
#[test]
fn music_keeps_the_movies_permissions() {
    use std::os::unix::fs::PermissionsExt;
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let m = video(d, "m.mp4", 1.0);
    std::fs::set_permissions(&m, std::fs::Permissions::from_mode(0o644)).unwrap();
    let t = sine(d, "t.mp3", 2.0);
    add_music(
        Path::new("ffmpeg"),
        Path::new("ffprobe"),
        &m,
        &[t],
        &[2.0],
        &CancelToken::never(),
    )
    .unwrap();
    let mode = std::fs::metadata(&m).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o644, "{mode:o}");
}

#[test]
fn track_without_audio_is_rejected() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // A silent video has a length but nothing to play.
    let v = video(dir.path(), "silent.mp4", 2.0);
    let e = probe_track(Path::new("ffprobe"), &v).unwrap_err();
    assert!(e.starts_with("silent.mp4"), "{e}");
    assert!(e.contains("no audio"), "{e}");
}

#[test]
fn track_length_is_measured_not_estimated() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let t = vbr_no_header(dir.path(), "vbr.mp3");
    let d = probe_track(Path::new("ffprobe"), &t).unwrap();
    assert!((d - 30.0).abs() < 0.1, "{d}");
}

#[test]
fn track_probe_names_a_missing_file() {
    let e = probe_track(Path::new("ffprobe"), Path::new("/nope/gone.mp3")).unwrap_err();
    assert!(e.starts_with("gone.mp3"), "{e}");
}
