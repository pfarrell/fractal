//! ffmpeg helpers shared by the movie and soundtrack tests.
#![allow(dead_code)]
use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether ffmpeg and ffprobe are both on PATH; prints why a test skips.
pub fn have_tools() -> bool {
    let ok = ["ffmpeg", "ffprobe"]
        .iter()
        .all(|t| Command::new(t).arg("-version").output().is_ok());
    if !ok {
        eprintln!("ffmpeg/ffprobe not installed; skipping");
    }
    ok
}

fn run(args: &[&str]) -> String {
    let out = Command::new(args[0]).args(&args[1..]).output().unwrap();
    assert!(
        out.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// A `secs`-long 440 Hz tone at `dir/name` (format from the extension).
pub fn sine(dir: &Path, name: &str, secs: f64) -> PathBuf {
    let p = dir.join(name);
    let src = format!("sine=frequency=440:duration={secs}");
    run(&[
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        &src,
        p.to_str().unwrap(),
    ]);
    p
}

/// A `secs`-long 64×48 test-pattern video at `dir/name`.
pub fn video(dir: &Path, name: &str, secs: f64) -> PathBuf {
    let p = dir.join(name);
    let src = format!("testsrc=size=64x48:rate=12:duration={secs}");
    run(&[
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        &src,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        p.to_str().unwrap(),
    ]);
    p
}

/// MD5 of the video stream's packets, so a stream copy can be checked.
pub fn video_md5(p: &Path) -> String {
    run(&[
        "ffmpeg",
        "-v",
        "error",
        "-i",
        p.to_str().unwrap(),
        "-map",
        "0:v",
        "-c",
        "copy",
        "-f",
        "md5",
        "-",
    ])
}

/// The file's stream types in order, e.g. ["video", "audio"].
pub fn stream_types(p: &Path) -> Vec<String> {
    run(&[
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        p.to_str().unwrap(),
    ])
    .lines()
    .map(str::to_string)
    .collect()
}

/// Duration of the first stream of `kind` ("a" or "v"), in seconds.
pub fn stream_secs(p: &Path, kind: &str) -> f64 {
    let sel = format!("{kind}:0");
    run(&[
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        &sel,
        "-show_entries",
        "stream=duration",
        "-of",
        "csv=p=0",
        p.to_str().unwrap(),
    ])
    .trim()
    .parse()
    .unwrap()
}

/// File names in `dir`, sorted.
pub fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// 15 s of silence then 15 s of tone as VBR mp3 with no Xing header, so
/// its header-estimated length (~42 s) is far from the real 30 s.
pub fn vbr_no_header(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    run(&[
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo:d=15",
        "-f",
        "lavfi",
        "-i",
        "sine=f=440:d=15",
        "-filter_complex",
        "[0][1]concat=n=2:v=0:a=1",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "6",
        "-write_xing",
        "0",
        p.to_str().unwrap(),
    ]);
    p
}
