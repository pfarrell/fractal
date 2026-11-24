//! Music for a finished movie: the chosen tracks play in order, crossfade
//! at every join (including where the playlist wraps back to the start),
//! fade in and out, and are trimmed to the movie's length. One ffmpeg pass
//! adds them with the video stream copied, never re-encoded.

use crate::engine::movie::{FFMPEG_FALLBACK_DIRS, find_program, is_executable, stderr_tail};
use crate::engine::render::CancelToken;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

/// Overlap at every join between tracks, in seconds.
pub const CROSSFADE_S: f64 = 4.0;
/// Fade-in at the start and fade-out at the end, in seconds.
pub const FADE_S: f64 = 3.0;
/// Most track plays one soundtrack may list as ffmpeg inputs.
pub const MAX_INPUTS: usize = 500;

/// How the playlist fills one movie.
#[derive(Clone, Debug, PartialEq)]
pub struct SoundtrackPlan {
    /// Indices into the track list, in play order.
    pub sequence: Vec<usize>,
    pub crossfade: f64,
    pub fade_in: f64,
    pub fade_out: f64,
    /// The movie's length: the soundtrack is trimmed to it exactly.
    pub length: f64,
}

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Repeat the tracks in order until, overlapped by the crossfade, they
/// cover `movie_secs`.
pub fn plan_soundtrack(track_secs: &[f64], movie_secs: f64) -> Result<SoundtrackPlan, String> {
    if track_secs.is_empty() {
        return Err("no music tracks".into());
    }
    if !positive(movie_secs) {
        return Err("movie length must be positive".into());
    }
    if let Some(i) = track_secs.iter().position(|&t| !positive(t)) {
        return Err(format!("track {} has no usable length", i + 1));
    }
    let shortest = track_secs.iter().copied().fold(f64::INFINITY, f64::min);
    let crossfade = CROSSFADE_S.min(shortest / 2.0);
    let n = track_secs.len();
    let mut sequence = Vec::new();
    let mut covered = 0.0;
    let mut plays = 0usize;
    while covered < movie_secs {
        let i = plays % n;
        covered += track_secs[i] - if plays == 0 { 0.0 } else { crossfade };
        plays += 1;
        if plays <= MAX_INPUTS {
            sequence.push(i);
        }
    }
    if plays > MAX_INPUTS {
        return Err(format!(
            "tracks too short for a movie this long (needs {plays} plays, limit {MAX_INPUTS})"
        ));
    }
    let fade = FADE_S.min(movie_secs / 3.0);
    Ok(SoundtrackPlan {
        sequence,
        crossfade,
        fade_in: fade,
        fade_out: fade,
        length: movie_secs,
    })
}

fn secs(x: f64) -> String {
    format!("{x:.3}")
}

/// The `-filter_complex` graph for `plan`. Input 0 is the video; track
/// play `k` (from 1) is input `k`. The result is labelled `[aout]`.
pub fn filter_graph(plan: &SoundtrackPlan) -> String {
    let m = plan.sequence.len();
    let mut parts: Vec<String> = (1..=m)
        .map(|k| format!("[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{k}]"))
        .collect();
    let mut last = "a1".to_string();
    for k in 2..=m {
        let next = format!("x{k}");
        parts.push(format!(
            "[{last}][a{k}]acrossfade=d={}:c1=tri:c2=tri[{next}]",
            secs(plan.crossfade)
        ));
        last = next;
    }
    parts.push(format!(
        "[{last}]atrim=0:{},asetpts=N/SR/TB,afade=t=in:d={},afade=t=out:st={}:d={}[aout]",
        secs(plan.length),
        secs(plan.fade_in),
        secs(plan.length - plan.fade_out),
        secs(plan.fade_out)
    ));
    parts.join(";")
}

/// ffmpeg arguments that copy `video`'s picture and add the soundtrack
/// whose graph is in the file `script`, writing `out`.
pub fn mux_args(
    video: &Path,
    tracks: &[PathBuf],
    plan: &SoundtrackPlan,
    script: &Path,
    out: &Path,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["-y", "-v", "error", "-i"]
        .iter()
        .map(OsString::from)
        .collect();
    args.push(video.as_os_str().to_owned());
    for &i in &plan.sequence {
        args.push("-i".into());
        args.push(tracks[i].as_os_str().to_owned());
    }
    args.push("-filter_complex_script".into());
    args.push(script.as_os_str().to_owned());
    for a in [
        "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar",
        "48000",
    ] {
        args.push(a.into());
    }
    // The file is complete by now: a plain MP4 with its index up front
    // plays straight away on the web.
    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    if matches!(ext.as_deref(), Some("mp4" | "mov")) {
        args.push("-movflags".into());
        args.push("+faststart".into());
    }
    args.push(out.as_os_str().to_owned());
    args
}

/// ffprobe: next to `ffmpeg` if it's there, else on `PATH` or in a usual
/// install folder.
pub fn find_ffprobe(ffmpeg: &Path) -> Option<PathBuf> {
    let file = format!("ffprobe{}", std::env::consts::EXE_SUFFIX);
    ffmpeg
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.join(&file))
        .filter(|p| is_executable(p))
        .or_else(|| {
            find_program(
                "ffprobe",
                std::env::var_os("PATH").as_deref(),
                FFMPEG_FALLBACK_DIRS.iter().map(Path::new),
            )
        })
}

fn file_label(file: &Path) -> String {
    file.file_name()
        .unwrap_or(file.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// `file`'s length in seconds. Errors start with the file's name.
pub fn probe_duration(ffprobe: &Path, file: &Path) -> Result<f64, String> {
    let name = file_label(file);
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-show_entries", "format=duration"])
        .args(["-of", "default=nw=1:nk=1"])
        .arg(file)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("{name}: can't run ffprobe: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "{name}: can't be read: {}",
            stderr_tail(err.trim())
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    match text.trim().parse::<f64>() {
        Ok(d) if positive(d) => Ok(d),
        _ => Err(format!("{name}: has no length ffprobe can read")),
    }
}

/// Add the tracks (lengths `track_secs`) to the finished movie `video`,
/// replacing it. On any error, including `Err("cancelled")`, `video` is
/// left as it was and nothing else is left behind.
pub fn add_music(
    ffmpeg: &Path,
    ffprobe: &Path,
    video: &Path,
    tracks: &[PathBuf],
    track_secs: &[f64],
    cancel: &CancelToken,
) -> Result<(), String> {
    let length = probe_duration(ffprobe, video)?;
    let plan = plan_soundtrack(track_secs, length)?;
    let io = |e: std::io::Error| format!("file error: {e}");
    let mut script = tempfile::NamedTempFile::new().map_err(io)?;
    script
        .write_all(filter_graph(&plan).as_bytes())
        .and_then(|()| script.flush())
        .map_err(io)?;
    let dir = match video.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let ext = video
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    let out = tempfile::Builder::new()
        .prefix(".fractal-music-")
        .suffix(&ext)
        .tempfile_in(dir)
        .map_err(io)?;
    let mut child = Command::new(ffmpeg)
        .args(mux_args(video, tracks, &plan, script.path(), out.path()))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("can't run ffmpeg: {e}"))?;
    let mut stderr = child.stderr.take().expect("stderr piped");
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let status = loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".into());
        }
        if let Some(status) = child.try_wait().map_err(io)? {
            break status;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let err = reader.join().unwrap_or_default();
    if !status.success() {
        return Err(format!(
            "ffmpeg failed (exit {:?}):\n{}",
            status.code(),
            stderr_tail(&err)
        ));
    }
    // The temp file is private (0600); the movie keeps its own permissions.
    let perms = std::fs::metadata(video).map_err(io)?.permissions();
    std::fs::set_permissions(out.path(), perms).map_err(io)?;
    out.persist(video).map_err(|e| io(e.error))?;
    Ok(())
}
