//! Zoom movies: render a sequence of frames zooming toward a point and
//! stream them straight into `ffmpeg` as raw RGBA, so a movie of any
//! length needs no disk space beyond the finished file.
//!
//! The zoom is geometric: every frame is the same factor deeper than the
//! last, so the zoom feels like constant speed. The first frame is the
//! start view (center c₀, span s₀) and the last is the end view (center
//! `target` = c₁, span s₁). Every frame in between is the start view
//! scaled about one fixed point Q — the "zoom line" — chosen so that
//! scaling by r = s₁/s₀ carries c₀ onto c₁:
//!
//! Q = (c₁ − c₀·r) / (1 − r),   center(t) = Q + (c₀ − Q)·span(t)/s₀.
//!
//! Q stays on the same pixel throughout while everything else rushes
//! outward from it. With no zoom (r = 1) there is no fixed point, so the
//! center pans in a straight line instead.

use crate::engine::color::{ColorSettings, colorize};
use crate::engine::escape::FractalKind;
use crate::engine::render::{CancelToken, F64Sampler, render};
use crate::engine::viewport::{MaxIter, Viewport, min_pixel_scale};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MovieSpec {
    pub kind: FractalKind,
    pub start_center: Complex64,
    pub start_span: f64,
    pub target: Complex64,
    pub end_span: f64,
    pub fps: u32,
    pub seconds: f64,
    pub width: usize,
    pub height: usize,
    pub max_iter: MaxIter,
    pub color: ColorSettings,
    pub output: PathBuf,
}

pub enum MovieProgress {
    Frame {
        done: usize,
        total: usize,
        last_frame_ms: u64,
    },
    Encoding,
    Done(PathBuf),
}

#[derive(Debug)]
pub enum MovieError {
    Invalid(String),
    Cancelled,
    Io(std::io::Error),
    Ffmpeg {
        status: Option<i32>,
        stderr_tail: String,
    },
    FfmpegMissing,
}

impl std::fmt::Display for MovieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MovieError::Invalid(why) => write!(f, "invalid settings: {why}"),
            MovieError::Cancelled => write!(f, "cancelled"),
            MovieError::Io(e) => write!(f, "file error: {e}"),
            MovieError::Ffmpeg {
                status,
                stderr_tail,
            } => {
                write!(f, "ffmpeg failed (exit {status:?}):\n{stderr_tail}")
            }
            MovieError::FfmpegMissing => write!(f, "ffmpeg not found on PATH"),
        }
    }
}

impl From<std::io::Error> for MovieError {
    fn from(e: std::io::Error) -> Self {
        MovieError::Io(e)
    }
}

/// How much of ffmpeg's error output to keep.
const STDERR_TAIL_BYTES: usize = 2048;

impl MovieSpec {
    pub fn validate(&self) -> Result<(), MovieError> {
        let bad = |why: &str| Err(MovieError::Invalid(why.into()));
        if self.width < 16 || self.height < 16 || self.width % 2 == 1 || self.height % 2 == 1 {
            return bad("width and height must be even and at least 16");
        }
        if !(1..=120).contains(&self.fps) {
            return bad("fps must be 1–120");
        }
        if !(self.seconds.is_finite() && self.seconds > 0.0) {
            return bad("duration must be positive");
        }
        let spans_ok = [self.start_span, self.end_span]
            .iter()
            .all(|s| s.is_finite() && *s > 0.0);
        if !spans_ok {
            return bad("zoom spans must be positive");
        }
        // Small tolerance so "as deep as allowed" survives float rounding.
        if self.end_span / (self.height as f64) < min_pixel_scale(self.target) * (1.0 - 1e-9) {
            return bad("end zoom is past the f64 precision limit for this point");
        }
        self.validate_output()
    }

    /// Catch a bad output path before spending minutes rendering frames.
    fn validate_output(&self) -> Result<(), MovieError> {
        let bad = |why: String| Err(MovieError::Invalid(why));
        let ext = self
            .output
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        if !matches!(ext.as_deref(), Some("mp4" | "mkv" | "mov")) {
            return bad("output file must end in .mp4, .mkv or .mov".into());
        }
        let dir = match self.output.parent() {
            Some(d) if d.as_os_str().is_empty() => Path::new("."),
            Some(d) => d,
            None => return bad("output path has no folder".into()),
        };
        if !dir.is_dir() {
            return bad(format!("folder {} does not exist", dir.display()));
        }
        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        ((self.fps as f64 * self.seconds).round() as usize).max(2)
    }

    pub fn frame_view(&self, i: usize) -> Viewport {
        let t = i as f64 / (self.frame_count() - 1) as f64;
        let r = self.end_span / self.start_span;
        let span = self.start_span * r.powf(t);
        let (c0, c1) = (self.start_center, self.target);
        let center = if (1.0 - r).abs() < 1e-9 {
            c0 + (c1 - c0) * t
        } else {
            let fixed = (c1 - c0 * r) / (1.0 - r);
            fixed + (c0 - fixed) * (span / self.start_span)
        };
        Viewport::new(center, span, self.width, self.height)
    }
}

/// ffmpeg reads raw RGBA frames of `width`×`height` from stdin.
pub fn ffmpeg_args(fps: u32, width: usize, height: usize, output: &Path) -> Vec<OsString> {
    let size = format!("{width}x{height}");
    let fps = fps.to_string();
    let mut args: Vec<OsString> = [
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-video_size",
        &size,
        "-framerate",
        &fps,
        "-i",
        "-",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    for a in ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"] {
        args.push(a.into());
    }
    args.push(output.as_os_str().to_owned());
    args
}

/// Frames rendered ahead of the encoder. Each is width×height×4 bytes.
const FRAMES_IN_FLIGHT: usize = 2;

/// Render a movie, streaming each frame into ffmpeg as it's done.
/// Rendering the next frame overlaps encoding the last. On failure or
/// cancel, a partial output file this run created is removed.
pub fn render_movie(
    spec: &MovieSpec,
    threads: usize,
    ffmpeg: &Path,
    cancel: &CancelToken,
    progress: &mut dyn FnMut(MovieProgress),
) -> Result<PathBuf, MovieError> {
    spec.validate()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| MovieError::Invalid(e.to_string()))?;
    let existed_before = spec.output.exists();
    let mut child = Command::new(ffmpeg)
        .args(ffmpeg_args(spec.fps, spec.width, spec.height, &spec.output))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => MovieError::FfmpegMissing,
            _ => MovieError::Io(e),
        })?;
    // Drain stderr on a thread so ffmpeg can't block on a full pipe.
    let mut stderr = child.stderr.take().expect("stderr piped");
    let stderr_reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    // A writer thread feeds ffmpeg, so a stalled encoder never blocks the
    // render loop's cancel checks. Dropping `tx` closes ffmpeg's stdin.
    let mut stdin = child.stdin.take().expect("stdin piped");
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(FRAMES_IN_FLIGHT);
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        for frame in rx {
            stdin.write_all(&frame)?;
        }
        Ok(())
    });

    let rendered = render_frames(spec, &pool, cancel, &tx, progress);
    drop(tx);
    let result = match rendered {
        Ok(()) => {
            progress(MovieProgress::Encoding);
            let wrote = writer
                .join()
                .unwrap_or_else(|_| Err(std::io::Error::other("writer panicked")));
            finish(&mut child, stderr_reader, cancel, wrote)
        }
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = writer.join();
            Err(e)
        }
    };
    match result {
        Ok(()) => {
            progress(MovieProgress::Done(spec.output.clone()));
            Ok(spec.output.clone())
        }
        Err(e) => {
            if !existed_before {
                let _ = std::fs::remove_file(&spec.output);
            }
            Err(e)
        }
    }
}

/// Render every frame and hand it to the writer thread.
fn render_frames(
    spec: &MovieSpec,
    pool: &rayon::ThreadPool,
    cancel: &CancelToken,
    tx: &std::sync::mpsc::SyncSender<Vec<u8>>,
    progress: &mut dyn FnMut(MovieProgress),
) -> Result<(), MovieError> {
    let total = spec.frame_count();
    let mut pixels = vec![[0u8; 4]; spec.width * spec.height];
    for i in 0..total {
        if cancel.is_cancelled() {
            return Err(MovieError::Cancelled);
        }
        let start = Instant::now();
        let view = spec.frame_view(i);
        let sampler = F64Sampler {
            kind: spec.kind,
            max_iter: spec.max_iter.resolve(view.magnification()),
        };
        let samples = render(&sampler, &view, pool, cancel).ok_or(MovieError::Cancelled)?;
        colorize(&samples, &spec.color, &mut pixels);
        if tx.send(pixels.as_flattened().to_vec()).is_err() {
            // The writer stopped: ffmpeg exited early. `finish` reports why.
            return Ok(());
        }
        progress(MovieProgress::Frame {
            done: i + 1,
            total,
            last_frame_ms: start.elapsed().as_millis() as u64,
        });
    }
    Ok(())
}

/// Wait for ffmpeg to finish encoding (killing it if cancelled) and turn
/// its exit status into a result.
fn finish(
    child: &mut Child,
    stderr_reader: std::thread::JoinHandle<String>,
    cancel: &CancelToken,
    wrote: std::io::Result<()>,
) -> Result<(), MovieError> {
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MovieError::Cancelled);
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let err = stderr_reader.join().unwrap_or_default();
    if status.success() {
        // ffmpeg can't succeed without reading every frame, but be sure.
        return wrote.map_err(MovieError::Io);
    }
    let tail_start = err.len().saturating_sub(STDERR_TAIL_BYTES);
    let tail_start = (tail_start..err.len())
        .find(|&i| err.is_char_boundary(i))
        .unwrap_or(err.len());
    Err(MovieError::Ffmpeg {
        status: status.code(),
        stderr_tail: err[tail_start..].to_string(),
    })
}
