//! Zoom movies: render a sequence of frames zooming toward a point, then
//! encode them to MP4 with `ffmpeg`.
//!
//! The zoom is geometric: every frame is the same factor deeper than the
//! last, so the zoom feels like constant speed. The view zooms about the
//! target point P: center(t) = P + (c₀ − P)·span(t)/span₀, which keeps P
//! fixed on screen while everything else rushes outward.

use crate::engine::color::{ColorSettings, colorize};
use crate::engine::escape::FractalKind;
use crate::engine::render::{CancelToken, F64Sampler, render};
use crate::engine::viewport::{MaxIter, Viewport, min_pixel_scale};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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
        if self.end_span / (self.height as f64) < min_pixel_scale(self.target) {
            return bad("end zoom is past the f64 precision limit for this point");
        }
        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        ((self.fps as f64 * self.seconds).round() as usize).max(2)
    }

    pub fn frame_view(&self, i: usize) -> Viewport {
        let t = i as f64 / (self.frame_count() - 1) as f64;
        let span = self.start_span * (self.end_span / self.start_span).powf(t);
        let center = self.target + (self.start_center - self.target) * (span / self.start_span);
        Viewport::new(center, span, self.width, self.height)
    }
}

pub fn ffmpeg_args(fps: u32, frames_dir: &Path, output: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["-y", "-framerate"].iter().map(OsString::from).collect();
    args.push(fps.to_string().into());
    args.push("-i".into());
    args.push(frames_dir.join("frame_%05d.png").into_os_string());
    for a in ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"] {
        args.push(a.into());
    }
    args.push(output.as_os_str().to_owned());
    args
}

/// Render and encode a movie, keeping frames in the system temp dir.
pub fn render_movie(
    spec: &MovieSpec,
    threads: usize,
    ffmpeg: &Path,
    cancel: &CancelToken,
    progress: &mut dyn FnMut(MovieProgress),
) -> Result<PathBuf, MovieError> {
    render_movie_in(
        &std::env::temp_dir(),
        spec,
        threads,
        ffmpeg,
        cancel,
        progress,
    )
}

/// Like [`render_movie`], with the frames' temporary directory created
/// under `temp_parent`. The directory is deleted on every exit path.
pub fn render_movie_in(
    temp_parent: &Path,
    spec: &MovieSpec,
    threads: usize,
    ffmpeg: &Path,
    cancel: &CancelToken,
    progress: &mut dyn FnMut(MovieProgress),
) -> Result<PathBuf, MovieError> {
    spec.validate()?;
    let frames = tempfile::Builder::new()
        .prefix("fractal-movie-")
        .tempdir_in(temp_parent)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| MovieError::Invalid(e.to_string()))?;
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
        let samples = render(&sampler, &view, &pool, cancel).ok_or(MovieError::Cancelled)?;
        colorize(&samples, &spec.color, &mut pixels);
        let path = frames.path().join(format!("frame_{i:05}.png"));
        image::save_buffer(
            &path,
            pixels.as_flattened(),
            spec.width as u32,
            spec.height as u32,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| MovieError::Io(std::io::Error::other(e)))?;
        progress(MovieProgress::Frame {
            done: i + 1,
            total,
            last_frame_ms: start.elapsed().as_millis() as u64,
        });
    }

    progress(MovieProgress::Encoding);
    encode(ffmpeg, spec, frames.path(), cancel)?;
    progress(MovieProgress::Done(spec.output.clone()));
    Ok(spec.output.clone())
}

/// Run ffmpeg, killing it if cancelled.
fn encode(
    ffmpeg: &Path,
    spec: &MovieSpec,
    frames_dir: &Path,
    cancel: &CancelToken,
) -> Result<(), MovieError> {
    let mut child = Command::new(ffmpeg)
        .args(ffmpeg_args(spec.fps, frames_dir, &spec.output))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => MovieError::FfmpegMissing,
            _ => MovieError::Io(e),
        })?;
    // Drain stderr on a thread so ffmpeg can't block on a full pipe.
    let mut stderr = child.stderr.take().expect("stderr piped");
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
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
    let err = reader.join().unwrap_or_default();
    if status.success() {
        return Ok(());
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
