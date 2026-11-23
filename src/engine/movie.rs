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
//!
//! That is the `Straight` camera path. `Circle` adds a drift of up to
//! [`CIRCLE_RADIUS`] spans around that center, so Q circles the screen
//! instead of staying put, and `rotation` turns each frame about the
//! screen center. The drift is zero on the first and last frames.

use crate::engine::color::{ColorSettings, colorize};
use crate::engine::escape::{FractalKind, Sample, iterate};
use crate::engine::keyframe::Quality;
use crate::engine::render::{CancelToken, Sampler, render};
use crate::engine::viewport::{MaxIter, Viewport, min_pixel_scale};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// How the camera moves on its way to the target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraPath {
    /// Straight in, the target fixed at the screen center.
    #[default]
    Straight,
    /// The target circles the screen center, eased out by the end.
    Circle,
}

impl CameraPath {
    pub const ALL: [CameraPath; 2] = [CameraPath::Straight, CameraPath::Circle];

    pub fn label(&self) -> &'static str {
        match self {
            CameraPath::Straight => "Straight",
            CameraPath::Circle => "Circle",
        }
    }
}

/// Seconds of movie per loop of the circle path.
pub const CIRCLE_PERIOD_S: f64 = 15.0;
/// The circle's radius, as a fraction of the current view's span.
pub const CIRCLE_RADIUS: f64 = 0.15;

/// The point at pixel position (`x`, `y`) of `view` turned by `angle`
/// radians (counterclockwise) about the view's center.
pub fn rotated_point(view: &Viewport, angle: f64, x: f64, y: f64) -> Complex64 {
    if angle == 0.0 {
        return view.pixel_to_complex(x, y);
    }
    let s = view.pixel_scale();
    let dx = (x - view.width as f64 / 2.0) * s;
    let dy = -(y - view.height as f64 / 2.0) * s;
    let (sin, cos) = angle.sin_cos();
    view.center + Complex64::new(dx * cos - dy * sin, dx * sin + dy * cos)
}

/// One `iterate` per pixel, with the picture turned by `angle` radians.
#[derive(Clone, Copy, Debug)]
pub struct RotatedSampler {
    pub kind: FractalKind,
    pub max_iter: u32,
    pub angle: f64,
}

impl Sampler for RotatedSampler {
    fn sample(&self, view: &Viewport, x: usize, y: usize) -> Sample {
        let point = rotated_point(view, self.angle, x as f64 + 0.5, y as f64 + 0.5);
        iterate(&self.kind, point, self.max_iter)
    }
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

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
    #[serde(default)]
    pub path: CameraPath,
    /// Degrees per second the picture turns (negative: the other way).
    #[serde(default)]
    pub rotation: f64,
    /// Exact renders every frame; Fast and Smooth resample keyframes.
    #[serde(default)]
    pub quality: Quality,
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

/// What a cancelled movie left behind.
#[derive(Debug)]
pub struct Kept {
    pub path: PathBuf,
    /// Frames rendered and handed to ffmpeg.
    pub frames: usize,
    /// ffmpeg closed the file cleanly, so it holds all of `frames`.
    /// Otherwise it was stopped or failed and may hold fewer.
    pub finished: bool,
}

#[derive(Debug)]
pub enum MovieError {
    Invalid(String),
    /// Stopped by the user. `kept` is set when any frames were rendered.
    Cancelled {
        kept: Option<Kept>,
    },
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
            MovieError::Cancelled { kept: None } => write!(f, "cancelled"),
            MovieError::Cancelled { kept: Some(k) } => {
                let path = k.path.display();
                let n = k.frames;
                match (k.finished, n) {
                    (true, 1) => write!(f, "cancelled; the 1 frame rendered is in {path}"),
                    (true, _) => write!(f, "cancelled; the {n} frames rendered are in {path}"),
                    (false, _) => write!(
                        f,
                        "cancelled; ffmpeg didn't finish the file, so {path} may hold \
                         fewer than the {n} frames rendered"
                    ),
                }
            }
            MovieError::Io(e) => write!(f, "file error: {e}"),
            MovieError::Ffmpeg {
                status,
                stderr_tail,
            } => {
                write!(f, "ffmpeg failed (exit {status:?}):\n{stderr_tail}")
            }
            MovieError::FfmpegMissing => write!(
                f,
                "ffmpeg not found — movie export needs it. {}, then try again.",
                ffmpeg_install_hint()
            ),
        }
    }
}

impl From<std::io::Error> for MovieError {
    fn from(e: std::io::Error) -> Self {
        MovieError::Io(e)
    }
}

/// How to install ffmpeg on this platform.
pub fn ffmpeg_install_hint() -> &'static str {
    if cfg!(target_os = "macos") {
        "Install it with Homebrew: brew install ffmpeg"
    } else if cfg!(windows) {
        "Install it with: winget install ffmpeg"
    } else {
        "Install it with your package manager, e.g. sudo apt install ffmpeg \
         (Debian/Ubuntu) or sudo dnf install ffmpeg (Fedora)"
    }
}

/// Folders package managers put ffmpeg in, checked after `PATH`. Apps
/// launched from the macOS Finder get a minimal `PATH` without Homebrew's.
pub const FFMPEG_FALLBACK_DIRS: [&str; 4] = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/local/bin",
    "/usr/bin",
];

/// Locate the ffmpeg executable: on `PATH`, else in a usual install folder.
pub fn find_ffmpeg() -> Option<PathBuf> {
    find_program(
        "ffmpeg",
        std::env::var_os("PATH").as_deref(),
        FFMPEG_FALLBACK_DIRS.iter().map(Path::new),
    )
}

/// First executable `name` in the `path_var` folders, then in `fallbacks`.
pub fn find_program<'a>(
    name: &str,
    path_var: Option<&OsStr>,
    fallbacks: impl IntoIterator<Item = &'a Path>,
) -> Option<PathBuf> {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    path_var
        .into_iter()
        .flat_map(std::env::split_paths)
        .chain(fallbacks.into_iter().map(Path::to_path_buf))
        .map(|dir| dir.join(&file))
        .find(|p| is_executable(p))
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(p: &Path) -> bool {
    p.is_file()
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
        if !self.rotation.is_finite() {
            return bad("rotation must be a number");
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
        let center = center + self.drift(t * self.seconds) * span;
        Viewport::new(center, span, self.width, self.height)
    }

    /// How far frame `i` is turned, in radians.
    pub fn frame_angle(&self, i: usize) -> f64 {
        let t = i as f64 / (self.frame_count() - 1) as f64;
        self.rotation.to_radians() * t * self.seconds
    }

    /// Frame `i`'s four corners in the complex plane, turned with the
    /// frame: top-left, top-right, bottom-right, bottom-left.
    pub fn frame_corners(&self, i: usize) -> [Complex64; 4] {
        let view = self.frame_view(i);
        let angle = self.frame_angle(i);
        let (w, h) = (view.width as f64, view.height as f64);
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| rotated_point(&view, angle, x, y))
    }

    /// The camera path's offset from the straight zoom at `time` seconds,
    /// in units of the view's span. Zero at the first and last frames.
    fn drift(&self, time: f64) -> Complex64 {
        match self.path {
            CameraPath::Straight => Complex64::new(0.0, 0.0),
            CameraPath::Circle => {
                let total = self.seconds;
                let ease_in = 2.0_f64.min(total / 4.0);
                let ease_out = 4.0_f64.min(total / 4.0);
                let fade = smoothstep(0.0, ease_in, time)
                    * (1.0 - smoothstep(total - ease_out, total, time));
                let phase = std::f64::consts::TAU * time / CIRCLE_PERIOD_S;
                Complex64::from_polar(CIRCLE_RADIUS * fade, phase)
            }
        }
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
    // A plain MP4 keeps its index at the end, so nothing plays until the
    // encode finishes and a crash loses everything. Fragmented MP4 writes
    // the index up front and a fragment per keyframe: the file plays while
    // it renders and survives being cut short. (Matroska does this anyway.)
    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    if matches!(ext.as_deref(), Some("mp4" | "mov")) {
        args.push("-movflags".into());
        args.push("+frag_keyframe+empty_moov+default_base_moof".into());
    }
    args.push(output.as_os_str().to_owned());
    args
}

/// Frames rendered ahead of the encoder. Each is width×height×4 bytes.
const FRAMES_IN_FLIGHT: usize = 2;

/// After a cancel, how long ffmpeg gets to encode the frames it has and
/// close the file properly before it is killed.
const CANCEL_GRACE: Duration = Duration::from_secs(30);

/// Render a movie, streaming each frame into ffmpeg as it's done.
/// Rendering the next frame overlaps encoding the last. On cancel, the
/// frames rendered so far are kept as a shorter movie; on failure (or a
/// cancel before any frame), a partial output file this run created is
/// removed.
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

    let mut sent = 0;
    let rendered = render_frames(spec, &pool, cancel, &tx, &mut sent, progress);
    drop(tx);
    let result = match rendered {
        Err(MovieError::Cancelled { .. }) if sent > 0 => {
            // stdin is closed once the writer drains, so ffmpeg encodes
            // what it has and finalizes the file.
            progress(MovieProgress::Encoding);
            let finished = wind_down(&mut child);
            let _ = writer.join();
            Err(MovieError::Cancelled {
                kept: Some(Kept {
                    path: spec.output.clone(),
                    frames: sent,
                    finished,
                }),
            })
        }
        Ok(()) => {
            progress(MovieProgress::Encoding);
            let wrote = writer
                .join()
                .unwrap_or_else(|_| Err(std::io::Error::other("writer panicked")));
            finish(&mut child, stderr_reader, cancel, wrote).map_err(|e| match e {
                MovieError::Cancelled { kept: Some(k) } if sent > 0 => MovieError::Cancelled {
                    kept: Some(Kept {
                        path: spec.output.clone(),
                        frames: sent,
                        ..k
                    }),
                },
                MovieError::Cancelled { .. } => MovieError::Cancelled { kept: None },
                e => e,
            })
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
        Err(e @ MovieError::Cancelled { kept: Some(_) }) => Err(e),
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
    sent: &mut usize,
    progress: &mut dyn FnMut(MovieProgress),
) -> Result<(), MovieError> {
    let total = spec.frame_count();
    let mut pixels = vec![[0u8; 4]; spec.width * spec.height];
    for i in 0..total {
        if cancel.is_cancelled() {
            return Err(MovieError::Cancelled { kept: None });
        }
        let start = Instant::now();
        let view = spec.frame_view(i);
        let sampler = RotatedSampler {
            kind: spec.kind,
            max_iter: spec.max_iter.resolve(view.magnification()),
            angle: spec.frame_angle(i),
        };
        let samples =
            render(&sampler, &view, pool, cancel).ok_or(MovieError::Cancelled { kept: None })?;
        colorize(&samples, &spec.color, &mut pixels);
        if tx.send(pixels.as_flattened().to_vec()).is_err() {
            // The writer stopped: ffmpeg exited early. `finish` reports why.
            return Ok(());
        }
        *sent += 1;
        progress(MovieProgress::Frame {
            done: i + 1,
            total,
            last_frame_ms: start.elapsed().as_millis() as u64,
        });
    }
    Ok(())
}

/// Let ffmpeg exit on its own within [`CANCEL_GRACE`], else kill it.
/// Returns whether it exited cleanly, so the file holds every frame sent.
/// A killed fragmented MP4 still plays up to its last finished fragment.
fn wind_down(child: &mut Child) -> bool {
    let deadline = Instant::now() + CANCEL_GRACE;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Err(_) => return false,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    false
}

/// Wait for ffmpeg to finish encoding and turn its exit status into a
/// result. A cancel here gives it [`CANCEL_GRACE`] to finish first.
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
            // The caller fills in the file and frame count.
            let finished = wind_down(child);
            return Err(MovieError::Cancelled {
                kept: Some(Kept {
                    path: PathBuf::new(),
                    frames: 0,
                    finished,
                }),
            });
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
