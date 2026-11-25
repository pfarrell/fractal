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
//! A movie can pass through more stops than two: see
//! [`crate::engine::route`], which paces the steps and blends them at each
//! stop. Each stop's `rotation` turns the picture about the screen center
//! on the step arriving there.

use crate::engine::color::{ColorSettings, colorize};
use crate::engine::escape::{FractalKind, Sample, iterate};
use crate::engine::keyframe::{
    Block, KEYFRAME_MAX_PIXELS, Quality, frame_work, keyframe_work, plan_range, planned_work,
    render_keyframe, resample,
};
use crate::engine::render::{CancelToken, Sampler, render};
use crate::engine::route::{JOINT_HALF_WIDTH_S, Route, Stop, step_length};
use crate::engine::soundtrack::{add_music, plan_soundtrack, probe_track};
use crate::engine::viewport::{MaxIter, Viewport, min_pixel_scale};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

/// The smallest span (frame height in the complex plane) a frame
/// `height` pixels tall can have at `center` and still resolve each pixel
/// in f64.
pub fn precision_floor(center: Complex64, height: usize) -> f64 {
    min_pixel_scale(center) * height as f64
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MovieSpec {
    pub kind: FractalKind,
    /// The views the camera passes through, first to last (at least two).
    pub stops: Vec<Stop>,
    pub fps: u32,
    pub seconds: f64,
    pub width: usize,
    pub height: usize,
    pub max_iter: MaxIter,
    pub color: ColorSettings,
    pub output: PathBuf,
    /// Exact renders every frame; Fast and Smooth resample keyframes.
    #[serde(default)]
    pub quality: Quality,
    /// Tracks to add as music once the video is done, in play order.
    #[serde(default)]
    pub music: Vec<PathBuf>,
}

pub enum MovieProgress {
    /// `done` frames of `total` are out. `work` is (done, planned) in
    /// Exact samples (see [`planned_work`]), keyframes included, and
    /// `elapsed_ms` the render time so far: together, an ETA.
    Frame {
        done: usize,
        total: usize,
        work: (f64, f64),
        elapsed_ms: u64,
    },
    /// A keyframe is about to render: `done` keyframes are finished of
    /// `total`.
    Keyframe {
        done: usize,
        total: usize,
    },
    Encoding,
    /// The video is done and the music is being added.
    AddingMusic,
    Done(PathBuf),
}

/// A finished movie.
#[derive(Debug)]
pub struct Saved {
    pub path: PathBuf,
    /// Why the chosen music isn't in it, if it isn't.
    pub music_error: Option<String>,
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
    /// Why the chosen music isn't in it, if it isn't.
    pub music_error: Option<String>,
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
    /// Music was chosen but ffprobe, needed to read it, isn't installed.
    FfprobeMissing,
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
                }?;
                match &k.music_error {
                    Some(e) => write!(f, "; music not added: {e}"),
                    None => Ok(()),
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
            MovieError::FfprobeMissing => write!(
                f,
                "ffprobe not found — adding music needs it. {}, then try again.",
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
pub(crate) fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
pub(crate) fn is_executable(p: &Path) -> bool {
    p.is_file()
}

/// How much of ffmpeg's error output to keep.
const STDERR_TAIL_BYTES: usize = 2048;

/// The last [`STDERR_TAIL_BYTES`] of `err`, cut on a char boundary.
pub(crate) fn stderr_tail(err: &str) -> &str {
    let start = err.len().saturating_sub(STDERR_TAIL_BYTES);
    let start = (start..err.len())
        .find(|&i| err.is_char_boundary(i))
        .unwrap_or(err.len());
    &err[start..]
}

impl MovieSpec {
    /// Everything [`Self::validate_frames`] checks, plus the output path.
    pub fn validate(&self) -> Result<(), MovieError> {
        self.validate_frames()?;
        self.validate_output()
    }

    /// The route, size, rate and duration are renderable (the output path
    /// is not checked: a preview has none).
    pub fn validate_frames(&self) -> Result<(), MovieError> {
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
        if self.stops.len() < 2 {
            return bad("a movie needs at least two stops");
        }
        if !self.stops.iter().all(|s| s.rotation.is_finite()) {
            return bad("rotation must be a number");
        }
        if !self
            .stops
            .iter()
            .all(|s| s.span.is_finite() && s.span > 0.0)
        {
            return bad("zoom spans must be positive");
        }
        // Two stops may be the same view (the movie just turns in place);
        // in a longer route a zero-length step would get no time at all.
        if self.stops.len() > 2
            && self
                .stops
                .windows(2)
                .any(|w| step_length(&w[0], &w[1]) == 0.0)
        {
            return bad("two stops in a row are the same view");
        }
        let last = self.stops.len() - 1;
        for (k, s) in self.stops.iter().enumerate() {
            // Small tolerance so "as deep as allowed" survives float rounding.
            if s.span < precision_floor(s.center, self.height) * (1.0 - 1e-9) {
                return Err(MovieError::Invalid(if k == last {
                    "end zoom is past the f64 precision limit for this point".into()
                } else {
                    format!("stop {} is past the f64 precision limit", k + 1)
                }));
            }
        }
        Ok(())
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

    /// Raise every stop deeper than its [`precision_floor`] at this height
    /// to that floor, so the movie renders as deep as f64 allows rather
    /// than failing validation. Returns the indices of the stops raised.
    pub fn clamp_depth(&mut self) -> Vec<usize> {
        let height = self.height;
        let mut raised = Vec::new();
        for (k, s) in self.stops.iter_mut().enumerate() {
            let floor = precision_floor(s.center, height);
            if s.span < floor {
                s.span = floor;
                raised.push(k);
            }
        }
        raised
    }

    pub fn frame_count(&self) -> usize {
        ((self.fps as f64 * self.seconds).round() as usize).max(2)
    }

    /// The camera route through `stops` over the movie's duration.
    pub fn route(&self) -> Route {
        Route::new(&self.stops, self.seconds, JOINT_HALF_WIDTH_S)
    }

    /// Movie time of frame `i`, in seconds.
    pub fn frame_time(&self, i: usize) -> f64 {
        i as f64 / (self.frame_count() - 1) as f64 * self.seconds
    }

    pub fn frame_view(&self, i: usize) -> Viewport {
        let (center, span) = self.route().view_at(self.frame_time(i));
        Viewport::new(center, span, self.width, self.height)
    }

    /// How far frame `i` is turned, in radians.
    pub fn frame_angle(&self, i: usize) -> f64 {
        self.route().angle_at(self.frame_time(i))
    }

    /// Frame `i`'s four corners in the complex plane, turned with the
    /// frame: top-left, top-right, bottom-right, bottom-left.
    pub fn frame_corners(&self, i: usize) -> [Complex64; 4] {
        let view = self.frame_view(i);
        let angle = self.frame_angle(i);
        let (w, h) = (view.width as f64, view.height as f64);
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| rotated_point(&view, angle, x, y))
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
/// removed. Chosen music (read with `ffprobe`) is checked before the first
/// frame and added afterwards, to a kept cancelled movie too; if adding it
/// fails, the movie is kept without it.
pub fn render_movie(
    spec: &MovieSpec,
    threads: usize,
    ffmpeg: &Path,
    ffprobe: Option<&Path>,
    cancel: &CancelToken,
    progress: &mut dyn FnMut(MovieProgress),
) -> Result<Saved, MovieError> {
    spec.validate()?;
    let music = check_music(spec, ffprobe)?;
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
                    music_error: None,
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
            let music_error = match &music {
                None => None,
                Some(m) => {
                    progress(MovieProgress::AddingMusic);
                    match m.add(ffmpeg, &spec.output, cancel) {
                        Ok(()) => None,
                        Err(e) if e == "cancelled" => {
                            return Err(MovieError::Cancelled {
                                kept: Some(Kept {
                                    path: spec.output.clone(),
                                    frames: sent,
                                    finished: true,
                                    music_error: Some(e),
                                }),
                            });
                        }
                        Err(e) => Some(e),
                    }
                }
            };
            progress(MovieProgress::Done(spec.output.clone()));
            Ok(Saved {
                path: spec.output.clone(),
                music_error,
            })
        }
        Err(MovieError::Cancelled { kept: Some(mut k) }) => {
            if let Some(m) = &music {
                // The first Cancel stopped the render; another stops this.
                progress(MovieProgress::AddingMusic);
                k.music_error = m.add(ffmpeg, &spec.output, &cancel.renewed()).err();
            }
            Err(MovieError::Cancelled { kept: Some(k) })
        }
        Err(e) => {
            if !existed_before {
                let _ = std::fs::remove_file(&spec.output);
            }
            Err(e)
        }
    }
}

/// Chosen music, read and checked before rendering.
struct Music<'a> {
    ffprobe: &'a Path,
    tracks: &'a [PathBuf],
    secs: Vec<f64>,
}

impl Music<'_> {
    fn add(&self, ffmpeg: &Path, video: &Path, cancel: &CancelToken) -> Result<(), String> {
        add_music(ffmpeg, self.ffprobe, video, self.tracks, &self.secs, cancel)
    }
}

/// Read every chosen track's length and check the playlist can fill the
/// movie, so a bad track fails before any rendering. `None`: no music.
fn check_music<'a>(
    spec: &'a MovieSpec,
    ffprobe: Option<&'a Path>,
) -> Result<Option<Music<'a>>, MovieError> {
    if spec.music.is_empty() {
        return Ok(None);
    }
    let ffprobe = ffprobe.ok_or(MovieError::FfprobeMissing)?;
    let secs = spec
        .music
        .iter()
        .map(|t| probe_track(ffprobe, t))
        .collect::<Result<Vec<f64>, String>>()
        .map_err(MovieError::Invalid)?;
    plan_soundtrack(&secs, spec.seconds).map_err(MovieError::Invalid)?;
    Ok(Some(Music {
        ffprobe,
        tracks: &spec.music,
        secs,
    }))
}

/// Render frame `i` the exact way: one `iterate` per pixel. `out` is
/// `width × height`.
pub fn render_exact_frame(
    spec: &MovieSpec,
    i: usize,
    pool: &rayon::ThreadPool,
    cancel: &CancelToken,
    out: &mut [[u8; 4]],
) -> Result<(), MovieError> {
    let view = spec.frame_view(i);
    let sampler = RotatedSampler {
        kind: spec.kind,
        max_iter: spec.max_iter.resolve(view.magnification()),
        angle: spec.frame_angle(i),
    };
    let samples =
        render(&sampler, &view, pool, cancel).ok_or(MovieError::Cancelled { kept: None })?;
    colorize(&samples, &spec.color, out);
    Ok(())
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
    let mut to_ffmpeg = |_: usize, pixels: &[[u8; 4]]| {
        // A failed send means the writer stopped: ffmpeg exited early, and
        // `finish` reports why.
        let ok = tx.send(pixels.as_flattened().to_vec()).is_ok();
        if ok {
            *sent += 1;
        }
        ok
    };
    render_range(
        spec,
        0..spec.frame_count(),
        pool,
        cancel,
        &mut to_ffmpeg,
        progress,
    )
}

/// Render the frames in `frames`, in order, handing each to `sink` with
/// its index. `sink` returning false stops the render (`Ok`). Progress
/// counts frames and planned work within `frames`.
pub fn render_range(
    spec: &MovieSpec,
    frames: Range<usize>,
    pool: &rayon::ThreadPool,
    cancel: &CancelToken,
    sink: &mut dyn FnMut(usize, &[[u8; 4]]) -> bool,
    progress: &mut dyn FnMut(MovieProgress),
) -> Result<(), MovieError> {
    let total = frames.len();
    let cancelled = || MovieError::Cancelled { kept: None };
    let mut pixels = vec![[0u8; 4]; spec.width * spec.height];
    let started = Instant::now();
    let elapsed_ms = || started.elapsed().as_millis() as u64;
    let mut done = 0;
    let Some(q) = spec.quality.q() else {
        let px = (spec.width * spec.height) as f64;
        for i in frames {
            if cancel.is_cancelled() {
                return Err(cancelled());
            }
            render_exact_frame(spec, i, pool, cancel, &mut pixels)?;
            if !sink(i, &pixels) {
                return Ok(());
            }
            done += 1;
            progress(MovieProgress::Frame {
                done,
                total,
                work: (done as f64 * px, total as f64 * px),
                elapsed_ms: elapsed_ms(),
            });
        }
        return Ok(());
    };

    let blocks = plan_range(spec, frames, q, KEYFRAME_MAX_PIXELS);
    let keyframes = blocks
        .iter()
        .filter(|b| matches!(b, Block::Keyframe(_)))
        .count();
    let mut keyframes_done = 0;
    let work_total = planned_work(spec, &blocks);
    let mut work_done = 0.0;
    for block in &blocks {
        let (frames, keyframe) = match block {
            Block::Exact(frames) => (frames.clone(), None),
            Block::Keyframe(stretch) => {
                if cancel.is_cancelled() {
                    return Err(cancelled());
                }
                progress(MovieProgress::Keyframe {
                    done: keyframes_done,
                    total: keyframes,
                });
                keyframes_done += 1;
                let kf = render_keyframe(spec.kind, stretch, pool, cancel).ok_or_else(cancelled)?;
                work_done += keyframe_work(stretch);
                (stretch.frames.clone(), Some(kf))
            }
        };
        for i in frames {
            if cancel.is_cancelled() {
                return Err(cancelled());
            }
            match &keyframe {
                None => render_exact_frame(spec, i, pool, cancel, &mut pixels)?,
                Some(kf) => {
                    let view = spec.frame_view(i);
                    let max_iter = spec.max_iter.resolve(view.magnification());
                    resample(
                        kf,
                        &view,
                        spec.frame_angle(i),
                        max_iter,
                        &spec.color,
                        pool,
                        &mut pixels,
                    );
                }
            }
            if !sink(i, &pixels) {
                return Ok(());
            }
            done += 1;
            work_done += frame_work(spec, block, i);
            progress(MovieProgress::Frame {
                done,
                total,
                work: (work_done, work_total),
                elapsed_ms: elapsed_ms(),
            });
        }
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
                    music_error: None,
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
    Err(MovieError::Ffmpeg {
        status: status.code(),
        stderr_tail: stderr_tail(&err).to_string(),
    })
}
