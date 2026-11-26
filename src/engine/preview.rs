//! Previews: a movie's path rendered small, choppy and fast, in windows
//! of movie time, reusing every frame whose picture hasn't changed.

use crate::engine::color::ColorSettings;
use crate::engine::escape::FractalKind;
use crate::engine::keyframe::Quality;
use crate::engine::movie::{MovieError, MovieProgress, MovieSpec, render_range};
use crate::engine::render::CancelToken;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::thread::JoinHandle;

/// A preview fits the final movie's shape inside this box.
pub const PREVIEW_BOX: (usize, usize) = (320, 180);
pub const PREVIEW_FPS: u32 = 12;
/// Movie time held at once (≈ 600 MB of RGB frames at 320×180).
pub const WINDOW_SECONDS: f64 = 300.0;

/// `width × height` scaled down to fit [`PREVIEW_BOX`] (never up), each
/// side even and at least 16.
pub fn preview_size(width: usize, height: usize) -> (usize, usize) {
    let (bw, bh) = PREVIEW_BOX;
    let scale = (bw as f64 / width.max(1) as f64)
        .min(bh as f64 / height.max(1) as f64)
        .min(1.0);
    let side = |v: usize| (((v as f64 * scale) / 2.0).round() as usize * 2).max(16);
    (side(width), side(height))
}

/// The preview of `final_spec`: same path, timing, rotations, iterations
/// and colors, at preview size and rate, Fast.
pub fn preview_spec(final_spec: &MovieSpec) -> MovieSpec {
    let (width, height) = preview_size(final_spec.width, final_spec.height);
    MovieSpec {
        width,
        height,
        fps: PREVIEW_FPS,
        quality: Quality::Fast,
        ..final_spec.clone()
    }
}

/// Frames in one window.
fn window_len(spec: &MovieSpec) -> usize {
    (WINDOW_SECONDS * spec.fps as f64).round() as usize
}

/// The window starting at `start_frame`, clipped to the movie; a start
/// too near the end gives the last full window, so it is never empty.
pub fn window_at(spec: &MovieSpec, start_frame: usize) -> Range<usize> {
    let (n, len) = (spec.frame_count(), window_len(spec));
    if n <= len {
        return 0..n;
    }
    let start = start_frame.min(n - len);
    start..start + len
}

/// The frame shown at movie time `t`.
pub fn frame_at_time(spec: &MovieSpec, t: f64) -> usize {
    let n = spec.frame_count();
    let i = (t / spec.seconds * (n - 1) as f64).round();
    (i.max(0.0) as usize).min(n - 1)
}

/// Everything a frame's picture depends on. Equal fingerprints render the
/// same picture.
#[derive(Clone, Debug, PartialEq)]
pub struct Fingerprint {
    center: crate::engine::deep::HiComplex,
    span: f64,
    angle: f64,
    max_iter: u32,
    kind: FractalKind,
    color: ColorSettings,
    size: (usize, usize),
}

pub fn fingerprint(spec: &MovieSpec, i: usize) -> Fingerprint {
    let view = spec.frame_view(i);
    Fingerprint {
        center: view.center.clone(),
        span: view.span,
        angle: spec.frame_angle(i),
        max_iter: spec.max_iter.resolve(view.magnification()),
        kind: spec.kind,
        color: spec.color,
        size: (spec.width, spec.height),
    }
}

/// The longest runs of frames in `window` that `have` says are missing.
pub fn runs_to_render(window: Range<usize>, have: impl Fn(usize) -> bool) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    for i in window.filter(|&i| !have(i)) {
        match runs.last_mut() {
            Some(r) if r.end == i => r.end += 1,
            _ => runs.push(i..i + 1),
        }
    }
    runs
}

/// Messages from a preview job, tagged with its cancel token's
/// generation so a replaced job's late messages can be ignored.
pub enum PreviewMsg {
    /// Frame `index`, RGB, `width × height × 3` bytes.
    Frame {
        generation: u64,
        index: usize,
        rgb: Arc<[u8]>,
    },
    /// `done` of `total` frames (over all runs) are out.
    Progress {
        generation: u64,
        done: usize,
        total: usize,
    },
    Finished {
        generation: u64,
        result: Result<(), MovieError>,
    },
}

/// Render `runs` of `spec`'s frames on a thread of their own with
/// `threads` workers, sending each frame as it is done. `wake` is called
/// after every message (to repaint the UI).
pub fn spawn_preview(
    spec: MovieSpec,
    runs: Vec<Range<usize>>,
    threads: usize,
    cancel: CancelToken,
    tx: Sender<PreviewMsg>,
    wake: impl Fn() + Send + 'static,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let generation = cancel.generation();
        let total: usize = runs.iter().map(|r| r.len()).sum();
        let send = |m: PreviewMsg| {
            let _ = tx.send(m);
            wake();
        };
        let result = (|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.max(1))
                .build()
                .map_err(|e| MovieError::Invalid(e.to_string()))?;
            let mut done = 0;
            for run in runs {
                if cancel.is_cancelled() {
                    return Err(MovieError::Cancelled { kept: None });
                }
                render_range(
                    &spec,
                    run,
                    &pool,
                    &cancel,
                    &mut |index, pixels| {
                        let rgb: Arc<[u8]> =
                            pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
                        send(PreviewMsg::Frame {
                            generation,
                            index,
                            rgb,
                        });
                        done += 1;
                        send(PreviewMsg::Progress {
                            generation,
                            done,
                            total,
                        });
                        true
                    },
                    &mut |_: MovieProgress| {},
                )?;
            }
            Ok(())
        })();
        send(PreviewMsg::Finished { generation, result });
    })
}
