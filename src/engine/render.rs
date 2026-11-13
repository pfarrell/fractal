//! Parallel rendering.
//!
//! The image is split into rows and rayon hands the rows out to its worker
//! threads. Rows are a good unit of work: each one is independent (no
//! shared state except the output buffer, and each thread writes only its
//! own row), and there are enough of them that rayon's work-stealing keeps
//! every core busy even though rows through the set take far longer than
//! rows outside it.

use crate::engine::escape::{FractalKind, Sample, iterate};
use crate::engine::viewport::Viewport;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Computes one pixel. This is the seam where a future deep-zoom
/// (arbitrary precision / perturbation) sampler can plug in.
pub trait Sampler: Send + Sync {
    /// Sample pixel (x, y) of `view` (at the pixel's center).
    fn sample(&self, view: &Viewport, x: usize, y: usize) -> Sample;
}

/// The plain f64 sampler: one `iterate` call per pixel.
#[derive(Clone, Copy, Debug)]
pub struct F64Sampler {
    pub kind: FractalKind,
    pub max_iter: u32,
}

impl Sampler for F64Sampler {
    fn sample(&self, view: &Viewport, x: usize, y: usize) -> Sample {
        let point = view.pixel_to_complex(x as f64 + 0.5, y as f64 + 0.5);
        iterate(&self.kind, point, self.max_iter)
    }
}

/// Cancellation by generation number: whoever owns `generation` bumps it
/// to cancel every render started with an older number.
#[derive(Clone, Debug)]
pub struct CancelToken {
    generation: Arc<AtomicU64>,
    mine: u64,
}

impl CancelToken {
    pub fn new(generation: Arc<AtomicU64>, mine: u64) -> Self {
        Self { generation, mine }
    }

    /// A token that is never cancelled.
    pub fn never() -> Self {
        Self::new(Arc::new(AtomicU64::new(0)), 0)
    }

    /// The generation this token belongs to.
    pub fn generation(&self) -> u64 {
        self.mine
    }

    pub fn is_cancelled(&self) -> bool {
        self.generation.load(Ordering::Relaxed) != self.mine
    }
}

/// How many pixels to compute between cancellation checks. A whole row
/// can take hundreds of milliseconds deep inside the set, so checking only
/// once per row would make cancellation feel sluggish.
const CANCEL_CHECK_PIXELS: usize = 64;

/// Render `view` on `pool`, one rayon task per row. Returns `None` if
/// cancelled; cancellation is checked every [`CANCEL_CHECK_PIXELS`]
/// pixels, so it takes effect almost immediately.
pub fn render(
    sampler: &dyn Sampler,
    view: &Viewport,
    pool: &rayon::ThreadPool,
    cancel: &CancelToken,
) -> Option<Vec<Sample>> {
    if view.is_empty() {
        return Some(Vec::new());
    }
    let mut buf = vec![Sample::Inside; view.width * view.height];
    pool.install(|| {
        buf.par_chunks_mut(view.width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, px) in row.iter_mut().enumerate() {
                    if x % CANCEL_CHECK_PIXELS == 0 && cancel.is_cancelled() {
                        return;
                    }
                    *px = sampler.sample(view, x, y);
                }
            });
    });
    (!cancel.is_cancelled()).then_some(buf)
}

/// Single-threaded reference renderer (used to check the parallel one).
pub fn render_serial(sampler: &dyn Sampler, view: &Viewport) -> Vec<Sample> {
    let mut buf = Vec::with_capacity(view.width * view.height);
    for y in 0..view.height {
        for x in 0..view.width {
            buf.push(sampler.sample(view, x, y));
        }
    }
    buf
}
