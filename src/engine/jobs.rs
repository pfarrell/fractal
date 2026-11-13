//! Background render jobs.
//!
//! A [`RenderSlot`] owns one view's rendering: its own rayon thread pool,
//! a generation counter, and a channel back to the UI. Submitting a new
//! request bumps the generation, which cancels whatever is still running
//! for that slot (see [`CancelToken`]). Results from stale generations are
//! thrown away, so a burst of mouse moves can never show an old image.

use crate::engine::escape::Sample;
use crate::engine::render::{CancelToken, Sampler, render};
use crate::engine::viewport::Viewport;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What to render.
pub struct RenderRequest {
    pub sampler: Arc<dyn Sampler>,
    pub view: Viewport,
    /// Render a quarter-resolution preview first, then full resolution.
    pub progressive: bool,
}

/// One finished pass.
pub struct RenderOutput {
    pub generation: u64,
    /// The view actually rendered (smaller than requested for previews).
    pub view: Viewport,
    pub samples: Vec<Sample>,
    pub elapsed: Duration,
    pub threads: usize,
    /// `false` for the low-resolution preview pass.
    pub final_pass: bool,
}

/// Previews are only worth it for views at least this tall.
const PREVIEW_MIN_HEIGHT: usize = 64;
const PREVIEW_DIVISOR: usize = 4;

type Notify = Arc<dyn Fn() + Send + Sync>;

pub struct RenderSlot {
    pool: Arc<rayon::ThreadPool>,
    threads: usize,
    generation: Arc<AtomicU64>,
    tx: Sender<RenderOutput>,
    rx: Mutex<Receiver<RenderOutput>>,
    notify: Notify,
}

fn build_pool(threads: usize) -> Arc<rayon::ThreadPool> {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build()
            .expect("failed to build thread pool"),
    )
}

impl RenderSlot {
    /// `notify` is called (from a worker thread) after every finished pass;
    /// the UI uses it to request a repaint.
    pub fn new(threads: usize, notify: Notify) -> Self {
        let (tx, rx) = channel();
        Self {
            pool: build_pool(threads),
            threads: threads.max(1),
            generation: Arc::new(AtomicU64::new(0)),
            tx,
            rx: Mutex::new(rx),
            notify,
        }
    }

    /// Cancel any in-flight render and start this one. Returns its generation.
    pub fn submit(&self, req: RenderRequest) -> u64 {
        let generation = self.generation.fetch_add(1, Ordering::Relaxed) + 1;
        let token = CancelToken::new(self.generation.clone(), generation);
        let pool = self.pool.clone();
        let tx = self.tx.clone();
        let notify = self.notify.clone();
        let threads = self.threads;

        self.pool.spawn(move || {
            let mut passes = Vec::with_capacity(2);
            let v = req.view;
            if req.progressive && v.height >= PREVIEW_MIN_HEIGHT {
                let small = v.with_size(
                    (v.width / PREVIEW_DIVISOR).max(1),
                    (v.height / PREVIEW_DIVISOR).max(1),
                );
                passes.push((small, false));
            }
            passes.push((v, true));

            for (view, final_pass) in passes {
                let start = Instant::now();
                let Some(samples) = render(req.sampler.as_ref(), &view, &pool, &token) else {
                    return; // superseded by a newer request
                };
                let out = RenderOutput {
                    generation,
                    view,
                    samples,
                    elapsed: start.elapsed(),
                    threads,
                    final_pass,
                };
                if tx.send(out).is_err() {
                    return; // slot dropped
                }
                notify();
            }
        });
        generation
    }

    /// Cancel any in-flight render without starting a new one.
    pub fn cancel(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// All outputs received for the current generation, oldest first.
    /// Outputs from older generations are discarded.
    pub fn poll_all(&self) -> Vec<RenderOutput> {
        let current = self.generation.load(Ordering::Relaxed);
        let rx = self.rx.lock().expect("receiver lock poisoned");
        rx.try_iter().filter(|o| o.generation == current).collect()
    }

    /// The newest output for the current generation, if any arrived.
    pub fn poll(&self) -> Option<RenderOutput> {
        self.poll_all().pop()
    }

    /// Rebuild the pool with a different thread count. Cancels in-flight work.
    pub fn set_threads(&mut self, threads: usize) {
        self.cancel();
        self.threads = threads.max(1);
        self.pool = build_pool(self.threads);
    }

    pub fn threads(&self) -> usize {
        self.threads
    }
}
