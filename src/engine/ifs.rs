//! Iterated function systems and the chaos game.
//!
//! An IFS is a handful of contracting affine maps. Its attractor is the one
//! shape that equals the union of its images under all the maps: a fern
//! made of smaller ferns. The chaos game finds it by starting anywhere and
//! repeatedly applying a randomly chosen map; after a few steps every point
//! lies (to within a pixel) on the attractor.

use crate::engine::render::CancelToken;
use crate::engine::viewport::Viewport;
use num_complex::Complex64;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// (x, y) → (a·x + b·y + e, c·x + d·y + f), chosen with weight `p`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AffineMap {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
    pub p: f64,
}

impl AffineMap {
    pub const fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, p: f64) -> Self {
        Self {
            a,
            b,
            c,
            d,
            e,
            f,
            p,
        }
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.e,
            self.c * x + self.d * y + self.f,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IfsSystem {
    pub name: String,
    pub maps: Vec<AffineMap>,
}

/// Points each chaos-game run throws away while it lands on the attractor.
const WARMUP: usize = 20;
/// Points between cancellation checks.
const CANCEL_CHECK_POINTS: u64 = 65_536;
const BOUNDS_POINTS: usize = 20_000;

/// Picks maps according to cumulative probabilities.
struct Chooser {
    cumulative: Vec<f64>,
}

impl Chooser {
    fn new(sys: &IfsSystem) -> Self {
        let mut acc = 0.0;
        let cumulative = sys
            .normalized_probabilities()
            .into_iter()
            .map(|p| {
                acc += p;
                acc
            })
            .collect();
        Self { cumulative }
    }

    fn pick(&self, r: f64) -> usize {
        self.cumulative
            .iter()
            .position(|&c| r < c)
            .unwrap_or(self.cumulative.len() - 1)
    }
}

impl IfsSystem {
    pub fn sierpinski() -> Self {
        let p = 1.0 / 3.0;
        Self {
            name: "Sierpinski triangle".into(),
            maps: vec![
                AffineMap::new(0.5, 0.0, 0.0, 0.5, 0.0, 0.0, p),
                AffineMap::new(0.5, 0.0, 0.0, 0.5, 0.5, 0.0, p),
                AffineMap::new(0.5, 0.0, 0.0, 0.5, 0.25, 0.5, p),
            ],
        }
    }

    pub fn barnsley_fern() -> Self {
        Self {
            name: "Barnsley fern".into(),
            maps: vec![
                AffineMap::new(0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
                AffineMap::new(0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
                AffineMap::new(0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
                AffineMap::new(-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07),
            ],
        }
    }

    pub fn heighway_dragon() -> Self {
        Self {
            name: "Heighway dragon".into(),
            maps: vec![
                AffineMap::new(0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.5),
                AffineMap::new(-0.5, -0.5, 0.5, -0.5, 1.0, 0.0, 0.5),
            ],
        }
    }

    pub fn sierpinski_carpet() -> Self {
        let third = 1.0 / 3.0;
        let mut maps = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                if (i, j) != (1, 1) {
                    let (e, f) = (i as f64 * third, j as f64 * third);
                    maps.push(AffineMap::new(third, 0.0, 0.0, third, e, f, 1.0 / 8.0));
                }
            }
        }
        Self {
            name: "Sierpinski carpet".into(),
            maps,
        }
    }

    pub fn presets() -> Vec<Self> {
        vec![
            Self::sierpinski(),
            Self::barnsley_fern(),
            Self::heighway_dragon(),
            Self::sierpinski_carpet(),
        ]
    }

    /// Probabilities cleaned up for sampling: negative or NaN weights count
    /// as 0; if nothing is left, all maps are equally likely; then scaled
    /// to sum to 1.
    pub fn normalized_probabilities(&self) -> Vec<f64> {
        let clean: Vec<f64> = self
            .maps
            .iter()
            .map(|m| {
                if m.p.is_finite() && m.p > 0.0 {
                    m.p
                } else {
                    0.0
                }
            })
            .collect();
        let sum: f64 = clean.iter().sum();
        if sum <= 0.0 {
            let n = self.maps.len().max(1) as f64;
            return vec![1.0 / n; self.maps.len()];
        }
        clean.iter().map(|p| p / sum).collect()
    }

    /// (min_x, min_y, max_x, max_y) of the attractor, estimated from a
    /// short single-threaded chaos game.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        if self.maps.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let chooser = Chooser::new(self);
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut x, mut y) = (0.0, 0.0);
        let mut b = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for i in 0..BOUNDS_POINTS + WARMUP {
            (x, y) = self.maps[chooser.pick(rng.random())].apply(x, y);
            if i >= WARMUP && x.is_finite() && y.is_finite() {
                b = (b.0.min(x), b.1.min(y), b.2.max(x), b.3.max(y));
            }
        }
        if b.0 > b.2 {
            return (0.0, 0.0, 0.0, 0.0); // diverged: nothing finite seen
        }
        b
    }

    /// A view framing the attractor with a 5% margin (x → re, y → im).
    pub fn fit_view(&self, width: usize, height: usize) -> Viewport {
        let (x0, y0, x1, y1) = self.bounds();
        let aspect = width.max(1) as f64 / height.max(1) as f64;
        let span = ((y1 - y0).max((x1 - x0) / aspect) * 1.05).max(1e-6);
        let center = Complex64::new((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        Viewport::new(center, span, width, height)
    }

    /// Map `i` applied to the corners of [`Self::bounds`]: shows where that
    /// map puts its shrunken copy of the whole picture. Corner order:
    /// (min,min), (max,min), (max,max), (min,max).
    pub fn map_outline(&self, i: usize) -> [(f64, f64); 4] {
        let (x0, y0, x1, y1) = self.bounds();
        let m = &self.maps[i];
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|(x, y)| m.apply(x, y))
    }
}

/// Hit counts per pixel.
pub struct Density {
    pub width: usize,
    pub height: usize,
    pub counts: Vec<u32>,
}

/// Play the chaos game with `points` points, in parallel on `pool`.
///
/// Each thread runs its own independent game with its own seeded RNG and
/// its own count buffer (no sharing while running); the buffers are summed
/// at the end. Same seed + same thread count ⇒ same picture.
pub fn chaos_game(
    sys: &IfsSystem,
    view: &Viewport,
    points: u64,
    seed: u64,
    pool: &rayon::ThreadPool,
    cancel: &CancelToken,
) -> Option<Density> {
    let (w, h) = (view.width, view.height);
    let empty = || vec![0u32; w * h];
    if sys.maps.is_empty() || view.is_empty() {
        return Some(Density {
            width: w,
            height: h,
            counts: empty(),
        });
    }
    let chooser = Chooser::new(sys);
    let chunks = pool.current_num_threads().max(1) as u64;

    let counts = pool.install(|| {
        (0..chunks)
            .into_par_iter()
            .map(|k| {
                let mut counts = empty();
                let n = points / chunks + if k == chunks - 1 { points % chunks } else { 0 };
                let mut rng = SmallRng::seed_from_u64(seed + k);
                let (mut x, mut y) = (0.0, 0.0);
                for _ in 0..WARMUP {
                    (x, y) = sys.maps[chooser.pick(rng.random())].apply(x, y);
                }
                for i in 0..n {
                    if i % CANCEL_CHECK_POINTS == 0 && cancel.is_cancelled() {
                        break;
                    }
                    (x, y) = sys.maps[chooser.pick(rng.random())].apply(x, y);
                    let (px, py) = view.complex_to_pixel(Complex64::new(x, y));
                    if px >= 0.0 && py >= 0.0 && (px as usize) < w && (py as usize) < h {
                        counts[py as usize * w + px as usize] += 1;
                    }
                }
                counts
            })
            .reduce(empty, |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            })
    });
    (!cancel.is_cancelled()).then_some(Density {
        width: w,
        height: h,
        counts,
    })
}
