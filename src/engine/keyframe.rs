//! Keyframe movies: frames grouped into stretches, each resampled from
//! one large keyframe whose samples are packed into 4 bytes so a movie
//! frame can be colored with its own iteration cutoff.
use crate::engine::color::{ColorSettings, Palette, sample_color};
use crate::engine::escape::{FractalKind, Sample};
use crate::engine::movie::MovieSpec;
#[cfg(doc)]
use crate::engine::movie::rotated_point;
use crate::engine::render::{CancelToken, F64Sampler, Sampler, render};
use crate::engine::viewport::{Viewport, min_pixel_scale};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::LazyLock;

/// How a movie's frames are made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quality {
    /// Every frame rendered on its own.
    #[default]
    Exact,
    /// Frames resampled from keyframes, one sample per deepest-frame pixel.
    Fast,
    /// Like Fast with 2×2 samples per deepest-frame pixel.
    Smooth,
}

impl Quality {
    pub const ALL: [Quality; 3] = [Quality::Exact, Quality::Fast, Quality::Smooth];

    pub fn label(&self) -> &'static str {
        match self {
            Quality::Exact => "Exact",
            Quality::Fast => "Fast",
            Quality::Smooth => "Smooth",
        }
    }

    /// Keyframe samples per deepest-frame pixel, per axis. `None`: Exact.
    pub fn q(&self) -> Option<u32> {
        match self {
            Quality::Exact => None,
            Quality::Fast => Some(1),
            Quality::Smooth => Some(2),
        }
    }
}

/// The most samples one keyframe may hold.
pub const KEYFRAME_MAX_PIXELS: usize = 80_000_000;

/// Largest span / smallest span allowed within one stretch.
const MAX_SPAN_RATIO: f64 = 2.0;

/// Margin around a stretch's frames, in keyframe pixels, for the
/// resampling filter.
const MARGIN_PIXELS: f64 = 2.0;

/// Consecutive frames resampled from one keyframe.
#[derive(Clone, Debug, PartialEq)]
pub struct Stretch {
    pub frames: Range<usize>,
    /// The keyframe: axis-aligned, covering every frame in `frames`.
    pub view: Viewport,
    /// Samples per deepest-frame pixel, per axis.
    pub q: u32,
    /// Iterations for the keyframe: the deepest frame's budget.
    pub max_iter: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Keyframe(Stretch),
    /// Frames too big for any keyframe, rendered the Exact way.
    Exact(Range<usize>),
}

/// What the frames of a growing stretch cover.
#[derive(Clone, Copy)]
struct Bounds {
    lo: Complex64,
    hi: Complex64,
    min_span: f64,
    max_span: f64,
    /// The frame with the smallest span.
    deepest: usize,
}

impl Bounds {
    fn of(spec: &MovieSpec, i: usize) -> Bounds {
        let span = spec.frame_view(i).span;
        let empty = Bounds {
            lo: Complex64::new(f64::INFINITY, f64::INFINITY),
            hi: Complex64::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
            min_span: span,
            max_span: span,
            deepest: i,
        };
        empty.with(spec, i)
    }

    fn with(mut self, spec: &MovieSpec, i: usize) -> Bounds {
        for c in spec.frame_corners(i) {
            self.lo = Complex64::new(self.lo.re.min(c.re), self.lo.im.min(c.im));
            self.hi = Complex64::new(self.hi.re.max(c.re), self.hi.im.max(c.im));
        }
        let span = spec.frame_view(i).span;
        if span < self.min_span {
            self.min_span = span;
            self.deepest = i;
        }
        self.max_span = self.max_span.max(span);
        self
    }

    /// The keyframe covering these frames, `q` samples per deepest-frame
    /// pixel but no finer than f64 can resolve.
    fn keyframe(&self, frame_height: usize, q: u32) -> Viewport {
        let center = (self.lo + self.hi) / 2.0;
        let spacing = (self.min_span / frame_height as f64 / q as f64).max(min_pixel_scale(center));
        let size = self.hi - self.lo;
        let cells = |extent: f64| (extent / spacing + 2.0 * MARGIN_PIXELS).ceil() as usize;
        let (width, height) = (cells(size.re), cells(size.im));
        Viewport::new(center, height as f64 * spacing, width, height)
    }
}

fn fits(view: &Viewport, max_pixels: usize) -> bool {
    view.width.saturating_mul(view.height) <= max_pixels
}

/// Group `spec`'s frames into stretches, greedily: each grows while its
/// span ratio stays within 2 and its keyframe within `max_pixels`. A
/// frame whose keyframe is too big even alone drops to `q` = 1, and
/// failing that is rendered Exact. The blocks cover every frame in order.
pub fn plan(spec: &MovieSpec, q: u32, max_pixels: usize) -> Vec<Block> {
    let (n, h) = (spec.frame_count(), spec.height);
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < n {
        let mut bounds = Bounds::of(spec, i);
        let Some(q) = [q, 1]
            .into_iter()
            .find(|&q| fits(&bounds.keyframe(h, q), max_pixels))
        else {
            match blocks.last_mut() {
                Some(Block::Exact(r)) => r.end = i + 1,
                _ => blocks.push(Block::Exact(i..i + 1)),
            }
            i += 1;
            continue;
        };
        let mut end = i + 1;
        while end < n {
            let grown = bounds.with(spec, end);
            if grown.max_span / grown.min_span > MAX_SPAN_RATIO
                || !fits(&grown.keyframe(h, q), max_pixels)
            {
                break;
            }
            bounds = grown;
            end += 1;
        }
        let magnification = spec.frame_view(bounds.deepest).magnification();
        blocks.push(Block::Keyframe(Stretch {
            frames: i..end,
            view: bounds.keyframe(h, q),
            q,
            max_iter: spec.max_iter.resolve(magnification),
        }));
        i = end;
    }
    blocks
}

/// One resampling tap's cost, in Exact samples. Measured on 1080p Smooth
/// zooms into seahorse valley (release, 12 threads): a tap takes 2.8–3.2
/// ns of wall time at any depth; an Exact sample 160–290 ns down to 1e7
/// and ~1 200 ns past it, as more of the view escapes slowly. 3.0 ns over
/// a moderate-depth 230 ns. Deep down that overstates resampling, but
/// there keyframes dominate the cost.
pub const TAP_COST: f64 = 0.013;

/// Work, in Exact samples, of rendering stretch `s`'s keyframe.
pub fn keyframe_work(s: &Stretch) -> f64 {
    (s.view.width * s.view.height) as f64
}

/// Work, in Exact samples, of making frame `i` of `block` once any
/// keyframe is rendered.
pub fn frame_work(spec: &MovieSpec, block: &Block, i: usize) -> f64 {
    match block {
        Block::Exact(_) => (spec.width * spec.height) as f64,
        Block::Keyframe(s) => {
            let n = taps_per_axis(&spec.frame_view(i), &s.view);
            (n * n * spec.width * spec.height) as f64 * TAP_COST
        }
    }
}

/// Taps per axis when resampling `frame` from keyframe view `k`: frame
/// pixel / keyframe pixel, rounded up.
fn taps_per_axis(frame: &Viewport, k: &Viewport) -> usize {
    // Ratios a hair over a whole number are float noise, not another tap.
    ((frame.pixel_scale() / k.pixel_scale() - 1e-9).ceil() as usize).max(1)
}

/// Work, in Exact samples, of every block: keyframes and frames.
pub fn planned_work(spec: &MovieSpec, blocks: &[Block]) -> f64 {
    blocks
        .iter()
        .map(|b| {
            let frames = match b {
                Block::Keyframe(s) => s.frames.clone(),
                Block::Exact(r) => r.clone(),
            };
            let kf = match b {
                Block::Keyframe(s) => keyframe_work(s),
                Block::Exact(_) => 0.0,
            };
            kf + frames.map(|i| frame_work(spec, b, i)).sum::<f64>()
        })
        .sum()
}

/// Planned work for `spec` relative to Exact's: keyframe samples, taps
/// and frames rendered Exact, over every frame's pixels.
pub fn estimated_cost(spec: &MovieSpec) -> f64 {
    let Some(q) = spec.quality.q() else {
        return 1.0;
    };
    let exact = (spec.width * spec.height * spec.frame_count()) as f64;
    planned_work(spec, &plan(spec, q, KEYFRAME_MAX_PIXELS)) / exact
}

const TAG_SHIFT: u32 = 30;
const TAG_ESCAPED: u32 = 1;
const TAG_CONVERGED: u32 = 2;
const ESC_ITER_MAX: u32 = (1 << 22) - 1;
const CONV_ITER_MAX: u32 = (1 << 24) - 1;

/// One sample in 4 bytes. Top 2 bits tag: 00 Inside, 01 Escaped
/// (22-bit iter, 8-bit smooth fraction), 10 Converged (6-bit root,
/// 24-bit iter). Iteration counts saturate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packed(u32);

impl Packed {
    pub fn pack(s: Sample) -> Packed {
        Packed(match s {
            Sample::Inside => 0,
            Sample::Escaped { iter, smooth } => {
                let iter = iter.min(ESC_ITER_MAX);
                let frac = ((smooth - iter as f32).clamp(0.0, 255.0 / 256.0) * 256.0).round();
                TAG_ESCAPED << TAG_SHIFT | iter << 8 | frac as u32
            }
            Sample::Converged { root, iter } => {
                TAG_CONVERGED << TAG_SHIFT | (root as u32 & 0x3f) << 24 | iter.min(CONV_ITER_MAX)
            }
        })
    }

    /// Unpack, turning anything that took more than `max_iter` steps into
    /// `Inside`.
    pub fn cut(self, max_iter: u32) -> Sample {
        let v = self.0;
        match v >> TAG_SHIFT {
            TAG_ESCAPED => {
                let iter = (v >> 8) & ESC_ITER_MAX;
                if iter > max_iter {
                    return Sample::Inside;
                }
                Sample::Escaped {
                    iter,
                    smooth: iter as f32 + (v & 0xff) as f32 / 256.0,
                }
            }
            TAG_CONVERGED => {
                let iter = v & CONV_ITER_MAX;
                if iter > max_iter {
                    return Sample::Inside;
                }
                Sample::Converged {
                    root: ((v >> 24) & 0x3f) as u8,
                    iter,
                }
            }
            _ => Sample::Inside,
        }
    }
}

pub fn pack(s: Sample) -> Packed {
    Packed::pack(s)
}

/// Color a packed sample as `colorize` would its unpacked, cut sample.
pub fn tap_color(p: Packed, max_iter: u32, color: &ColorSettings, palette: &Palette) -> [u8; 4] {
    sample_color(p.cut(max_iter), color, palette)
}

/// Linear light for every sRGB byte.
static SRGB_TO_LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|v| {
        let c = v as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    })
});

pub fn srgb_to_linear(v: u8) -> f32 {
    SRGB_TO_LINEAR[v as usize]
}

/// For each sRGB byte `v` from 1, the least linear value that
/// [`linear_to_srgb`] turns into `v` or more.
static SRGB_THRESHOLDS: LazyLock<[f32; 255]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let v = i as u8 + 1;
        // Bisect over the bit patterns of [0, 1], which order like the
        // values they hold.
        let (mut lo, mut hi) = (0u32, 1f32.to_bits());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if linear_to_srgb(f32::from_bits(mid)) >= v {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        f32::from_bits(lo)
    })
});

/// [`linear_to_srgb`] by table lookup instead of `powf`; the same byte
/// for every input.
fn linear_to_srgb_fast(x: f32, thresholds: &[f32; 255]) -> u8 {
    thresholds.partition_point(|&t| t <= x) as u8
}

pub fn linear_to_srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let c = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round() as u8
}

/// Keyframe rows rendered (and packed) at a time, so a whole keyframe of
/// unpacked samples never exists.
const KEYFRAME_BAND_ROWS: usize = 64;

/// A stretch's rendered keyframe: packed samples, row-major,
/// `view.width × view.height`.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub view: Viewport,
    pub data: Vec<Packed>,
}

/// Samples a band of a keyframe at the keyframe's own pixel coordinates,
/// so a banded render is bit-for-bit the plain render of the whole view.
struct BandSampler<'a> {
    inner: F64Sampler,
    keyframe: &'a Viewport,
    row0: usize,
}

impl Sampler for BandSampler<'_> {
    fn sample(&self, _band: &Viewport, x: usize, y: usize) -> Sample {
        self.inner.sample(self.keyframe, x, self.row0 + y)
    }
}

/// Render `s`'s keyframe in bands of [`KEYFRAME_BAND_ROWS`], packing each
/// band as it finishes. `None` if cancelled.
pub fn render_keyframe(
    kind: FractalKind,
    s: &Stretch,
    pool: &rayon::ThreadPool,
    cancel: &CancelToken,
) -> Option<Keyframe> {
    let view = s.view;
    let (w, h, scale) = (view.width, view.height, view.pixel_scale());
    let inner = F64Sampler {
        kind,
        max_iter: s.max_iter,
    };
    let mut data = Vec::with_capacity(w * h);
    for row0 in (0..h).step_by(KEYFRAME_BAND_ROWS) {
        let rows = KEYFRAME_BAND_ROWS.min(h - row0);
        let mid = (row0 as f64 + rows as f64 / 2.0 - h as f64 / 2.0) * scale;
        // `render` takes only the band's size from this; `BandSampler`
        // places every point on the keyframe's own grid.
        let band = Viewport::new(
            Complex64::new(view.center.re, view.center.im - mid),
            rows as f64 * scale,
            w,
            rows,
        );
        let sampler = BandSampler {
            inner,
            keyframe: &view,
            row0,
        };
        let samples = render(&sampler, &band, pool, cancel)?;
        data.extend(samples.into_iter().map(Packed::pack));
    }
    Some(Keyframe { view, data })
}

/// Opaque black, as [`TapColors`] stores colors.
const BLACK: u32 = 0;

/// The most escaped colors [`TapColors`] tabulates: cutoffs up to 16 383
/// iterations, 16 MB. Beyond that taps are colored one by one.
const TAP_TABLE_MAX: usize = 1 << 22;

/// [`tap_color`] for one frame's cutoff and colors, tabulated: an
/// escaped sample's low 30 bits (iterations and smooth fraction) index its
/// color directly, so a tap costs a lookup instead of a palette blend.
/// Colors are packed `r | g << 8 | b << 16`.
struct TapColors<'a> {
    /// Indexed by `packed − (escaped tag)`; one past the cutoff's last
    /// entry is black, which every out-of-range index lands on.
    escaped: Vec<u32>,
    /// Converged colors by root bits and iterations (capped at 20, past
    /// which the shade stops changing).
    converged: Vec<u32>,
    max_iter: u32,
    color: &'a ColorSettings,
    palette: &'a Palette,
}

fn pack_rgb(c: [u8; 4]) -> u32 {
    c[0] as u32 | (c[1] as u32) << 8 | (c[2] as u32) << 16
}

impl<'a> TapColors<'a> {
    fn new(
        max_iter: u32,
        color: &'a ColorSettings,
        palette: &'a Palette,
        pool: &rayon::ThreadPool,
    ) -> TapColors<'a> {
        let entries = (max_iter as usize + 1).saturating_mul(256);
        let mut escaped = Vec::new();
        if entries <= TAP_TABLE_MAX {
            escaped = vec![BLACK; entries + 1];
            pool.install(|| {
                escaped[..entries]
                    .par_chunks_mut(1 << 12)
                    .enumerate()
                    .for_each(|(chunk, cs)| {
                        for (i, c) in cs.iter_mut().enumerate() {
                            let bits = ((chunk << 12) + i) as u32;
                            let p = Packed(TAG_ESCAPED << TAG_SHIFT | bits);
                            *c = pack_rgb(tap_color(p, max_iter, color, palette));
                        }
                    });
            });
        }
        let converged = (0..64u32 * 21)
            .map(|k| {
                let p = Packed(TAG_CONVERGED << TAG_SHIFT | (k / 21) << 24 | (k % 21));
                pack_rgb(tap_color(p, max_iter, color, palette))
            })
            .collect();
        TapColors {
            escaped,
            converged,
            max_iter,
            color,
            palette,
        }
    }

    #[inline(always)]
    fn rgb(&self, p: Packed) -> u32 {
        let v = p.0;
        if self.escaped.is_empty() {
            return pack_rgb(tap_color(p, self.max_iter, self.color, self.palette));
        }
        // Inside wraps around to a huge index, and escapes past the cutoff
        // run off the end: both clamp onto the closing black entry. No
        // branch on what the sample is, which is as random as the fractal.
        let last = self.escaped.len() - 1;
        let i = (v.wrapping_sub(TAG_ESCAPED << TAG_SHIFT) as usize).min(last);
        let c = self.escaped[i];
        if v >> TAG_SHIFT == TAG_CONVERGED {
            let iter = v & CONV_ITER_MAX;
            if iter > self.max_iter {
                return BLACK;
            }
            return self.converged[((v >> 24) & 0x3f) as usize * 21 + iter.min(20) as usize];
        }
        c
    }
}

/// Where frame pixel coordinates land in a keyframe, in keyframe pixels:
/// `u = u0 + ux·x + uy·y`, `v = v0 + vx·x + vy·y`. The same map as
/// [`rotated_point`] followed by the keyframe's pixel grid, set up once
/// per frame instead of per tap.
#[derive(Clone, Copy, Debug)]
struct PixelMap {
    u0: f64,
    ux: f64,
    uy: f64,
    v0: f64,
    vx: f64,
    vy: f64,
}

impl PixelMap {
    fn new(frame: &Viewport, angle: f64, k: &Viewport) -> PixelMap {
        let scale = k.pixel_scale();
        let r = frame.pixel_scale() / scale;
        let (sin, cos) = if angle == 0.0 {
            (0.0, 1.0)
        } else {
            angle.sin_cos()
        };
        // The frame's center in keyframe pixels, taken as offsets from the
        // keyframe's center so deep zooms keep their precision.
        let cu = (frame.center.re - k.center.re) / scale + k.width as f64 / 2.0;
        let cv = (k.center.im - frame.center.im) / scale + k.height as f64 / 2.0;
        let (hw, hh) = (frame.width as f64 / 2.0, frame.height as f64 / 2.0);
        let (ux, uy, vx, vy) = (r * cos, r * sin, -r * sin, r * cos);
        PixelMap {
            u0: cu - ux * hw - uy * hh,
            ux,
            uy,
            v0: cv - vx * hw - vy * hh,
            vx,
            vy,
        }
    }
}

/// Frame pixels per side of the square tiles a frame is resampled in, so
/// a turned frame's taps stay within a small, cached patch of keyframe.
const RESAMPLE_TILE: usize = 32;

/// Make `frame`, turned by `angle` radians, from keyframe `kf`: each
/// output pixel averages, in linear light, `n × n` taps spread over its
/// square, where `n` = frame pixel / keyframe pixel rounded up. Each tap
/// takes the nearest keyframe sample, colored with the frame's
/// `max_iter` cutoff. `out` is `frame.width × frame.height`.
pub fn resample(
    kf: &Keyframe,
    frame: &Viewport,
    angle: f64,
    max_iter: u32,
    color: &ColorSettings,
    pool: &rayon::ThreadPool,
    out: &mut [[u8; 4]],
) {
    assert_eq!(out.len(), frame.width * frame.height);
    debug_assert_eq!(kf.data.len(), kf.view.width * kf.view.height);
    if frame.is_empty() || kf.view.is_empty() {
        return;
    }
    let palette = Palette::get(color.palette);
    let colors = TapColors::new(max_iter, color, &palette, pool);
    let lin: &[f32; 256] = &SRGB_TO_LINEAR;
    let thresholds: &[f32; 255] = &SRGB_THRESHOLDS;
    let k = &kf.view;
    let (kw, max_x, max_y) = (k.width, k.width - 1, k.height - 1);
    let data = &kf.data[..];
    let n = taps_per_axis(frame, k);
    let taps = (n * n) as f32;
    let m = PixelMap::new(frame, angle, k);
    // Keyframe coordinates in 32.32 fixed point: a pixel's corner is
    // mapped in floating point, and its taps are whole-number steps from
    // there, floored by a shift.
    const ONE: f64 = (1u64 << 32) as f64;
    let offsets: Vec<f64> = (0..n).map(|a| (a as f64 + 0.5) / n as f64).collect();
    let steps: Vec<(i64, i64)> = offsets
        .iter()
        .flat_map(|&oy| offsets.iter().map(move |&ox| (ox, oy)))
        .map(|(ox, oy)| {
            let du = m.ux * ox + m.uy * oy;
            let dv = m.vx * ox + m.vy * oy;
            ((du * ONE).round() as i64, (dv * ONE).round() as i64)
        })
        .collect();
    // Far enough outside the keyframe that every tap clamps to its edge,
    // near enough that adding a step can't overflow.
    let (lim_u, lim_v) = (kw as f64 + 4.0 * n as f64, k.height as f64 + 4.0 * n as f64);
    let fixed = |c: f64, lim: f64| (c.clamp(-lim, 2.0 * lim) * ONE) as i64;
    let (max_x, max_y) = (max_x as i64, max_y as i64);
    let pixel = |x: usize, y: usize| -> [u8; 4] {
        let (x, y) = (x as f64, y as f64);
        let bu = fixed(m.u0 + m.ux * x + m.uy * y, lim_u);
        let bv = fixed(m.v0 + m.vx * x + m.vy * y, lim_v);
        let mut sum = [0f32; 3];
        for &(du, dv) in &steps {
            let kx = ((bu + du) >> 32).clamp(0, max_x) as usize;
            let ky = ((bv + dv) >> 32).clamp(0, max_y) as usize;
            let c = colors.rgb(data[ky * kw + kx]);
            sum[0] += lin[(c & 0xff) as usize];
            sum[1] += lin[(c >> 8 & 0xff) as usize];
            sum[2] += lin[(c >> 16) as usize];
        }
        let [r, g, b] = sum.map(|s| linear_to_srgb_fast(s / taps, thresholds));
        [r, g, b, 255]
    };
    let w = frame.width;
    pool.install(|| {
        out.par_chunks_mut(w * RESAMPLE_TILE)
            .enumerate()
            .for_each(|(band, rows)| {
                let y0 = band * RESAMPLE_TILE;
                for x0 in (0..w).step_by(RESAMPLE_TILE) {
                    let x1 = (x0 + RESAMPLE_TILE).min(w);
                    for (dy, row) in rows.chunks_mut(w).enumerate() {
                        for (x, px) in row[x0..x1].iter_mut().enumerate() {
                            *px = pixel(x0 + x, y0 + dy);
                        }
                    }
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::color::{Palette, colorize};

    #[test]
    fn inside_and_converged_round_trip_exactly() {
        let mut v = vec![Sample::Inside];
        for root in 0..4 {
            for iter in [0, 7, 20] {
                v.push(Sample::Converged { root, iter });
            }
        }
        for s in v {
            assert_eq!(pack(s).cut(u32::MAX), s);
        }
    }

    #[test]
    fn escaped_keeps_iter_and_smooth_within_a_256th() {
        for iter in [1u32, 99, 20_000] {
            for f in [0.0f32, 0.3, 0.999] {
                let smooth = iter as f32 + f;
                match pack(Sample::Escaped { iter, smooth }).cut(u32::MAX) {
                    Sample::Escaped { iter: i, smooth: s } => {
                        assert_eq!(i, iter);
                        assert!((s - smooth).abs() <= 1.0 / 256.0, "{smooth} -> {s}");
                    }
                    other => panic!("{other:?}"),
                }
            }
        }
    }

    #[test]
    fn cutoff_turns_deeper_escapes_inside() {
        let p = pack(Sample::Escaped {
            iter: 500,
            smooth: 500.5,
        });
        assert_eq!(p.cut(499), Sample::Inside);
        assert!(matches!(p.cut(500), Sample::Escaped { .. }));
    }

    #[test]
    fn huge_iteration_counts_saturate() {
        let p = pack(Sample::Escaped {
            iter: 5_000_000,
            smooth: 5e6,
        });
        match p.cut(u32::MAX) {
            Sample::Escaped { iter, .. } => assert_eq!(iter, (1 << 22) - 1),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn packed_colors_match_colorize() {
        let samples = [
            Sample::Inside,
            Sample::Escaped {
                iter: 12,
                smooth: 12.25,
            },
            Sample::Escaped {
                iter: 300,
                smooth: 300.5,
            },
            Sample::Converged { root: 1, iter: 3 },
            Sample::Converged { root: 2, iter: 40 },
        ];
        let settings = ColorSettings::default();
        let palette = Palette::get(settings.palette);
        for s in samples {
            let mut out = [[0u8; 4]];
            colorize(&[s], &settings, &mut out);
            assert_eq!(tap_color(pack(s), 1_000_000, &settings, &palette), out[0]);
        }
    }

    /// Every `stride`-th f32 bit pattern from 0 to 1, each threshold and its
    /// neighbors, and values outside [0, 1].
    fn srgb_probe(stride: usize) -> impl Iterator<Item = f32> {
        let grid = (0..=1f32.to_bits()).step_by(stride).map(f32::from_bits);
        let edges = SRGB_THRESHOLDS.iter().flat_map(|t| {
            let b = t.to_bits();
            [b - 1, b, b + 1].map(f32::from_bits)
        });
        let outside = [-1.0, -0.0, 1.0 + f32::EPSILON, 7.0, f32::INFINITY, f32::NAN];
        grid.chain(edges).chain(outside)
    }

    #[test]
    fn fast_linear_to_srgb_matches_powf() {
        let t: &[f32; 255] = &SRGB_THRESHOLDS;
        for x in srgb_probe(61) {
            assert_eq!(linear_to_srgb_fast(x, t), linear_to_srgb(x), "{x:e}");
        }
    }

    /// Exhaustive over every f32 in [0, 1] (about a billion): slow, so run
    /// by hand with `--ignored` after touching either conversion.
    #[test]
    #[ignore]
    fn fast_linear_to_srgb_matches_powf_everywhere() {
        let t: &[f32; 255] = &SRGB_THRESHOLDS;
        assert!(srgb_probe(1).all(|x| linear_to_srgb_fast(x, t) == linear_to_srgb(x)));
    }

    #[test]
    fn tabulated_tap_colors_match_tap_color() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let custom = ColorSettings {
            palette: crate::engine::color::PaletteId::Ice,
            offset: 0.37,
            scale: 0.013,
        };
        for color in [ColorSettings::default(), custom] {
            let palette = Palette::get(color.palette);
            // 20 000 is past the table's limit: colored tap by tap.
            for max_iter in [100, 461, 20_000] {
                let table = TapColors::new(max_iter, &color, &palette, &pool);
                let mut samples = vec![Sample::Inside];
                for iter in (0..max_iter + 50)
                    .step_by(7)
                    .chain([max_iter, max_iter + 1])
                {
                    for f in [0.0, 0.004, 0.5, 0.99] {
                        let smooth = iter as f32 + f;
                        samples.push(Sample::Escaped { iter, smooth });
                    }
                }
                for root in [0, 1, 3, 5, 63] {
                    for iter in [0, 4, 20, 21, 99, max_iter, max_iter + 1] {
                        samples.push(Sample::Converged { root, iter });
                    }
                }
                for s in samples {
                    let p = pack(s);
                    let want = pack_rgb(tap_color(p, max_iter, &color, &palette));
                    assert_eq!(table.rgb(p), want, "{s:?} at cutoff {max_iter}");
                }
            }
        }
    }

    #[test]
    fn srgb_round_trips_every_byte() {
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb(srgb_to_linear(v)), v);
        }
    }
}
