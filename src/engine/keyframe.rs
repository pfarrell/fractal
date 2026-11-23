//! Keyframe movies: frames grouped into stretches, each resampled from
//! one large keyframe whose samples are packed into 4 bytes so a movie
//! frame can be colored with its own iteration cutoff.
use crate::engine::color::{ColorSettings, Palette, sample_color};
use crate::engine::escape::Sample;
use crate::engine::movie::MovieSpec;
use crate::engine::viewport::{Viewport, min_pixel_scale};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::ops::Range;

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

/// Samples rendered for `spec`, relative to Exact: keyframe samples plus
/// the pixels of frames rendered Exact, over every frame's pixels.
pub fn estimated_cost(spec: &MovieSpec) -> f64 {
    let Some(q) = spec.quality.q() else {
        return 1.0;
    };
    let frame = (spec.width * spec.height) as f64;
    let samples: f64 = plan(spec, q, KEYFRAME_MAX_PIXELS)
        .iter()
        .map(|b| match b {
            Block::Keyframe(s) => (s.view.width * s.view.height) as f64,
            Block::Exact(r) => r.len() as f64 * frame,
        })
        .sum();
    samples / (frame * spec.frame_count() as f64)
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

pub fn srgb_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
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

    #[test]
    fn srgb_round_trips_every_byte() {
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb(srgb_to_linear(v)), v);
        }
    }
}
