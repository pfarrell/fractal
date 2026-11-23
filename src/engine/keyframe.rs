//! Keyframe samples packed into 4 bytes so a movie frame can be colored
//! with its own iteration cutoff.
use crate::engine::color::{ColorSettings, Palette, sample_color};
use crate::engine::escape::Sample;

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
