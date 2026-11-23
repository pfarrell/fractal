//! Turning samples into pixels.
//!
//! Escape-time pixels use *smooth* iteration counts mapped onto a cyclic
//! gradient, so colors flow continuously instead of forming bands.

use crate::engine::escape::Sample;
use serde::{Deserialize, Serialize};

/// Named gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaletteId {
    Classic,
    Fire,
    Ice,
    Grayscale,
}

impl PaletteId {
    pub const ALL: [PaletteId; 4] = [
        PaletteId::Classic,
        PaletteId::Fire,
        PaletteId::Ice,
        PaletteId::Grayscale,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PaletteId::Classic => "Classic",
            PaletteId::Fire => "Fire",
            PaletteId::Ice => "Ice",
            PaletteId::Grayscale => "Grayscale",
        }
    }
}

/// A cyclic gradient: evenly spaced color stops, the last blending back
/// into the first.
#[derive(Clone, Debug)]
pub struct Palette {
    stops: Vec<[u8; 3]>,
}

impl Palette {
    pub fn get(id: PaletteId) -> Self {
        let stops = match id {
            PaletteId::Classic => vec![
                [0, 7, 100],
                [32, 107, 203],
                [237, 255, 255],
                [255, 170, 0],
                [0, 2, 0],
            ],
            PaletteId::Fire => vec![
                [20, 0, 0],
                [180, 30, 0],
                [255, 160, 0],
                [255, 255, 180],
                [120, 20, 0],
            ],
            PaletteId::Ice => vec![
                [0, 10, 30],
                [0, 80, 160],
                [120, 200, 255],
                [240, 255, 255],
                [0, 60, 120],
            ],
            PaletteId::Grayscale => vec![[30, 30, 30], [230, 230, 230]],
        };
        Self { stops }
    }

    /// Color at position `t`; wraps around every 1.0.
    pub fn at(&self, t: f32) -> [u8; 3] {
        let n = self.stops.len();
        let pos = t.rem_euclid(1.0) * n as f32;
        let i = (pos as usize).min(n - 1);
        let frac = pos - i as f32;
        let (a, b) = (self.stops[i], self.stops[(i + 1) % n]);
        let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * frac).round() as u8;
        [mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2])]
    }
}

/// User-adjustable coloring parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorSettings {
    pub palette: PaletteId,
    /// Shifts where in the gradient colors start (0..1).
    pub offset: f32,
    /// Gradient cycles per iteration: smaller = slower color change.
    pub scale: f32,
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            palette: PaletteId::Classic,
            offset: 0.0,
            scale: 1.0 / 32.0,
        }
    }
}

/// Fixed hues for Newton roots (index = root number).
const ROOT_COLORS: [[u8; 3]; 4] = [[230, 60, 70], [60, 190, 90], [60, 110, 230], [235, 200, 50]];

fn rgba(c: [u8; 3]) -> [u8; 4] {
    [c[0], c[1], c[2], 255]
}

/// Color escape-time samples. `out` must be the same length as `samples`.
pub fn colorize(samples: &[Sample], s: &ColorSettings, out: &mut [[u8; 4]]) {
    debug_assert_eq!(samples.len(), out.len());
    let palette = Palette::get(s.palette);
    for (px, sample) in out.iter_mut().zip(samples) {
        *px = sample_color(*sample, s, &palette);
    }
}

/// Color one escape-time sample.
pub fn sample_color(sample: Sample, s: &ColorSettings, palette: &Palette) -> [u8; 4] {
    match sample {
        Sample::Inside => [0, 0, 0, 255],
        Sample::Escaped { smooth, .. } => rgba(palette.at(smooth * s.scale + s.offset)),
        Sample::Converged { root, iter } => {
            // Darker = took longer to converge; shows the basin structure.
            let shade = 0.85f32.powi(iter.min(20) as i32).max(0.25);
            let c = ROOT_COLORS[root as usize % ROOT_COLORS.len()];
            rgba(c.map(|v| (v as f32 * shade) as u8))
        }
    }
}

/// The densest IFS pixels map to this gradient position. Gradients are
/// cyclic, so running all the way to 1.0 would wrap the brightest pixels
/// back to the dark starting color.
pub const DENSITY_TOP: f32 = 0.6;

/// Color an IFS hit-count buffer on a log scale (densities span many
/// orders of magnitude). `out` must be the same length as `counts`.
pub fn colorize_density(counts: &[u32], s: &ColorSettings, out: &mut [[u8; 4]]) {
    debug_assert_eq!(counts.len(), out.len());
    let palette = Palette::get(s.palette);
    let max = counts.iter().copied().max().unwrap_or(0);
    let denom = (1.0 + max as f32).ln();
    for (px, &n) in out.iter_mut().zip(counts) {
        *px = if n == 0 {
            [0, 0, 0, 255]
        } else {
            rgba(palette.at((1.0 + n as f32).ln() / denom * DENSITY_TOP))
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::escape::Sample;

    const BLACK: [u8; 4] = [0, 0, 0, 255];

    #[test]
    fn inside_is_black() {
        let mut out = [[9u8; 4]; 1];
        colorize(&[Sample::Inside], &ColorSettings::default(), &mut out);
        assert_eq!(out[0], BLACK);
    }

    #[test]
    fn palette_wraps() {
        for id in PaletteId::ALL {
            let p = Palette::get(id);
            assert_eq!(p.at(0.25), p.at(1.25), "{id:?}");
        }
    }

    #[test]
    fn palette_continuous() {
        for id in PaletteId::ALL {
            let p = Palette::get(id);
            for i in 0..1000 {
                let a = p.at(i as f32 / 1000.0);
                let b = p.at((i + 1) as f32 / 1000.0);
                for ch in 0..3 {
                    assert!((a[ch] as i32 - b[ch] as i32).abs() <= 2, "{id:?} at {i}");
                }
            }
        }
    }

    #[test]
    fn newton_roots_distinct() {
        let s = ColorSettings::default();
        let mut out = [[0u8; 4]; 3];
        let samples = [0u8, 1, 2].map(|root| Sample::Converged { root, iter: 0 });
        colorize(&samples, &s, &mut out);
        assert_ne!(out[0], out[1]);
        assert_ne!(out[1], out[2]);
        assert_ne!(out[0], out[2]);
    }

    #[test]
    fn density_zero_black_max_bright() {
        let s = ColorSettings::default();
        let mut out = [[0u8; 4]; 3];
        colorize_density(&[0, 5, 100], &s, &mut out);
        assert_eq!(out[0], BLACK);
        let top = Palette::get(s.palette).at(DENSITY_TOP);
        assert_eq!(out[2], [top[0], top[1], top[2], 255]);
    }

    #[test]
    fn colorize_equal_lengths_fill_everything() {
        let samples = [Sample::Escaped {
            iter: 3,
            smooth: 3.2,
        }; 4];
        let mut out = [[0u8; 4]; 4];
        colorize(&samples, &ColorSettings::default(), &mut out);
        assert!(out.iter().all(|px| px[3] == 255));
    }
}
