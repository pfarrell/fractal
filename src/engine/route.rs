//! A camera route through a list of stops.
//!
//! Each step (stop k → stop k+1) is the straight zoom described in
//! [`crate::engine::movie`]: one fixed point stays put while the picture
//! scales about it. Steps get time in proportion to their length in the
//! zoom-and-pan metric (hyperbolic distance in the (center, span)
//! half-plane), so a pure zoom keeps the same speed from step to step and a
//! sideways move gets time for the distance it covers.
//!
//! At each interior stop the two steps' cameras are blended with a
//! smoothstep over a short window, so the direction of motion turns
//! smoothly instead of kinking. The camera passes close to the stop rather
//! than exactly through it. The first and last stops are hit exactly.
//!
//! Rotation: each step turns by a planned amount: the arriving stop's
//! `rotation`, either as total degrees or as °/s times the step's time.
//! The angle follows a monotone cubic (PCHIP, Fritsch–Carlson) through the
//! cumulative angle at each stop's time, so it is exact at every stop and
//! never overshoots. The turning speed is continuous and comes to rest only
//! at the first and last stops, where the turn reverses, and next to a step
//! that doesn't turn.

use crate::engine::deep::{HiComplex, bits_for};
use serde::{Deserialize, Serialize};

/// How a step's `rotation` is measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotationUnit {
    /// Degrees per second, averaged over the step.
    #[default]
    PerSecond,
    /// Degrees turned over the whole step.
    Total,
}

impl RotationUnit {
    pub const ALL: [RotationUnit; 2] = [RotationUnit::PerSecond, RotationUnit::Total];

    /// Unit label, as a slider suffix.
    pub fn suffix(&self) -> &'static str {
        match self {
            RotationUnit::PerSecond => " °/s",
            RotationUnit::Total => " °",
        }
    }

    /// Slider range for a turn in this unit.
    pub fn range(&self) -> std::ops::RangeInclusive<f64> {
        match self {
            RotationUnit::PerSecond => -30.0..=30.0,
            RotationUnit::Total => -720.0..=720.0,
        }
    }
}

/// One point the camera passes through.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stop {
    pub center: HiComplex,
    pub span: f64,
    /// How far the picture turns on the step arriving here, in
    /// `rotation_unit`; ignored on the first stop.
    pub rotation: f64,
    #[serde(default)]
    pub rotation_unit: RotationUnit,
}

/// Default half-width of the blend window around an interior stop, in
/// seconds of movie.
pub const JOINT_HALF_WIDTH_S: f64 = 0.5;

/// Distance between two views in the (center, span) half-plane. For two
/// views with the same center it is `|ln(b.span / a.span)|`.
pub fn step_length(a: &Stop, b: &Stop) -> f64 {
    // acosh(1 + h²/2) with h² = (|Δc|² + Δs²)/(a·b), written as
    // 2·asinh(h/2) so nothing is squared: a·b underflows below spans of
    // about 1e-154, and h² overflows for deep stops far apart.
    let g = a.span.sqrt() * b.span.sqrt();
    let dc = b.center.diff_c64(&a.center).norm() / g;
    let ds = (b.span - a.span) / g;
    2.0 * (dc.hypot(ds) / 2.0).asinh()
}

/// `next`'s center lies outside the frame of `prev` (span is the frame's
/// height) at this aspect ratio.
pub fn outside_previous(prev: &Stop, next: &Stop, width: usize, height: usize) -> bool {
    let d = next.center.diff_c64(&prev.center);
    let half_h = prev.span / 2.0;
    let half_w = half_h * width as f64 / height.max(1) as f64;
    d.re.abs() > half_w || d.im.abs() > half_h
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub struct Route {
    stops: Vec<Stop>,
    /// Precision of the route's centre arithmetic.
    bits: usize,
    /// Movie time of each stop: 0 … seconds.
    times: Vec<f64>,
    /// Blend half-width at each stop (0 at the first and last).
    half: Vec<f64>,
    /// Planned angle at each stop's time, radians (0 at the first).
    knot_angle: Vec<f64>,
    /// Turning speed at each stop's time, radians per second.
    knot_rate: Vec<f64>,
}

/// Each step's planned turn in radians, from the arriving stop.
fn step_turns(stops: &[Stop], times: &[f64]) -> Vec<f64> {
    (0..stops.len() - 1)
        .map(|k| {
            let r = stops[k + 1].rotation.to_radians();
            match stops[k + 1].rotation_unit {
                RotationUnit::Total => r,
                RotationUnit::PerSecond => r * (times[k + 1] - times[k]),
            }
        })
        .collect()
}

/// PCHIP knot speeds for steps of duration `h` turning `turns`: 0 at the
/// ends and where the turn changes sign or stops, else the weighted
/// harmonic mean of the neighbouring steps' average speeds.
fn knot_rates(h: &[f64], turns: &[f64]) -> Vec<f64> {
    let d: Vec<f64> = h
        .iter()
        .zip(turns)
        .map(|(&h, &a)| if h > 0.0 { a / h } else { 0.0 })
        .collect();
    let mut m = vec![0.0; h.len() + 1];
    for k in 1..h.len() {
        let (d0, d1) = (d[k - 1], d[k]);
        if d0 * d1 > 0.0 {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            m[k] = (w1 + w2) / (w1 / d0 + w2 / d1);
        }
    }
    m
}

impl Route {
    /// `stops` must hold at least two stops with no zero-length step
    /// (the movie spec validates this).
    pub fn new(stops: &[Stop], seconds: f64, max_half_width: f64) -> Route {
        assert!(stops.len() >= 2, "a route needs at least two stops");
        let lengths: Vec<f64> = stops
            .windows(2)
            .map(|w| step_length(&w[0], &w[1]))
            .collect();
        let total: f64 = lengths.iter().sum();
        let mut times = Vec::with_capacity(stops.len());
        let mut acc = 0.0;
        times.push(0.0);
        for l in &lengths {
            acc += l;
            times.push(acc / total * seconds);
        }
        *times.last_mut().unwrap() = seconds;
        let n = stops.len();
        let mut half = vec![0.0; n];
        for k in 1..n - 1 {
            let before = times[k] - times[k - 1];
            let after = times[k + 1] - times[k];
            half[k] = max_half_width.min(before / 3.0).min(after / 3.0).max(0.0);
        }
        let turns = step_turns(stops, &times);
        let mut knot_angle = Vec::with_capacity(n);
        knot_angle.push(0.0);
        for a in &turns {
            knot_angle.push(knot_angle.last().unwrap() + a);
        }
        let h: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        let knot_rate = knot_rates(&h, &turns);
        // Enough for a frame up to 8192 pixels tall at the deepest stop.
        let min_span = stops.iter().map(|s| s.span).fold(f64::MAX, f64::min);
        Route {
            stops: stops.to_vec(),
            bits: bits_for(min_span / 8192.0),
            times,
            half,
            knot_angle,
            knot_rate,
        }
    }

    /// Movie time at which the camera reaches (or passes) each stop.
    pub fn step_times(&self) -> &[f64] {
        &self.times
    }

    fn steps(&self) -> usize {
        self.stops.len() - 1
    }

    /// Step `k`'s straight zoom at time `t`, extended past its own ends.
    fn step_view(&self, k: usize, t: f64) -> (HiComplex, f64) {
        let (a, b) = (&self.stops[k], &self.stops[k + 1]);
        let u = (t - self.times[k]) / (self.times[k + 1] - self.times[k]);
        let r = b.span / a.span;
        let span = a.span * r.powf(u);
        let (ca, cb) = (a.center.with_bits(self.bits), &b.center);
        let center = if (1.0 - r).abs() < 1e-9 {
            ca.lerp(cb, u)
        } else {
            // (b − a·r)/(1 − r), written so a tiny r isn't lost: 1/(1 − r)
            // would round to exactly 1 in f64 past ×10¹⁶.
            let fixed = cb.add(&cb.sub(&ca).mul_f64(r / (1.0 - r)));
            fixed.add(&ca.sub(&fixed).mul_f64(span / a.span))
        };
        (center, span)
    }

    /// The blend of steps `j − 1` and `j` around interior stop `j`.
    fn joint_view(&self, j: usize, t: f64) -> (HiComplex, f64) {
        let h = self.half[j];
        let w = smoothstep(self.times[j] - h, self.times[j] + h, t);
        let (ca, sa) = self.step_view(j - 1, t);
        let (cb, sb) = self.step_view(j, t);
        let span = (sa.ln() + (sb.ln() - sa.ln()) * w).exp();
        // A step extended past its end keeps zooming; next to a step at
        // one depth that would dip past the stop and back. Stay within
        // the depths of the stops around the joint.
        let near = &self.stops[j - 1..=j + 1];
        let lo = near.iter().map(|s| s.span).fold(f64::MAX, f64::min);
        let hi = near.iter().map(|s| s.span).fold(f64::MIN, f64::max);
        (ca.lerp(&cb, w), span.clamp(lo, hi))
    }

    /// The step whose time range holds `t` (the last step holds the end).
    fn step_at(&self, t: f64) -> usize {
        let n = self.steps();
        (0..n).find(|&k| t < self.times[k + 1]).unwrap_or(n - 1)
    }

    /// (center, span) at movie time `t`.
    pub fn view_at(&self, t: f64) -> (HiComplex, f64) {
        let last = self.stops.len() - 1;
        if t <= 0.0 {
            return (self.stops[0].center.clone(), self.stops[0].span);
        }
        if t >= self.times[last] {
            return (self.stops[last].center.clone(), self.stops[last].span);
        }
        let k = self.step_at(t);
        if k >= 1 && t < self.times[k] + self.half[k] {
            return self.joint_view(k, t);
        }
        if k + 1 < last && t > self.times[k + 1] - self.half[k + 1] {
            return self.joint_view(k + 1, t);
        }
        self.step_view(k, t)
    }

    /// Radians turned by time `t` (counterclockwise positive).
    pub fn angle_at(&self, t: f64) -> f64 {
        let last = self.times.len() - 1;
        let t = t.clamp(0.0, self.times[last]);
        let k = self.step_at(t);
        let (a0, a1) = (self.knot_angle[k], self.knot_angle[k + 1]);
        let (m0, m1) = (self.knot_rate[k], self.knot_rate[k + 1]);
        if a0 == 0.0 && a1 == 0.0 && m0 == 0.0 && m1 == 0.0 {
            return 0.0;
        }
        let h = self.times[k + 1] - self.times[k];
        if h <= 0.0 {
            return a1;
        }
        let s = (t - self.times[k]) / h;
        let (s2, s3) = (s * s, s * s * s);
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        h00 * a0 + h10 * h * m0 + h01 * a1 + h11 * h * m1
    }
}
