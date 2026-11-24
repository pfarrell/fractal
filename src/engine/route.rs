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
//! Rotation: step k turns at the arriving stop's `rotation` (°/s); the
//! rate blends across the same windows, and the angle is its integral.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// One point the camera passes through.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stop {
    pub center: Complex64,
    pub span: f64,
    /// °/s the picture turns on the step arriving here; ignored on the
    /// first stop.
    pub rotation: f64,
}

/// Default half-width of the blend window around an interior stop, in
/// seconds of movie.
pub const JOINT_HALF_WIDTH_S: f64 = 0.5;

/// Distance between two views in the (center, span) half-plane. For two
/// views with the same center it is `|ln(b.span / a.span)|`.
pub fn step_length(a: &Stop, b: &Stop) -> f64 {
    let dc = (b.center - a.center).norm_sqr();
    let ds = b.span - a.span;
    (1.0 + (dc + ds * ds) / (2.0 * a.span * b.span)).acosh()
}

/// `next`'s center lies outside the frame of `prev` (span is the frame's
/// height) at this aspect ratio.
pub fn outside_previous(prev: &Stop, next: &Stop, width: usize, height: usize) -> bool {
    let d = next.center - prev.center;
    let half_h = prev.span / 2.0;
    let half_w = half_h * width as f64 / height.max(1) as f64;
    d.re.abs() > half_w || d.im.abs() > half_h
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// ∫ from −∞ to `t` of a unit step at `at`, smoothed by a smoothstep over
/// `[at − h, at + h]` (a plain step when `h` is 0).
fn smoothed_step_integral(at: f64, h: f64, t: f64) -> f64 {
    if t <= at - h {
        0.0
    } else if t >= at + h {
        t - at
    } else {
        let x = (t - (at - h)) / (2.0 * h);
        2.0 * h * (x * x * x - x * x * x * x / 2.0)
    }
}

pub struct Route {
    stops: Vec<Stop>,
    /// Movie time of each stop: 0 … seconds.
    times: Vec<f64>,
    /// Blend half-width at each stop (0 at the first and last).
    half: Vec<f64>,
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
        Route {
            stops: stops.to_vec(),
            times,
            half,
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
    fn step_view(&self, k: usize, t: f64) -> (Complex64, f64) {
        let (a, b) = (&self.stops[k], &self.stops[k + 1]);
        let u = (t - self.times[k]) / (self.times[k + 1] - self.times[k]);
        let r = b.span / a.span;
        let span = a.span * r.powf(u);
        let center = if (1.0 - r).abs() < 1e-9 {
            a.center + (b.center - a.center) * u
        } else {
            let fixed = (b.center - a.center * r) / (1.0 - r);
            fixed + (a.center - fixed) * (span / a.span)
        };
        (center, span)
    }

    /// The blend of steps `j − 1` and `j` around interior stop `j`.
    fn joint_view(&self, j: usize, t: f64) -> (Complex64, f64) {
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
        (ca + (cb - ca) * w, span.clamp(lo, hi))
    }

    /// The step whose time range holds `t` (the last step holds the end).
    fn step_at(&self, t: f64) -> usize {
        let n = self.steps();
        (0..n).find(|&k| t < self.times[k + 1]).unwrap_or(n - 1)
    }

    /// (center, span) at movie time `t`.
    pub fn view_at(&self, t: f64) -> (Complex64, f64) {
        let last = self.stops.len() - 1;
        if t <= 0.0 {
            return (self.stops[0].center, self.stops[0].span);
        }
        if t >= self.times[last] {
            return (self.stops[last].center, self.stops[last].span);
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
        let rate = |k: usize| self.stops[k + 1].rotation.to_radians();
        let mut angle = rate(0) * t;
        for j in 1..self.steps() {
            angle +=
                (rate(j) - rate(j - 1)) * smoothed_step_integral(self.times[j], self.half[j], t);
        }
        angle
    }
}
