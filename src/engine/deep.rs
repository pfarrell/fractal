//! High-precision complex numbers for view centres past the f64 limit.
//!
//! Only centres and reference orbits use this type; per-pixel work stays
//! in f64 offsets from a centre (see `engine::perturb`).

use dashu_float::round::mode::HalfEven;
use dashu_float::{DBig, FBig};
use dashu_int::ops::UnsignedAbs;
use dashu_int::{IBig, UBig};
use num_complex::Complex64;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

type F = FBig<HalfEven, 2>;

/// The fewest bits a value is ever held at.
const MIN_BITS: usize = 64;

/// Bits needed to place pixels `pixel_scale` apart near |c| ≈ 1, with 64
/// spare: `max(64, ceil(log2(1 / pixel_scale)) + 64)`.
pub fn bits_for(pixel_scale: f64) -> usize {
    let need = (1.0 / pixel_scale).log2().ceil();
    if need.is_finite() && need > 0.0 {
        (need as usize + 64).max(MIN_BITS)
    } else {
        MIN_BITS
    }
}

/// A complex number at arbitrary binary precision.
#[derive(Clone, Debug)]
pub struct HiComplex {
    re: F,
    im: F,
}

impl PartialEq for HiComplex {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

/// A decimal string that isn't a number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseHiError(pub String);

impl fmt::Display for ParseHiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseHiError {}

fn from_f64(x: f64) -> F {
    // Finite by construction everywhere this is called; NaN/inf map to 0
    // rather than panic.
    F::try_from(x)
        .unwrap_or(F::ZERO)
        .with_precision(MIN_BITS)
        .value()
}

fn to_f64(x: &F) -> f64 {
    x.to_f64().value()
}

fn at_bits(x: &F, bits: usize) -> F {
    x.clone().with_precision(bits.max(MIN_BITS)).value()
}

impl From<Complex64> for HiComplex {
    fn from(z: Complex64) -> Self {
        HiComplex::from_c64(z)
    }
}

impl HiComplex {
    /// `z` exactly (an f64 fits in 53 bits).
    pub fn from_c64(z: Complex64) -> Self {
        HiComplex {
            re: from_f64(z.re),
            im: from_f64(z.im),
        }
    }

    /// Nearest f64 value.
    pub fn to_c64(&self) -> Complex64 {
        Complex64::new(to_f64(&self.re), to_f64(&self.im))
    }

    /// `self − other`, computed at full precision then rounded to f64.
    pub fn diff_c64(&self, other: &HiComplex) -> Complex64 {
        Complex64::new(
            to_f64(&(&self.re - &other.re)),
            to_f64(&(&self.im - &other.im)),
        )
    }

    /// `self + d`, held at no fewer than `bits` bits.
    pub fn add_c64(&self, d: Complex64, bits: usize) -> HiComplex {
        let p = self.bits().max(bits);
        HiComplex {
            re: at_bits(&self.re, p) + from_f64(d.re),
            im: at_bits(&self.im, p) + from_f64(d.im),
        }
    }

    pub fn add(&self, o: &HiComplex) -> HiComplex {
        HiComplex {
            re: &self.re + &o.re,
            im: &self.im + &o.im,
        }
    }

    pub fn sub(&self, o: &HiComplex) -> HiComplex {
        HiComplex {
            re: &self.re - &o.re,
            im: &self.im - &o.im,
        }
    }

    /// Complex product.
    pub fn mul(&self, o: &HiComplex) -> HiComplex {
        HiComplex {
            re: &self.re * &o.re - &self.im * &o.im,
            im: &self.re * &o.im + &self.im * &o.re,
        }
    }

    pub fn mul_f64(&self, k: f64) -> HiComplex {
        let k = from_f64(k);
        HiComplex {
            re: &self.re * &k,
            im: &self.im * &k,
        }
    }

    /// `self + (o − self)·w`.
    pub fn lerp(&self, o: &HiComplex, w: f64) -> HiComplex {
        self.add(&o.sub(self).mul_f64(w))
    }

    /// The same value held at `bits` bits (rounded if fewer).
    pub fn with_bits(&self, bits: usize) -> HiComplex {
        HiComplex {
            re: at_bits(&self.re, bits),
            im: at_bits(&self.im, bits),
        }
    }

    /// Precision in bits (the larger of the two parts).
    pub fn bits(&self) -> usize {
        self.re.precision().max(self.im.precision())
    }

    pub fn norm_sqr_f64(&self) -> f64 {
        self.to_c64().norm_sqr()
    }

    /// Fixed-point decimals with a sign and exactly `frac_digits` digits
    /// after the point, e.g. `("-0.500", "+0.250")`.
    pub fn to_decimal(&self, frac_digits: usize) -> (String, String) {
        (fixed(&self.re, frac_digits), fixed(&self.im, frac_digits))
    }

    /// Parse two decimals (optional sign, digits, optional fraction,
    /// optional exponent). Precision follows the digits given.
    pub fn parse(re: &str, im: &str) -> Result<HiComplex, ParseHiError> {
        Ok(HiComplex {
            re: parse_decimal(re)?,
            im: parse_decimal(im)?,
        })
    }

    /// True when the value is exactly an f64 pair.
    fn is_f64_exact(&self) -> bool {
        let z = self.to_c64();
        from_f64(z.re) == self.re && from_f64(z.im) == self.im
    }
}

/// `x` rounded to `frac` decimals, as `[+-]int.frac`.
fn fixed(x: &F, frac: usize) -> String {
    let scale = IBig::from(UBig::from(10u8).pow(frac));
    let p = x.precision().max(MIN_BITS) + frac * 4 + 64;
    let scaled = at_bits(x, p) * F::from(scale).with_precision(p).value();
    let n: IBig = scaled.round().to_int().value();
    let neg = n < IBig::ZERO || (n == IBig::ZERO && x < &F::ZERO);
    let digits = n.clone().unsigned_abs().to_string();
    let digits = if digits.len() <= frac {
        format!("{}{}", "0".repeat(frac + 1 - digits.len()), digits)
    } else {
        digits
    };
    let (int, rest) = digits.split_at(digits.len() - frac);
    let sign = if neg { '-' } else { '+' };
    if frac == 0 {
        format!("{sign}{int}")
    } else {
        format!("{sign}{int}.{rest}")
    }
}

/// Decimal text → binary float. Grammar: `[+-]? digits ('.' digits)?
/// ([eE] [+-]? digits)?`, with at least one mantissa digit.
fn parse_decimal(s: &str) -> Result<F, ParseHiError> {
    let bad = || ParseHiError(format!("not a number: {s:?}"));
    let t = s.trim();
    let (neg, t) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (mant, exp) = match t.find(['e', 'E']) {
        Some(i) => (&t[..i], Some(&t[i + 1..])),
        None => (t, None),
    };
    let (int, frac) = match mant.split_once('.') {
        Some((a, b)) => (a, b),
        None => (mant, ""),
    };
    let all_digits = |x: &str| x.bytes().all(|b| b.is_ascii_digit());
    if int.len() + frac.len() == 0 || !all_digits(int) || !all_digits(frac) {
        return Err(bad());
    }
    let exp: isize = match exp {
        Some(e) => {
            let body = e.strip_prefix(['+', '-']).unwrap_or(e);
            if body.is_empty() || !all_digits(body) {
                return Err(bad());
            }
            e.parse().map_err(|_| bad())?
        }
        None => 0,
    };
    let digits = format!("{int}{frac}");
    let sig = digits.trim_start_matches('0').len().max(1);
    let mut m: IBig = digits.parse::<UBig>().map_err(|_| bad())?.into();
    if neg {
        m = -m;
    }
    let bits = (MIN_BITS).max((sig as f64 * 3.33).ceil() as usize + 8);
    let dec = DBig::from_parts(m, exp - frac.len() as isize);
    Ok(dec
        .with_rounding::<HalfEven>()
        .with_base_and_precision::<2>(bits)
        .value())
}

/// `x` as `<significand>e<exponent>` with `digits` significant digits.
fn scientific(x: &F, digits: usize) -> String {
    let d: DBig = x
        .clone()
        .with_rounding()
        .with_base_and_precision::<10>(digits)
        .value();
    let r = d.repr();
    format!("{}e{}", r.significand(), r.exponent())
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Repr {
    Num(f64, f64),
    Str { re: String, im: String },
}

impl Serialize for HiComplex {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.is_f64_exact() {
            let z = self.to_c64();
            return Repr::Num(z.re, z.im).serialize(s);
        }
        let digits = (self.bits() as f64 * std::f64::consts::LOG10_2).ceil() as usize + 2;
        Repr::Str {
            re: scientific(&self.re, digits),
            im: scientific(&self.im, digits),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for HiComplex {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match Repr::deserialize(d)? {
            Repr::Num(re, im) => Ok(HiComplex::from_c64(Complex64::new(re, im))),
            Repr::Str { re, im } => HiComplex::parse(&re, &im).map_err(serde::de::Error::custom),
        }
    }
}
