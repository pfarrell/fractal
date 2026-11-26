use fractal_explorer::engine::deep::{HiComplex, bits_for};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn deep() -> (HiComplex, HiComplex) {
    let a = HiComplex::from_c64(c(-0.75, 0.1));
    let b = a.add_c64(c(1e-290, -2e-295), bits_for(1e-300));
    (a, b)
}

#[test]
fn bits_for_values() {
    assert_eq!(bits_for(1e-3), 74);
    assert_eq!(bits_for(1.0), 64);
    assert_eq!(bits_for(1e-300), 1061);
}

#[test]
fn diff_survives_depth() {
    let (a, b) = deep();
    let d = b.diff_c64(&a);
    assert!((d.re - 1e-290).abs() <= 1e-15 * 1e-290, "{d}");
    assert!((d.im + 2e-295).abs() <= 1e-15 * 2e-295, "{d}");
}

#[test]
fn add_raises_precision() {
    let (a, b) = deep();
    assert!(b.bits() >= 1061, "{}", b.bits());
    assert_ne!(b, a);
}

#[test]
fn mul_matches_f64_when_shallow() {
    let (z, w) = (c(0.3, -0.2), c(1.5, 0.25));
    let p = HiComplex::from_c64(z).mul(&HiComplex::from_c64(w)).to_c64();
    let e = z * w;
    assert!((p.re - e.re).abs() <= f64::EPSILON * e.re.abs().max(1.0));
    assert!((p.im - e.im).abs() <= f64::EPSILON * e.im.abs().max(1.0));
}

#[test]
fn decimal_round_trip() {
    let (_, b) = deep();
    let (re, im) = b.to_decimal(40);
    let back = HiComplex::parse(&re, &im).unwrap();
    let d = back.diff_c64(&b);
    assert!(d.re.abs() < 1e-38 && d.im.abs() < 1e-38, "{d}");
    assert_eq!(
        HiComplex::from_c64(c(-0.5, 0.25)).to_decimal(3),
        ("-0.500".to_string(), "+0.250".to_string())
    );
}

#[test]
fn parse_rejects_garbage() {
    assert!(HiComplex::parse("1.2.3", "0").is_err());
    assert!(HiComplex::parse("", "0").is_err());
    assert!(HiComplex::parse("abc", "0").is_err());
}

#[test]
fn parse_accepts_signs_and_exponents() {
    let z = HiComplex::parse("+1.5e-3", "-2").unwrap();
    assert_eq!(z.to_c64(), c(1.5e-3, -2.0));
}

#[test]
fn serde_shallow_is_number_array() {
    let s = serde_json::to_string(&HiComplex::from_c64(c(-0.5, 0.0))).unwrap();
    assert_eq!(s, "[-0.5,0.0]");
    assert_eq!(s, serde_json::to_string(&c(-0.5, 0.0)).unwrap());
}

#[test]
fn serde_deep_is_strings_and_round_trips() {
    let (_, b) = deep();
    let s = serde_json::to_string(&b).unwrap();
    assert!(s.contains("\"re\":\""), "{s}");
    let back: HiComplex = serde_json::from_str(&s).unwrap();
    let d = back.diff_c64(&b);
    assert!(d.re.abs() < 1e-300 && d.im.abs() < 1e-300, "{d}");
}

#[test]
fn serde_reads_old_array() {
    let z: HiComplex = serde_json::from_str("[-0.743643887,0.131825904]").unwrap();
    assert_eq!(z, HiComplex::from_c64(c(-0.743643887, 0.131825904)));
}

#[test]
fn to_c64_is_cheap() {
    let (_, b) = deep();
    let t = std::time::Instant::now();
    let mut acc = 0.0;
    for _ in 0..1_000_000 {
        acc += std::hint::black_box(&b).to_c64().re;
    }
    assert!(acc.is_finite());
    assert!(t.elapsed().as_millis() < 50, "{:?}", t.elapsed());
}
