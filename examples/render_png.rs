//! Headless render to PNG, for checking the engine's output by eye.
//!
//! cargo run --release --example render_png -- <kind> <re> <im> <span> <out.png> [max_iter]
//! kind: mandelbrot | julia | ship | tricorn | multibrot3 | newton3 | newton4 | newtoncycle
//! or:   cargo run --release --example render_png -- ifs <preset 0-3> <out.png>

use fractal_explorer::engine::color::{ColorSettings, colorize, colorize_density};
use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::ifs::{IfsSystem, chaos_game};
use fractal_explorer::engine::newton::NewtonPoly;
use fractal_explorer::engine::render::{CancelToken, F64Sampler, render};
use fractal_explorer::engine::viewport::Viewport;
use num_complex::Complex64;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("ifs") {
        let sys = &IfsSystem::presets()[a[2].parse::<usize>().expect("preset")];
        let view = sys.fit_view(800, 600);
        let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
        let t = std::time::Instant::now();
        let d = chaos_game(sys, &view, 20_000_000, 1, &pool, &CancelToken::never()).unwrap();
        eprintln!("{}: 20M points in {:?}", sys.name, t.elapsed());
        let mut px = vec![[0u8; 4]; d.counts.len()];
        colorize_density(&d.counts, &ColorSettings::default(), &mut px);
        image::save_buffer(
            &a[3],
            px.as_flattened(),
            800,
            600,
            image::ExtendedColorType::Rgba8,
        )
        .unwrap();
        return;
    }
    if a.len() < 6 {
        eprintln!("usage: render_png <kind> <re> <im> <span> <out.png> [max_iter]");
        std::process::exit(2);
    }
    let kind = match a[1].as_str() {
        "mandelbrot" => FractalKind::Mandelbrot,
        "julia" => FractalKind::Julia {
            c: Complex64::new(-0.8, 0.156),
        },
        "ship" => FractalKind::BurningShip,
        "tricorn" => FractalKind::Tricorn,
        "multibrot3" => FractalKind::Multibrot { power: 3 },
        "newton3" => FractalKind::Newton {
            poly: NewtonPoly::CubeRootsOfUnity,
        },
        "newton4" => FractalKind::Newton {
            poly: NewtonPoly::FourthRootsOfUnity,
        },
        "newtoncycle" => FractalKind::Newton {
            poly: NewtonPoly::CycleCubic,
        },
        k => panic!("unknown kind {k}"),
    };
    let num = |i: usize| a[i].parse::<f64>().expect("number");
    let view = Viewport::new(Complex64::new(num(2), num(3)), num(4), 800, 600);
    let max_iter = a.get(6).map_or(500, |s| s.parse().expect("max_iter"));
    let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    let t = std::time::Instant::now();
    let samples = render(
        &F64Sampler { kind, max_iter },
        &view,
        &pool,
        &CancelToken::never(),
    )
    .unwrap();
    eprintln!("rendered in {:?}", t.elapsed());
    let mut px = vec![[0u8; 4]; samples.len()];
    colorize(&samples, &ColorSettings::default(), &mut px);
    image::save_buffer(
        &a[5],
        px.as_flattened(),
        800,
        600,
        image::ExtendedColorType::Rgba8,
    )
    .unwrap();
}
