//! Fractal Explorer: an interactive, multithreaded fractal explorer.
//!
//! `engine` holds all the math, coloring and parallel rendering and has no
//! GUI dependencies. `ui` is the eframe/egui front end.

pub mod engine;
pub mod ui;
