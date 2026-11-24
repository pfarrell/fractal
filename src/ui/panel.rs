//! The left-hand control panel for escape-time fractals.

use crate::engine::color::{ColorSettings, PaletteId};
use crate::engine::escape::FractalKind;
use crate::engine::escape::Sample;
use crate::engine::newton::NewtonPoly;
use crate::engine::viewport::MaxIter;
use crate::ui::app::{ExploreState, OrbitState};
use crate::ui::notes::{NOTE_ORBIT, note, note_for_kind};
use eframe::egui;

/// What the user changed this frame.
#[derive(Default)]
pub struct PanelChanges {
    /// Needs a re-render (kind, parameters, iterations, view reset).
    pub render: bool,
    /// Needs only a recolor of the existing samples.
    pub color: bool,
    /// New thread count requested.
    pub threads: Option<usize>,
    /// "Movie…" clicked.
    pub open_movie: bool,
}

/// Controls for the escape-time explorer. `threads` is the current main
/// pool size; `max_threads` the number of cores.
pub fn explore_panel(
    ui: &mut egui::Ui,
    state: &mut ExploreState,
    threads: usize,
    max_threads: usize,
) -> PanelChanges {
    let mut ch = PanelChanges::default();

    ui.heading("Fractal");
    let current = state.kind;
    egui::ComboBox::from_id_salt("kind")
        .selected_text(current.label())
        .show_ui(ui, |ui| {
            for k in FractalKind::gallery() {
                if ui
                    .selectable_label(k.label() == current.label(), k.label())
                    .clicked()
                    && k.label() != current.label()
                {
                    state.kind = k;
                    state.view = k.default_view(state.view.width, state.view.height);
                    ch.render = true;
                }
            }
        });

    match &mut state.kind {
        FractalKind::Julia { c } => {
            ui.horizontal(|ui| {
                ui.label("c =");
                let speed = 0.001;
                ch.render |= ui
                    .add(egui::DragValue::new(&mut c.re).speed(speed).max_decimals(6))
                    .changed();
                ui.label("+");
                ch.render |= ui
                    .add(egui::DragValue::new(&mut c.im).speed(speed).max_decimals(6))
                    .changed();
                ui.label("i");
            });
        }
        FractalKind::Multibrot { power } => {
            ch.render |=
                crate::ui::slider(ui, egui::Slider::new(power, 2..=8).text("power d")).changed();
        }
        FractalKind::Newton { poly } => {
            egui::ComboBox::from_id_salt("poly")
                .selected_text(poly.label())
                .show_ui(ui, |ui| {
                    for p in NewtonPoly::ALL {
                        ch.render |= ui.selectable_value(poly, p, p.label()).changed();
                    }
                });
        }
        _ => {}
    }

    note(ui, "kind_note", note_for_kind(&state.kind));

    ui.separator();
    ui.heading("Iterations");
    let mut auto = state.max_iter == MaxIter::Auto;
    let effective = state.max_iter.resolve(state.view.magnification());
    if ui
        .checkbox(&mut auto, "Auto (deeper zoom → more)")
        .changed()
    {
        state.max_iter = if auto {
            MaxIter::Auto
        } else {
            MaxIter::Fixed(effective)
        };
        ch.render = true;
    }
    match &mut state.max_iter {
        MaxIter::Fixed(n) => {
            ch.render |= crate::ui::slider(
                ui,
                egui::Slider::new(n, 50..=20_000)
                    .logarithmic(true)
                    .text("max"),
            )
            .changed();
        }
        MaxIter::Auto => {
            ui.label(format!("max = {effective}"));
        }
    }

    ui.separator();
    ui.heading("Color");
    ch.color |= palette_combo(ui, &mut state.color.palette);
    ch.color |= crate::ui::slider(
        ui,
        egui::Slider::new(&mut state.color.offset, 0.0..=1.0).text("offset"),
    )
    .changed();
    ch.color |= crate::ui::slider(
        ui,
        egui::Slider::new(&mut state.color.scale, 1.0 / 256.0..=0.25)
            .logarithmic(true)
            .text("scale"),
    )
    .changed();
    let defaults = ColorSettings::default();
    let at_defaults = state.color.offset == defaults.offset && state.color.scale == defaults.scale;
    if ui
        .add_enabled(!at_defaults, egui::Button::new("Reset offset & scale"))
        .clicked()
    {
        state.color.offset = defaults.offset;
        state.color.scale = defaults.scale;
        ch.color = true;
    }

    ui.separator();
    ui.heading("Performance");
    ch.threads = threads_slider(ui, threads, max_threads);
    match state.last_render {
        Some((elapsed, threads, pixels)) => {
            let secs = elapsed.as_secs_f64().max(1e-9);
            ui.label(format!(
                "{:.0} ms · {threads} threads · {:.1} Mpx/s",
                secs * 1000.0,
                pixels as f64 / secs / 1e6
            ));
        }
        None => {
            ui.label("rendering…");
        }
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Reset view").clicked() {
            state.view = state.kind.default_view(state.view.width, state.view.height);
            ch.render = true;
        }
        ch.open_movie = ui.button("Movie…").clicked();
    });
    ch
}

/// Orbit inspector readout. Returns true if "Clear orbit" was clicked.
pub fn orbit_panel(ui: &mut egui::Ui, orbit: &OrbitState) -> bool {
    ui.heading("Orbit");
    note(ui, "orbit_note", NOTE_ORBIT);
    ui.monospace(format!(
        "start {:+.6} {:+.6}i",
        orbit.point.re, orbit.point.im
    ));
    ui.label(if orbit.live {
        "following the cursor · click to freeze"
    } else {
        "frozen · click to follow the cursor"
    });
    let steps = orbit.path.len().saturating_sub(1);
    ui.label(match orbit.outcome {
        Sample::Escaped { iter, .. } => format!("escaped after {iter} steps"),
        Sample::Inside => format!("stayed bounded for {steps} steps"),
        Sample::Converged { root, iter } => format!("converged to root {root} after {iter} steps"),
    });
    for (i, z) in orbit.path.iter().take(10).enumerate() {
        ui.monospace(format!("z{i:<2} {:+.5} {:+.5}i", z.re, z.im));
    }
    ui.button("Clear orbit").clicked()
}

/// Palette picker. Returns true if changed.
pub fn palette_combo(ui: &mut egui::Ui, palette: &mut PaletteId) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt("palette")
        .selected_text(palette.label())
        .show_ui(ui, |ui| {
            for p in PaletteId::ALL {
                changed |= ui.selectable_value(palette, p, p.label()).changed();
            }
        });
    changed
}

/// Thread-count slider. Returns the new count if changed.
pub fn threads_slider(ui: &mut egui::Ui, threads: usize, max_threads: usize) -> Option<usize> {
    let mut n = threads;
    crate::ui::slider(
        ui,
        egui::Slider::new(&mut n, 1..=max_threads).text("threads"),
    )
    .changed()
    .then_some(n)
}
