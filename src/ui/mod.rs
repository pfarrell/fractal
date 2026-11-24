//! eframe/egui front end.

pub mod app;
pub mod bookmarks;
pub mod ifs_table;
pub mod julia;
pub mod movie_dialog;
pub mod notes;
pub mod panel;
pub mod preview;
pub mod view;

use eframe::egui;

/// Add a slider that takes keyboard focus when pressed, so ←/→ step it
/// (egui otherwise only focuses sliders with Tab).
pub fn slider(ui: &mut egui::Ui, slider: egui::Slider) -> egui::Response {
    let resp = ui.add(slider);
    if resp.is_pointer_button_down_on() && !resp.has_focus() {
        resp.request_focus();
    }
    resp
}
