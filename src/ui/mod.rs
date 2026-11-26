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

/// Decimals the coordinate readout needs to tell neighbouring pixels
/// apart, `pixel_scale` apart: `max(12, ceil(−log10(pixel_scale)) + 3)`.
pub fn readout_digits(pixel_scale: f64) -> usize {
    let need = (-pixel_scale.log10()).ceil();
    if need.is_finite() && need > 0.0 {
        (need as usize + 3).max(12)
    } else {
        12
    }
}

/// Add a slider that takes keyboard focus when pressed, so ←/→ step it
/// (egui otherwise only focuses sliders with Tab).
pub fn slider(ui: &mut egui::Ui, slider: egui::Slider) -> egui::Response {
    let resp = ui.add(slider);
    if resp.is_pointer_button_down_on() && !resp.has_focus() {
        resp.request_focus();
    }
    resp
}

#[cfg(test)]
mod tests {
    #[test]
    fn readout_digits_follow_depth() {
        assert_eq!(super::readout_digits(1e-3), 12);
        assert_eq!(super::readout_digits(1e-100), 103);
    }
}
