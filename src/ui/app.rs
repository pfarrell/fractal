//! Top-level application state and layout.

use eframe::egui;

/// The eframe application.
#[derive(Default)]
pub struct FractalApp {}

impl eframe::App for FractalApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.label("Fractal Explorer");
        });
    }
}
