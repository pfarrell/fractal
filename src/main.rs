use eframe::egui;
use fractal_explorer::cli::{self, Command, USAGE, VERSION};
use fractal_explorer::ui::app::FractalApp;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match cli::parse(&args) {
        Ok(Command::Gui) => match run_gui() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("fractal: {e}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Version) => {
            println!("fractal {VERSION}");
            ExitCode::SUCCESS
        }
        Ok(Command::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Update { .. }) => {
            eprintln!("fractal update: not yet implemented");
            ExitCode::FAILURE
        }
        Err(msg) => {
            eprintln!("{msg}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run_gui() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Fractal Explorer")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Fractal Explorer",
        options,
        Box::new(|cc| Ok(Box::new(FractalApp::new(cc)))),
    )
}
