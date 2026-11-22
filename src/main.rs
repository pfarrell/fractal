use eframe::egui;
use fractal_explorer::cli::{self, Command, USAGE, VERSION};
use fractal_explorer::ui::app::FractalApp;
use fractal_explorer::update::{self, Outcome, Source};
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
        Ok(Command::Update { check }) => update(check),
        Err(msg) => {
            eprintln!("{msg}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn update(check: bool) -> ExitCode {
    let result = std::env::current_exe()
        .map_err(|e| update::UpdateError::Replace(format!("find running binary: {e}")))
        .and_then(|exe| {
            update::run(
                &Source::github(),
                VERSION,
                &exe,
                std::env::consts::OS,
                std::env::consts::ARCH,
                check,
            )
        });
    match result {
        Ok(Outcome::UpToDate { latest }) => {
            println!("fractal {VERSION} is up to date (latest release {latest})");
        }
        Ok(Outcome::Available { latest }) => {
            println!("update available: {latest} (you have {VERSION}); run 'fractal update'");
        }
        Ok(Outcome::Updated { from, to }) => println!("updated {from} -> {to}"),
        Err(e) => {
            eprintln!("fractal update: {e}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
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
