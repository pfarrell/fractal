//! Command-line entry: `fractal` opens the GUI; a few subcommands manage
//! the installed binary.

/// The release this binary was built from (`FRACTAL_VERSION` at build
/// time), or `dev` for a local build.
pub const VERSION: &str = match option_env!("FRACTAL_VERSION") {
    Some(v) => v,
    None => "dev",
};

pub const USAGE: &str = "\
usage:
  fractal                   open the fractal explorer
  fractal update [--check]  install the latest release (--check only reports)
  fractal version           print the version
  fractal help              show this help";

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Gui,
    Update { check: bool },
    Version,
    Help,
}

/// Parse the arguments after the program name.
pub fn parse(args: &[String]) -> Result<Command, String> {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        [] => Ok(Command::Gui),
        ["update"] => Ok(Command::Update { check: false }),
        ["update", "--check"] => Ok(Command::Update { check: true }),
        ["update", flag, ..] => Err(format!("fractal update: unknown option {flag}")),
        ["version" | "--version"] => Ok(Command::Version),
        ["help" | "--help" | "-h"] => Ok(Command::Help),
        [
            cmd @ ("version" | "--version" | "help" | "--help" | "-h"),
            extra,
            ..,
        ] => Err(format!("fractal {cmd}: unexpected argument {extra}")),
        [cmd, ..] => Err(format!("unknown command {cmd}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(a: &[&str]) -> Result<Command, String> {
        parse(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn no_args_is_gui() {
        assert_eq!(p(&[]), Ok(Command::Gui));
    }

    #[test]
    fn update_and_check() {
        assert_eq!(p(&["update"]), Ok(Command::Update { check: false }));
        assert_eq!(
            p(&["update", "--check"]),
            Ok(Command::Update { check: true })
        );
    }

    #[test]
    fn version_and_help() {
        assert_eq!(p(&["version"]), Ok(Command::Version));
        assert_eq!(p(&["--version"]), Ok(Command::Version));
        for h in ["help", "--help", "-h"] {
            assert_eq!(p(&[h]), Ok(Command::Help));
        }
    }

    #[test]
    fn unknown_is_error() {
        assert!(p(&["frobnicate"]).unwrap_err().contains("frobnicate"));
        assert!(p(&["update", "--force"]).unwrap_err().contains("--force"));
        assert!(p(&["version", "extra"]).is_err());
    }

    #[test]
    fn dev_build_by_default() {
        if option_env!("FRACTAL_VERSION").is_none() {
            assert_eq!(VERSION, "dev");
        }
    }
}
