//! Self-update from GitHub Releases: find the latest `vX.Y.Z` release,
//! download this platform's binary, check its sha256 against the
//! release's `checksums.txt`, and swap it in for the running executable.

use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;

/// Where people can download a release by hand.
pub const RELEASES_PAGE: &str = "https://github.com/pfarrell/fractal/releases";

#[derive(Debug)]
pub enum UpdateError {
    /// A local build has no release version to compare.
    DevBuild,
    Http {
        url: String,
        status: Option<u16>,
        detail: String,
    },
    BadTag(String),
    UnsupportedPlatform {
        os: String,
        arch: String,
    },
    ChecksumMissing(String),
    ChecksumMismatch(String),
    Replace(String),
}

impl std::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateError::DevBuild => {
                write!(
                    f,
                    "development build; install a release or rebuild from source"
                )
            }
            UpdateError::Http {
                url,
                status: Some(code),
                detail,
            } => write!(f, "{url}: HTTP {code} {detail}"),
            UpdateError::Http {
                url,
                status: None,
                detail,
            } => write!(f, "{url}: {detail}"),
            UpdateError::BadTag(tag) => {
                write!(f, "latest release tag {tag:?} is not vMAJOR.MINOR.PATCH")
            }
            UpdateError::UnsupportedPlatform { os, arch } => write!(
                f,
                "no release build for {os}/{arch}; download one from {RELEASES_PAGE} \
                 or build from source"
            ),
            UpdateError::ChecksumMissing(asset) => {
                write!(
                    f,
                    "checksums.txt has no entry for {asset}; not installing it"
                )
            }
            UpdateError::ChecksumMismatch(asset) => write!(
                f,
                "{asset} does not match its checksum (corrupt download?); not installing it"
            ),
            UpdateError::Replace(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for UpdateError {}

/// `vMAJOR.MINOR.PATCH` with no leading zeros and nothing after it.
pub fn parse_tag(tag: &str) -> Option<(u64, u64, u64)> {
    let mut parts = tag.strip_prefix('v')?.split('.');
    let mut next = || {
        let p = parts.next()?;
        let ok = !p.is_empty()
            && p.bytes().all(|b| b.is_ascii_digit())
            && (p == "0" || !p.starts_with('0'));
        ok.then(|| p.parse().ok()).flatten()
    };
    let v = (next()?, next()?, next()?);
    parts.next().is_none().then_some(v)
}

/// Whether `latest` is a strictly higher release than `current`.
pub fn is_newer(current: &str, latest: &str) -> bool {
    match (parse_tag(current), parse_tag(latest)) {
        (Some(c), Some(l)) => l > c,
        _ => false,
    }
}

/// The release file for `std::env::consts::{OS, ARCH}`.
pub fn asset_name(os: &str, arch: &str) -> Result<String, UpdateError> {
    let unsupported = || UpdateError::UnsupportedPlatform {
        os: os.into(),
        arch: arch.into(),
    };
    let os_name = match os {
        "macos" => "darwin",
        "linux" => "linux",
        _ => return Err(unsupported()),
    };
    let arch_name = match arch {
        "aarch64" => "arm64",
        "x86_64" => "amd64",
        _ => return Err(unsupported()),
    };
    Ok(format!("fractal-{os_name}-{arch_name}"))
}

/// The hash `sha256sum` recorded for `asset`, lowercased.
pub fn expected_sha256(checksums: &str, asset: &str) -> Option<String> {
    checksums.lines().find_map(|line| {
        let (hash, name) = line.split_once(char::is_whitespace)?;
        let name = name.trim_start();
        let name = name.strip_prefix('*').unwrap_or(name);
        (name == asset).then(|| hash.to_ascii_lowercase())
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Atomically replace the file behind `exe` (following symlinks) with
/// `contents`. A temp file in the same folder is written, flushed and
/// renamed over the target, so any failure leaves the old binary intact.
/// A running process keeps its old inode, so replacing itself is safe.
pub fn replace_exe(exe: &Path, contents: &[u8]) -> Result<(), UpdateError> {
    let fail = |what: String| move |e: std::io::Error| UpdateError::Replace(format!("{what}: {e}"));
    let target = std::fs::canonicalize(exe).map_err(fail(format!("resolve {}", exe.display())))?;
    let dir = target
        .parent()
        .ok_or_else(|| UpdateError::Replace(format!("{} has no folder", target.display())))?;
    let mut tmp = tempfile::Builder::new()
        .prefix(".fractal-update-")
        .tempfile_in(dir)
        .map_err(fail(format!(
            "cannot write to {} (is it writable?)",
            dir.display()
        )))?;
    let write_err = fail("write update".into());
    tmp.write_all(contents).map_err(&write_err)?;
    tmp.as_file().sync_all().map_err(&write_err)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tmp.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o755))
            .map_err(fail("set permissions".into()))?;
    }
    tmp.persist(&target)
        .map_err(|e| UpdateError::Replace(format!("replace {}: {}", target.display(), e.error)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn tags() {
        assert_eq!(parse_tag("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_tag("v0.10.0"), Some((0, 10, 0)));
        for bad in [
            "1.2.3",
            "v1.2",
            "v1.2.3-rc1",
            "v01.2.3",
            "",
            "dev",
            "v1.2.3.4",
            "v1..3",
        ] {
            assert_eq!(parse_tag(bad), None, "{bad}");
        }
    }

    #[test]
    fn newer_is_numeric() {
        assert!(is_newer("v0.9.0", "v0.10.0"));
        assert!(!is_newer("v1.0.0", "v1.0.0"));
        assert!(!is_newer("v1.2.0", "v1.1.9"));
        assert!(!is_newer("v1.0.0", "garbage"));
        assert!(!is_newer("dev", "v1.0.0"));
    }

    #[test]
    fn assets() {
        assert_eq!(
            asset_name("macos", "aarch64").unwrap(),
            "fractal-darwin-arm64"
        );
        assert_eq!(
            asset_name("macos", "x86_64").unwrap(),
            "fractal-darwin-amd64"
        );
        assert_eq!(
            asset_name("linux", "x86_64").unwrap(),
            "fractal-linux-amd64"
        );
        assert_eq!(
            asset_name("linux", "aarch64").unwrap(),
            "fractal-linux-arm64"
        );
        let err = asset_name("windows", "x86_64").unwrap_err();
        assert!(matches!(err, UpdateError::UnsupportedPlatform { .. }));
        assert!(
            err.to_string()
                .contains("https://github.com/pfarrell/fractal/releases"),
            "{err}"
        );
    }

    #[test]
    fn checksum_lines() {
        let sums =
            "AAA  fractal-linux-amd64\nbbb *fractal-darwin-arm64\nccc  fractal-linux-amd64.bak\n";
        assert_eq!(
            expected_sha256(sums, "fractal-linux-amd64").as_deref(),
            Some("aaa")
        );
        assert_eq!(
            expected_sha256(sums, "fractal-darwin-arm64").as_deref(),
            Some("bbb")
        );
        assert_eq!(expected_sha256(sums, "fractal-linux-arm64"), None);
    }

    #[test]
    fn sha_of_known_input() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn dev_error_text() {
        assert_eq!(
            UpdateError::DevBuild.to_string(),
            "development build; install a release or rebuild from source"
        );
    }

    #[test]
    fn replace_swaps_contents_and_is_executable() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("fractal");
        fs::write(&exe, "old").unwrap();
        replace_exe(&exe, b"new").unwrap();
        assert_eq!(fs::read_to_string(&exe).unwrap(), "new");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_ne!(fs::metadata(&exe).unwrap().permissions().mode() & 0o111, 0);
        }
        let names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["fractal"], "no temp file left behind");
    }

    #[cfg(unix)]
    #[test]
    fn replace_follows_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let real_dir = dir.path().join("real");
        let bin_dir = dir.path().join("bin");
        fs::create_dir_all(&real_dir).unwrap();
        fs::create_dir_all(&bin_dir).unwrap();
        let real = real_dir.join("fractal");
        let link = bin_dir.join("fractal");
        fs::write(&real, "old").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();
        replace_exe(&link, b"new").unwrap();
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(&real).unwrap(), "new");
    }

    #[cfg(unix)]
    #[test]
    fn replace_unwritable_dir_errors_and_keeps_old() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("ro");
        fs::create_dir(&sub).unwrap();
        let exe = sub.join("fractal");
        fs::write(&exe, "old").unwrap();
        fs::set_permissions(&sub, fs::Permissions::from_mode(0o555)).unwrap();
        // Root ignores directory permissions; nothing to test then.
        let writable = fs::write(sub.join("probe"), "").is_ok();
        let result = (!writable).then(|| replace_exe(&exe, b"new"));
        fs::set_permissions(&sub, fs::Permissions::from_mode(0o755)).unwrap();
        let Some(result) = result else { return };
        match result {
            Err(UpdateError::Replace(msg)) => {
                assert!(msg.contains(&sub.display().to_string()), "{msg}")
            }
            other => panic!("expected Replace error, got {other:?}"),
        }
        assert_eq!(fs::read_to_string(&exe).unwrap(), "old");
    }
}
