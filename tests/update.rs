//! `fractal update` end to end against a fake release server on localhost.
//! Unix-only: the fake release binaries are shell scripts.
#![cfg(unix)]

use fractal_explorer::update::{Outcome, Source, UpdateError, run, sha256_hex};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Serves `routes` (path → status, body) until dropped; records each path.
struct FakeServer {
    base: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl FakeServer {
    fn start(routes: HashMap<String, (u16, Vec<u8>)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                // Skip the headers.
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).map_or(true, |n| n == 0) || h == "\r\n" {
                        break;
                    }
                }
                log.lock().unwrap().push(path.clone());
                let (status, body) = routes
                    .get(&path)
                    .cloned()
                    .unwrap_or((404, b"Not Found".to_vec()));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(&body);
            }
        });
        Self { base, requests }
    }

    fn source(&self) -> Source {
        Source {
            api_base: format!("{}/api", self.base),
            download_base: format!("{}/dl", self.base),
        }
    }

    fn requested(&self, suffix: &str) -> bool {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.ends_with(suffix))
    }
}

const ASSET: &str = "fractal-linux-amd64";

/// A release `tag` whose asset is `asset` and whose checksum line says `hash`.
fn release(tag: &str, asset: &[u8], hash: &str) -> HashMap<String, (u16, Vec<u8>)> {
    let mut r = HashMap::new();
    r.insert(
        "/api/releases/latest".into(),
        (
            200,
            format!(r#"{{"tag_name":"{tag}","name":"x"}}"#).into_bytes(),
        ),
    );
    r.insert(format!("/dl/{tag}/{ASSET}"), (200, asset.to_vec()));
    r.insert(
        format!("/dl/{tag}/checksums.txt"),
        (
            200,
            format!("{hash}  {ASSET}\nffff  fractal-darwin-arm64\n").into_bytes(),
        ),
    );
    r
}

/// A stand-in binary that answers `version` like a real build of `tag`.
fn fake_binary(tag: &str) -> Vec<u8> {
    format!("#!/bin/sh\necho \"fractal {tag}\"\n").into_bytes()
}

fn good_release(tag: &str) -> HashMap<String, (u16, Vec<u8>)> {
    let bin = fake_binary(tag);
    release(tag, &bin, &sha256_hex(&bin))
}

fn old_exe() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let exe = dir.path().join("fractal");
    std::fs::write(&exe, "old").unwrap();
    (dir, exe)
}

fn contents(p: &PathBuf) -> String {
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn updates_when_newer() {
    let server = FakeServer::start(good_release("v0.2.0"));
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", false);
    assert_eq!(
        r.unwrap(),
        Outcome::Updated {
            from: "v0.1.0".into(),
            to: "v0.2.0".into()
        }
    );
    assert_eq!(contents(&exe).as_bytes(), fake_binary("v0.2.0"));
}

#[test]
fn download_that_will_not_run_keeps_old() {
    // Intact (checksum matches) but not runnable here: wrong arch, too-new glibc...
    let server = FakeServer::start(release("v0.2.0", b"garbage", &sha256_hex(b"garbage")));
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", false);
    assert!(matches!(r, Err(UpdateError::WontRun { .. })), "{r:?}");
    assert_eq!(contents(&exe), "old");
}

#[test]
fn download_reporting_wrong_version_keeps_old() {
    let bin = fake_binary("v0.1.9");
    let server = FakeServer::start(release("v0.2.0", &bin, &sha256_hex(&bin)));
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", false);
    match &r {
        Err(e @ UpdateError::WontRun { .. }) => assert!(e.to_string().contains("v0.1.9"), "{e}"),
        other => panic!("expected WontRun, got {other:?}"),
    }
    assert_eq!(contents(&exe), "old");
}

#[test]
fn check_only_does_not_download() {
    let server = FakeServer::start(good_release("v0.2.0"));
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", true);
    assert_eq!(
        r.unwrap(),
        Outcome::Available {
            latest: "v0.2.0".into()
        }
    );
    assert_eq!(contents(&exe), "old");
    assert!(!server.requested(ASSET));
}

#[test]
fn up_to_date() {
    let server = FakeServer::start(good_release("v0.1.0"));
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", false);
    assert_eq!(
        r.unwrap(),
        Outcome::UpToDate {
            latest: "v0.1.0".into()
        }
    );
}

#[test]
fn older_latest_is_not_a_downgrade() {
    let server = FakeServer::start(good_release("v0.1.0"));
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.3.0", &exe, "linux", "x86_64", false);
    assert_eq!(
        r.unwrap(),
        Outcome::UpToDate {
            latest: "v0.1.0".into()
        }
    );
    assert_eq!(contents(&exe), "old");
    assert!(!server.requested(ASSET));
}

#[test]
fn dev_build_refuses_without_network() {
    // Nothing listens here: a request would fail with Http, not DevBuild.
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", closed.local_addr().unwrap());
    drop(closed);
    let src = Source {
        api_base: base.clone(),
        download_base: base,
    };
    let (_d, exe) = old_exe();
    let r = run(&src, "dev", &exe, "linux", "x86_64", false);
    assert!(matches!(r, Err(UpdateError::DevBuild)), "{r:?}");
}

#[test]
fn checksum_mismatch_keeps_old() {
    let server = FakeServer::start(release(
        "v0.2.0",
        &fake_binary("v0.2.0"),
        &sha256_hex(b"other"),
    ));
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", false);
    assert!(matches!(r, Err(UpdateError::ChecksumMismatch(_))), "{r:?}");
    assert_eq!(contents(&exe), "old");
}

#[test]
fn checksum_entry_missing_keeps_old() {
    let server = FakeServer::start(good_release("v0.2.0"));
    let (_d, exe) = old_exe();
    // darwin-amd64 has no checksum line (and no asset): the 404 comes first.
    let r = run(&server.source(), "v0.1.0", &exe, "macos", "x86_64", false);
    assert!(r.is_err(), "{r:?}");
    let mut routes = good_release("v0.2.0");
    routes.insert(
        "/dl/v0.2.0/checksums.txt".into(),
        (200, b"ffff  fractal-darwin-arm64\n".to_vec()),
    );
    let server = FakeServer::start(routes);
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", false);
    assert!(matches!(r, Err(UpdateError::ChecksumMissing(_))), "{r:?}");
    assert_eq!(contents(&exe), "old");
}

#[test]
fn missing_asset_errors_without_replacing() {
    let mut routes = good_release("v0.2.0");
    routes.remove(&format!("/dl/v0.2.0/{ASSET}"));
    let server = FakeServer::start(routes);
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", false);
    match r {
        Err(UpdateError::Http {
            status: Some(404),
            url,
            ..
        }) => assert!(url.ends_with(ASSET), "{url}"),
        other => panic!("expected 404 for the asset, got {other:?}"),
    }
    assert_eq!(contents(&exe), "old");
}

#[test]
fn api_error_status_is_reported() {
    let mut routes = HashMap::new();
    routes.insert(
        "/api/releases/latest".to_string(),
        (403, br#"{"message":"API rate limit exceeded"}"#.to_vec()),
    );
    let server = FakeServer::start(routes);
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", true);
    match &r {
        Err(
            e @ UpdateError::Http {
                status: Some(403), ..
            },
        ) => assert!(e.to_string().contains("rate limit"), "{e}"),
        other => panic!("expected 403, got {other:?}"),
    }

    let mut routes = HashMap::new();
    routes.insert(
        "/api/releases/latest".to_string(),
        (200, b"not json".to_vec()),
    );
    let server = FakeServer::start(routes);
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", true);
    assert!(
        matches!(
            r,
            Err(UpdateError::BadTag(_)) | Err(UpdateError::Http { .. })
        ),
        "{r:?}"
    );
}

#[test]
fn bad_tag_is_rejected() {
    let mut routes = HashMap::new();
    routes.insert(
        "/api/releases/latest".to_string(),
        (200, br#"{"tag_name":"nightly"}"#.to_vec()),
    );
    let server = FakeServer::start(routes);
    let (_d, exe) = old_exe();
    let r = run(&server.source(), "v0.1.0", &exe, "linux", "x86_64", true);
    assert!(
        matches!(r, Err(UpdateError::BadTag(ref t)) if t == "nightly"),
        "{r:?}"
    );
}
