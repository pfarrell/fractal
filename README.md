# Fractal Explorer

An interactive, multithreaded fractal explorer (Mandelbrot, Julia, Burning
Ship, Tricorn, Multibrot, Newton, IFS) written in Rust with eframe/egui. It runs on macOS and Linux.

## Build and run

```sh
make run          # build and open the explorer
make check        # fmt-check, clippy, tests: run before committing
make help         # every target
```

## Install

Download the file for your machine from the
[releases page](https://github.com/pfarrell/fractal/releases):

| Machine | File |
|---|---|
| Mac, Apple silicon | `fractal-darwin-arm64` |
| Mac, Intel | `fractal-darwin-amd64` |
| Linux, x86-64 | `fractal-linux-amd64` |
| Linux, ARM64 | `fractal-linux-arm64` |

```sh
curl -fL --create-dirs -o ~/.local/bin/fractal \
  https://github.com/pfarrell/fractal/releases/latest/download/fractal-darwin-arm64
chmod +x ~/.local/bin/fractal
```

Or build and install from source with `make install` (to `~/.local/bin`; set
`PREFIX=` to change it). Make sure that folder is on your `PATH`.

A binary downloaded with a browser is quarantined by macOS and blocked as
unsigned; `curl` and `fractal update` downloads are not.

Movie export needs `ffmpeg` (`brew install ffmpeg` on macOS,
`sudo apt install ffmpeg` on Debian/Ubuntu). The rest of the app works
without it.

## Update

```sh
fractal update          # install the latest release over this binary
fractal update --check  # only report whether there is one
fractal version
```

`fractal update` asks GitHub for the latest release, downloads the file for
your platform, checks it against the release's `checksums.txt`, and swaps it
in atomically, but only after the new binary runs here and reports the
expected version; if anything fails the old binary stays. It can run while the
explorer is open (restart the app to use the new version). A build from
source reports `dev` and does not update itself.

## Releasing

The version comes from the tag alone:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

That runs `.github/workflows/release.yml`, which checks the code, builds the
four binaries above on macOS and Ubuntu 22.04 runners (so the Linux ones
need only glibc 2.35), and publishes them with a
`checksums.txt`. Tags must look like `vMAJOR.MINOR.PATCH`. GitHub marks the
newest release as Latest, which is what `fractal update` installs, so when
tagging a hotfix on an older line after a newer release exists, publish it
with `gh release edit <tag> --latest=false`.
