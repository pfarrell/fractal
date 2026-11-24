use fractal_explorer::engine::soundtrack::{filter_graph, mux_args, plan_soundtrack};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[test]
fn long_track_plays_once() {
    let p = plan_soundtrack(&[600.0], 300.0).unwrap();
    assert_eq!(p.sequence, vec![0]);
    assert_eq!(
        (p.crossfade, p.fade_in, p.fade_out, p.length),
        (4.0, 3.0, 3.0, 300.0)
    );
}

#[test]
fn short_track_loops_onto_itself() {
    // 100, then +100-4 = 196 < 250, then +100-4 = 292 >= 250.
    assert_eq!(
        plan_soundtrack(&[100.0], 250.0).unwrap().sequence,
        vec![0, 0, 0]
    );
}

#[test]
fn playlist_wraps_in_order() {
    // 60, then +50-4 = 106, then +60-4 = 162 >= 150.
    assert_eq!(
        plan_soundtrack(&[60.0, 50.0], 150.0).unwrap().sequence,
        vec![0, 1, 0]
    );
}

#[test]
fn short_tracks_shrink_the_crossfade() {
    assert_eq!(plan_soundtrack(&[60.0, 5.0], 30.0).unwrap().crossfade, 2.5);
}

#[test]
fn one_second_movie_caps_fades() {
    let p = plan_soundtrack(&[60.0], 1.0).unwrap();
    assert_eq!(p.sequence, vec![0]);
    assert!((p.fade_in - 1.0 / 3.0).abs() < 1e-12);
    assert!((p.fade_out - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn too_many_plays_is_refused() {
    let e = plan_soundtrack(&[10.0], 3600.0).unwrap_err();
    assert!(
        e.starts_with("tracks too short for a movie this long (needs "),
        "{e}"
    );
    assert!(e.ends_with(" plays, limit 500)"), "{e}");
    // 10, then +6 per play: 10 + 6*(n-1) >= 3600 → n = 600.
    assert!(e.contains("needs 600 plays"), "{e}");
}

#[test]
fn empty_or_bad_lengths_are_refused() {
    assert!(plan_soundtrack(&[], 10.0).is_err());
    assert!(plan_soundtrack(&[0.0], 10.0).is_err());
    assert!(plan_soundtrack(&[f64::NAN], 10.0).is_err());
    assert!(plan_soundtrack(&[10.0], 0.0).is_err());
}

#[test]
fn graph_for_one_input() {
    let p = plan_soundtrack(&[600.0], 300.0).unwrap();
    assert_eq!(
        filter_graph(&p),
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a1];\
         [a1]atrim=0:300.000,asetpts=N/SR/TB,afade=t=in:d=3.000,afade=t=out:st=297.000:d=3.000[aout]"
    );
}

#[test]
fn graph_chains_crossfades() {
    let p = plan_soundtrack(&[60.0, 50.0], 150.0).unwrap();
    assert_eq!(
        filter_graph(&p),
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a1];\
         [2:a]aformat=sample_rates=48000:channel_layouts=stereo[a2];\
         [3:a]aformat=sample_rates=48000:channel_layouts=stereo[a3];\
         [a1][a2]acrossfade=d=4.000:c1=tri:c2=tri[x2];\
         [x2][a3]acrossfade=d=4.000:c1=tri:c2=tri[x3];\
         [x3]atrim=0:150.000,asetpts=N/SR/TB,afade=t=in:d=3.000,afade=t=out:st=147.000:d=3.000[aout]"
    );
}

#[test]
fn mux_args_exact() {
    let p = plan_soundtrack(&[60.0, 50.0], 150.0).unwrap();
    let tracks = [PathBuf::from("/m/a.mp3"), PathBuf::from("/m/b.mp3")];
    let got = mux_args(
        Path::new("/v/in.mp4"),
        &tracks,
        &p,
        Path::new("/t/g.txt"),
        Path::new("/v/out.mp4"),
    );
    let want: Vec<OsString> = [
        "-y",
        "-v",
        "error",
        "-i",
        "/v/in.mp4",
        "-i",
        "/m/a.mp3",
        "-i",
        "/m/b.mp3",
        "-i",
        "/m/a.mp3",
        "-filter_complex_script",
        "/t/g.txt",
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "/v/out.mp4",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    assert_eq!(got, want);
}

#[test]
fn mkv_mux_has_no_faststart() {
    let p = plan_soundtrack(&[60.0], 10.0).unwrap();
    let a = [PathBuf::from("a.mp3")];
    let mkv = mux_args(
        Path::new("i.mkv"),
        &a,
        &p,
        Path::new("g"),
        Path::new("o.MKV"),
    );
    assert!(!mkv.iter().any(|x| x == "-movflags"));
    let mov = mux_args(
        Path::new("i.mov"),
        &a,
        &p,
        Path::new("g"),
        Path::new("o.MOV"),
    );
    assert!(mov.iter().any(|x| x == "+faststart"));
}
