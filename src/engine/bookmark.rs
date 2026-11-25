//! Saved views, stored as JSON in the user's config directory.

use crate::engine::color::ColorSettings;
use crate::engine::escape::FractalKind;
use crate::engine::ifs::IfsSystem;
use crate::engine::route::{RotationUnit, Stop, step_length};
use crate::engine::viewport::MaxIter;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Everything needed to get back to a picture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BookmarkView {
    Escape {
        kind: FractalKind,
        center: Complex64,
        span: f64,
        max_iter: MaxIter,
        color: ColorSettings,
    },
    Ifs {
        system: IfsSystem,
        center: Complex64,
        span: f64,
        points: u64,
        color: ColorSettings,
    },
}

impl BookmarkView {
    /// Height of the view in the complex plane.
    pub fn span(&self) -> f64 {
        match self {
            BookmarkView::Escape { span, .. } | BookmarkView::Ifs { span, .. } => *span,
        }
    }

    /// The same view (zoom, settings) centered on `c`.
    pub fn with_center(mut self, c: Complex64) -> Self {
        match &mut self {
            BookmarkView::Escape { center, .. } | BookmarkView::Ifs { center, .. } => *center = c,
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bookmark {
    pub name: String,
    pub view: BookmarkView,
    /// The folder this bookmark is filed in; `None` = unfiled.
    #[serde(default)]
    pub folder: Option<String>,
    /// How far the picture turns on the step arriving here when its
    /// folder plays as a path, in `rotation_unit`. Unused on the folder's
    /// shallowest bookmark.
    #[serde(default)]
    pub rotation: f64,
    #[serde(default)]
    pub rotation_unit: RotationUnit,
}

/// Distinct folder names, alphabetical.
pub fn folders(items: &[Bookmark]) -> Vec<String> {
    let mut names: Vec<String> = items.iter().filter_map(|b| b.folder.clone()).collect();
    names.sort();
    names.dedup();
    names
}

/// Indices of the bookmarks in `folder` (`None` = unfiled), shallowest
/// (largest span) first, ties by name. This is the order a folder plays in.
pub fn folder_order(items: &[Bookmark], folder: Option<&str>) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..items.len())
        .filter(|&i| items[i].folder.as_deref() == folder)
        .collect();
    idx.sort_by(|&a, &b| {
        items[b]
            .view
            .span()
            .total_cmp(&items[a].view.span())
            .then_with(|| items[a].name.cmp(&items[b].name))
    });
    idx
}

/// A folder played as a movie route.
#[derive(Clone, Debug, PartialEq)]
pub struct FolderPath {
    /// The fractal, iterations and colors of the shallowest stop.
    pub kind: FractalKind,
    pub max_iter: MaxIter,
    pub color: ColorSettings,
    /// (bookmark index, stop) in path order; each stop's rotation is its
    /// bookmark's.
    pub stops: Vec<(usize, Stop)>,
    /// Bookmarks left out: IFS, or a different fractal from the first stop.
    pub skipped: usize,
}

/// `folder` as a route, or `None` if it holds no escape-time bookmark.
/// A bookmark with the same view as the stop before it is dropped.
pub fn folder_path(items: &[Bookmark], folder: &str) -> Option<FolderPath> {
    let order = folder_order(items, Some(folder));
    let mut path: Option<FolderPath> = None;
    let mut skipped = 0;
    for i in order {
        let BookmarkView::Escape {
            kind,
            center,
            span,
            max_iter,
            color,
        } = items[i].view
        else {
            skipped += 1;
            continue;
        };
        let stop = Stop {
            center,
            span,
            rotation: items[i].rotation,
            rotation_unit: items[i].rotation_unit,
        };
        match &mut path {
            None => {
                path = Some(FolderPath {
                    kind,
                    max_iter,
                    color,
                    stops: vec![(i, stop)],
                    skipped: 0,
                })
            }
            Some(p) if p.kind != kind => skipped += 1,
            Some(p) => {
                let prev = &p.stops.last().expect("first stop pushed").1;
                if step_length(prev, &stop) > 0.0 {
                    p.stops.push((i, stop));
                }
            }
        }
    }
    path.map(|p| FolderPath { skipped, ..p })
}

#[derive(Debug)]
pub enum BookmarkError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    /// The bookmark contains NaN or infinity, which JSON can't store.
    NonFinite,
}

impl std::fmt::Display for BookmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BookmarkError::Io(e) => write!(f, "couldn't read or write bookmarks: {e}"),
            BookmarkError::Parse(e) => write!(f, "bookmarks file is damaged: {e}"),
            BookmarkError::NonFinite => write!(f, "can't save a view containing NaN or infinity"),
        }
    }
}

impl From<std::io::Error> for BookmarkError {
    fn from(e: std::io::Error) -> Self {
        BookmarkError::Io(e)
    }
}

/// A bookmarks JSON file.
pub struct BookmarkStore {
    path: PathBuf,
}

impl BookmarkStore {
    /// `<config dir>/fractal-explorer/bookmarks.json`
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("fractal-explorer")
            .join("bookmarks.json")
    }

    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// A missing file is an empty list; a damaged file is an error (and is
    /// left untouched so nothing is lost).
    pub fn load(&self) -> Result<Vec<Bookmark>, BookmarkError> {
        match std::fs::read_to_string(&self.path) {
            Ok(text) => serde_json::from_str(&text).map_err(BookmarkError::Parse),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e.into()),
        }
    }

    /// Write via a temp file + rename so a crash can't leave half a file.
    pub fn save(&self, items: &[Bookmark]) -> Result<(), BookmarkError> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let text = serde_json::to_string_pretty(items).map_err(BookmarkError::Parse)?;
        // serde_json writes NaN/inf as `null` and then can't read it back;
        // check the round trip so a bad value never damages the file.
        if serde_json::from_str::<Vec<Bookmark>>(&text).is_err() {
            return Err(BookmarkError::NonFinite);
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    /// Move a damaged file aside to `bookmarks.json.bad`.
    pub fn quarantine(&self) -> Result<(), BookmarkError> {
        let bad = self.path.with_extension("json.bad");
        std::fs::rename(&self.path, bad)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::color::ColorSettings;
    use crate::engine::escape::FractalKind;
    use crate::engine::ifs::IfsSystem;
    use crate::engine::viewport::MaxIter;
    use num_complex::Complex64;

    #[test]
    fn with_center_moves_only_the_center() {
        for b in samples() {
            let c = Complex64::new(0.25, -0.125);
            let moved = b.view.clone().with_center(c);
            assert_eq!(moved.span(), b.view.span());
            match (&moved, &b.view) {
                (
                    BookmarkView::Escape {
                        center,
                        kind,
                        max_iter,
                        color,
                        ..
                    },
                    BookmarkView::Escape {
                        kind: k,
                        max_iter: m,
                        color: col,
                        ..
                    },
                ) => {
                    assert_eq!(*center, c);
                    assert_eq!((kind, max_iter, color), (k, m, col));
                }
                (BookmarkView::Ifs { center, points, .. }, BookmarkView::Ifs { points: p, .. }) => {
                    assert_eq!(*center, c);
                    assert_eq!(points, p);
                }
                _ => panic!("kind changed"),
            }
        }
    }

    fn samples() -> Vec<Bookmark> {
        vec![
            Bookmark {
                folder: Some("dive".into()),
                rotation: 7.5,
                rotation_unit: RotationUnit::PerSecond,
                name: "seahorse".into(),
                view: BookmarkView::Escape {
                    kind: FractalKind::Mandelbrot,
                    center: Complex64::new(-0.743643887, 0.131825904),
                    span: 1e-6,
                    max_iter: MaxIter::Fixed(2000),
                    color: ColorSettings::default(),
                },
            },
            Bookmark {
                folder: None,
                rotation: 0.0,
                rotation_unit: RotationUnit::PerSecond,
                name: "fern".into(),
                view: BookmarkView::Ifs {
                    system: IfsSystem::barnsley_fern(),
                    center: Complex64::new(0.0, 5.0),
                    span: 10.5,
                    points: 5_000_000,
                    color: ColorSettings::default(),
                },
            },
        ]
    }

    #[test]
    fn missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = BookmarkStore::new(dir.path().join("bookmarks.json"));
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = BookmarkStore::new(dir.path().join("bookmarks.json"));
        store.save(&samples()).unwrap();
        assert_eq!(store.load().unwrap(), samples());
    }

    #[test]
    fn corrupt_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bookmarks.json");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(BookmarkStore::new(path).load().is_err());
    }

    #[test]
    fn save_creates_parent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let store = BookmarkStore::new(dir.path().join("a/b/bookmarks.json"));
        store.save(&samples()).unwrap();
        assert_eq!(store.load().unwrap().len(), 2);
    }

    #[test]
    fn non_finite_bookmark_is_refused_and_file_stays_readable() {
        let dir = tempfile::tempdir().unwrap();
        let store = BookmarkStore::new(dir.path().join("bookmarks.json"));
        store.save(&samples()).unwrap();
        let mut bad = samples();
        bad[0].view = BookmarkView::Escape {
            kind: FractalKind::Julia {
                c: Complex64::new(f64::NAN, 0.0),
            },
            center: Complex64::new(0.0, 0.0),
            span: 3.0,
            max_iter: MaxIter::Auto,
            color: ColorSettings::default(),
        };
        assert!(matches!(store.save(&bad), Err(BookmarkError::NonFinite)));
        assert_eq!(store.load().unwrap(), samples());
    }

    #[test]
    fn quarantine_moves_bad_file_aside() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bookmarks.json");
        std::fs::write(&path, "{ not json").unwrap();
        let store = BookmarkStore::new(path.clone());
        store.quarantine().unwrap();
        assert!(!path.exists());
        assert!(dir.path().join("bookmarks.json.bad").exists());
        assert!(store.load().unwrap().is_empty());
    }

    fn esc(name: &str, folder: Option<&str>, kind: FractalKind, re: f64, span: f64) -> Bookmark {
        Bookmark {
            name: name.into(),
            view: BookmarkView::Escape {
                kind,
                center: Complex64::new(re, 0.1),
                span,
                max_iter: MaxIter::Auto,
                color: ColorSettings::default(),
            },
            folder: folder.map(String::from),
            rotation: span.log10(),
            rotation_unit: RotationUnit::PerSecond,
        }
    }

    fn m(name: &str, folder: Option<&str>, re: f64, span: f64) -> Bookmark {
        esc(name, folder, FractalKind::Mandelbrot, re, span)
    }

    #[test]
    fn old_format_file_loads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bookmarks.json");
        let old: Vec<serde_json::Value> = samples()
            .iter()
            .map(|b| serde_json::json!({ "name": b.name, "view": b.view }))
            .collect();
        std::fs::write(&path, serde_json::to_string(&old).unwrap()).unwrap();
        let items = BookmarkStore::new(path).load().unwrap();
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|b| b.folder.is_none()
            && b.rotation == 0.0
            && b.rotation_unit == RotationUnit::PerSecond));
    }

    #[test]
    fn folder_order_is_shallowest_first_then_name() {
        let items = vec![
            m("b", Some("dive"), -0.7, 1e-3),
            m("z", Some("dive"), -0.5, 3.0),
            m("loose", None, -0.5, 10.0),
            m("a", Some("dive"), -0.71, 1e-3),
        ];
        let names: Vec<&str> = folder_order(&items, Some("dive"))
            .into_iter()
            .map(|i| items[i].name.as_str())
            .collect();
        assert_eq!(names, ["z", "a", "b"]);
        assert_eq!(folder_order(&items, None), [2]);
    }

    #[test]
    fn folders_are_distinct_and_sorted() {
        let items = vec![
            m("1", Some("zeta"), -0.5, 1.0),
            m("2", None, -0.5, 1.0),
            m("3", Some("alpha"), -0.5, 0.1),
            m("4", Some("zeta"), -0.5, 0.01),
        ];
        assert_eq!(folders(&items), ["alpha", "zeta"]);
    }

    #[test]
    fn folder_path_skips_other_fractals_and_ifs() {
        let julia = FractalKind::Julia {
            c: Complex64::new(-0.8, 0.156),
        };
        let mut ifs = samples()[1].clone();
        ifs.folder = Some("dive".into());
        let items = vec![
            m("mid", Some("dive"), -0.74, 0.01),
            esc("j", Some("dive"), julia, 0.0, 1.0),
            m("top", Some("dive"), -0.5, 3.0),
            ifs,
            m("deep", Some("dive"), -0.743, 1e-5),
            m("elsewhere", Some("other"), -0.5, 2.0),
        ];
        let p = folder_path(&items, "dive").unwrap();
        assert_eq!(p.kind, FractalKind::Mandelbrot);
        assert_eq!(p.skipped, 2);
        let idx: Vec<usize> = p.stops.iter().map(|(i, _)| *i).collect();
        assert_eq!(idx, [2, 0, 4]);
        assert_eq!(p.stops[2].1.rotation, items[4].rotation);
        assert_eq!(p.stops[0].1.span, 3.0);
        assert!(folder_path(&items, "nothing").is_none());
    }

    #[test]
    fn folder_path_copies_rotation_unit() {
        let mut items = vec![
            m("top", Some("dive"), -0.5, 3.0),
            m("deep", Some("dive"), -0.743, 1e-5),
        ];
        items[1].rotation_unit = RotationUnit::Total;
        let p = folder_path(&items, "dive").unwrap();
        assert_eq!(p.stops[1].1.rotation_unit, RotationUnit::Total);
        assert_eq!(p.stops[0].1.rotation_unit, RotationUnit::PerSecond);
    }

    #[test]
    fn folder_path_drops_duplicate_stops() {
        let items = vec![
            m("top", Some("dive"), -0.5, 3.0),
            m("one", Some("dive"), -0.74, 0.01),
            m("two", Some("dive"), -0.74, 0.01),
        ];
        let p = folder_path(&items, "dive").unwrap();
        assert_eq!(p.stops.len(), 2);
        assert_eq!(p.skipped, 0);
    }
}
