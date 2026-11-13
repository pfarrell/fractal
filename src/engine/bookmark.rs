//! Saved views, stored as JSON in the user's config directory.

use crate::engine::color::ColorSettings;
use crate::engine::escape::FractalKind;
use crate::engine::ifs::IfsSystem;
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bookmark {
    pub name: String,
    pub view: BookmarkView,
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

    fn samples() -> Vec<Bookmark> {
        vec![
            Bookmark {
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
}
