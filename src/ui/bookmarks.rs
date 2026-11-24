//! The bookmarks window.

use crate::engine::bookmark::{Bookmark, BookmarkError, BookmarkStore, BookmarkView};
use eframe::egui;

pub struct BookmarksUi {
    pub open: bool,
    pub items: Vec<Bookmark>,
    store: BookmarkStore,
    /// Set if the file couldn't be loaded; saving is disabled until the
    /// user resets, so a damaged file is never silently overwritten.
    load_error: Option<String>,
    /// The last save failed (the list stays usable).
    save_error: Option<String>,
    name: String,
}

impl Default for BookmarksUi {
    fn default() -> Self {
        Self::with_store(BookmarkStore::new(BookmarkStore::default_path()))
    }
}

impl BookmarksUi {
    pub fn with_store(store: BookmarkStore) -> Self {
        let (items, error) = match store.load() {
            Ok(items) => (items, None),
            Err(e) => (Vec::new(), Some(e.to_string())),
        };
        Self {
            open: false,
            items,
            store,
            load_error: error,
            save_error: None,
            name: String::new(),
        }
    }

    fn persist(&mut self) {
        self.save_error = self.store.save(&self.items).err().map(|e| e.to_string());
    }

    /// Add a bookmark and save. A non-saveable bookmark is not kept.
    pub fn add(&mut self, name: String, view: BookmarkView) {
        self.items.push(Bookmark {
            name,
            view,
            folder: None,
            rotation: 0.0,
        });
        match self.store.save(&self.items) {
            Ok(()) => self.save_error = None,
            Err(e) => {
                if matches!(e, BookmarkError::NonFinite) {
                    self.items.pop(); // could never be saved; don't keep it
                }
                self.save_error = Some(e.to_string());
            }
        }
    }

    pub fn delete(&mut self, i: usize) {
        self.items.remove(i);
        self.persist();
    }

    /// Show the window. `current` describes the picture on screen now.
    /// Returns a bookmark the user chose to go to.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        current: impl FnOnce() -> BookmarkView,
    ) -> Option<BookmarkView> {
        if !self.open {
            return None;
        }
        let mut go = None;
        let mut open = self.open;
        egui::Window::new("Bookmarks")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                if let Some(err) = &self.load_error {
                    ui.colored_label(egui::Color32::from_rgb(240, 90, 90), err);
                    ui.small(format!("File: {}", self.store.path().display()));
                    if ui
                        .button("Reset bookmarks (moves the file to .bad)")
                        .clicked()
                    {
                        match self.store.quarantine() {
                            Ok(()) => {
                                self.load_error = None;
                                self.items.clear();
                            }
                            Err(e) => self.load_error = Some(e.to_string()),
                        }
                    }
                    return;
                }
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.name);
                    let name = self.name.trim().to_string();
                    if ui
                        .add_enabled(!name.is_empty(), egui::Button::new("Save current"))
                        .clicked()
                    {
                        self.add(name, current());
                        self.name.clear();
                    }
                });
                ui.separator();
                let mut delete = None;
                for (i, b) in self.items.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let kind = match &b.view {
                            BookmarkView::Escape { kind, .. } => kind.label().to_string(),
                            BookmarkView::Ifs { system, .. } => system.name.clone(),
                        };
                        if ui.button("Go").clicked() {
                            go = Some(b.view.clone());
                        }
                        if ui.button("🗑").on_hover_text("Delete").clicked() {
                            delete = Some(i);
                        }
                        ui.label(format!("{}  ({kind})", b.name));
                    });
                }
                if self.items.is_empty() {
                    ui.label("No bookmarks yet.");
                }
                if let Some(i) = delete {
                    self.delete(i);
                }
                if let Some(err) = &self.save_error {
                    ui.colored_label(egui::Color32::from_rgb(240, 170, 40), err);
                }
            });
        self.open = open;
        go
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::color::ColorSettings;
    use crate::engine::escape::FractalKind;
    use crate::engine::viewport::MaxIter;
    use num_complex::Complex64;

    fn view() -> BookmarkView {
        BookmarkView::Escape {
            kind: FractalKind::Mandelbrot,
            center: Complex64::new(-0.5, 0.0),
            span: 3.0,
            max_iter: MaxIter::Auto,
            color: ColorSettings::default(),
        }
    }

    #[test]
    fn failed_save_keeps_list_and_is_not_a_load_error() {
        let mut ui = BookmarksUi::with_store(BookmarkStore::new("/proc/no/bookmarks.json".into()));
        ui.add("home".into(), view());
        assert_eq!(ui.items.len(), 1);
        assert!(ui.load_error.is_none());
        assert!(ui.save_error.is_some());
    }
}
