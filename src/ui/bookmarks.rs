//! The bookmarks window.

use crate::engine::bookmark::{Bookmark, BookmarkStore, BookmarkView};
use eframe::egui;

pub struct BookmarksUi {
    pub open: bool,
    pub items: Vec<Bookmark>,
    store: BookmarkStore,
    /// Set if the file couldn't be loaded; saving is disabled until the
    /// user resets, so a damaged file is never silently overwritten.
    error: Option<String>,
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
            error,
            name: String::new(),
        }
    }

    fn persist(&mut self) {
        if let Err(e) = self.store.save(&self.items) {
            self.error = Some(e.to_string());
        }
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
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(240, 90, 90), err);
                    ui.small(format!("File: {}", self.store.path().display()));
                    if ui
                        .button("Reset bookmarks (moves the file to .bad)")
                        .clicked()
                    {
                        match self.store.quarantine() {
                            Ok(()) => {
                                self.error = None;
                                self.items.clear();
                            }
                            Err(e) => self.error = Some(e.to_string()),
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
                        self.items.push(Bookmark {
                            name,
                            view: current(),
                        });
                        self.name.clear();
                        self.persist();
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
                    self.items.remove(i);
                    self.persist();
                }
            });
        self.open = open;
        go
    }
}
