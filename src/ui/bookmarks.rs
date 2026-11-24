//! The bookmarks window: unfiled bookmarks, then one section per folder,
//! each in zoom order (the order a folder plays as a movie path).

use crate::engine::bookmark::{
    Bookmark, BookmarkError, BookmarkStore, BookmarkView, folder_order, folders,
};
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
    /// Folder "Save current" files into.
    save_folder: FolderChoice,
    /// Name typed for a new folder (save row and move menus share it).
    new_folder: String,
}

/// Where a bookmark goes.
#[derive(Clone, Debug, PartialEq)]
enum FolderChoice {
    Unfiled,
    Existing(String),
    New,
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
            save_folder: FolderChoice::Unfiled,
            new_folder: String::new(),
        }
    }

    fn persist(&mut self) {
        self.save_error = self.store.save(&self.items).err().map(|e| e.to_string());
    }

    /// Add a bookmark and save. A non-saveable bookmark is not kept.
    pub fn add(&mut self, name: String, view: BookmarkView, folder: Option<String>) {
        self.items.push(Bookmark {
            name,
            view,
            folder,
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

    /// File bookmark `i` in `folder` (`None` = unfiled) and save.
    pub fn set_folder(&mut self, i: usize, folder: Option<String>) {
        self.items[i].folder = folder;
        self.persist();
    }

    /// Set the rotation (°/s) of the step arriving at bookmark `i`, and save.
    pub fn set_rotation(&mut self, i: usize, deg_per_s: f64) {
        self.items[i].rotation = deg_per_s;
        self.persist();
    }

    /// The new-folder name, if one has been typed.
    fn typed_folder(&self) -> Option<String> {
        let name = self.new_folder.trim();
        (!name.is_empty()).then(|| name.to_string())
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
                let names = folders(&self.items);
                self.save_row(ui, &names, current);
                ui.separator();
                let mut action = None;
                for i in folder_order(&self.items, None) {
                    self.row(ui, i, &names, &mut go, &mut action);
                }
                for f in &names {
                    egui::CollapsingHeader::new(format!("📁 {f}"))
                        .id_salt(("bookmark_folder", f))
                        .default_open(true)
                        .show(ui, |ui| {
                            for i in folder_order(&self.items, Some(f)) {
                                self.row(ui, i, &names, &mut go, &mut action);
                            }
                        });
                }
                if self.items.is_empty() {
                    ui.label("No bookmarks yet.");
                }
                match action {
                    Some(RowAction::Delete(i)) => self.delete(i),
                    Some(RowAction::Move(i, f)) => self.set_folder(i, f),
                    None => {}
                }
                if let Some(err) = &self.save_error {
                    ui.colored_label(egui::Color32::from_rgb(240, 170, 40), err);
                }
            });
        self.open = open;
        go
    }

    /// Name, folder and "Save current".
    fn save_row(
        &mut self,
        ui: &mut egui::Ui,
        names: &[String],
        current: impl FnOnce() -> BookmarkView,
    ) {
        if let FolderChoice::Existing(f) = &self.save_folder
            && !names.contains(f)
        {
            self.save_folder = FolderChoice::Unfiled;
        }
        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.name);
            let label = match &self.save_folder {
                FolderChoice::Unfiled => "Unfiled".to_string(),
                FolderChoice::Existing(f) => f.clone(),
                FolderChoice::New => "New folder…".to_string(),
            };
            egui::ComboBox::from_id_salt("bookmark_save_folder")
                .selected_text(label)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.save_folder, FolderChoice::Unfiled, "Unfiled");
                    for f in names {
                        ui.selectable_value(
                            &mut self.save_folder,
                            FolderChoice::Existing(f.clone()),
                            f,
                        );
                    }
                    ui.selectable_value(&mut self.save_folder, FolderChoice::New, "New folder…");
                });
        });
        ui.horizontal(|ui| {
            let folder = match &self.save_folder {
                FolderChoice::Unfiled => Some(None),
                FolderChoice::Existing(f) => Some(Some(f.clone())),
                FolderChoice::New => {
                    ui.label("Folder name");
                    ui.text_edit_singleline(&mut self.new_folder);
                    self.typed_folder().map(Some)
                }
            };
            let name = self.name.trim().to_string();
            let ready = !name.is_empty() && folder.is_some();
            if ui
                .add_enabled(ready, egui::Button::new("Save current"))
                .clicked()
                && let Some(folder) = folder
            {
                if self.save_folder == FolderChoice::New {
                    self.save_folder = FolderChoice::Existing(folder.clone().unwrap_or_default());
                    self.new_folder.clear();
                }
                self.add(name, current(), folder);
                self.name.clear();
            }
        });
    }

    /// One bookmark: Go, delete, move, and its label.
    fn row(
        &mut self,
        ui: &mut egui::Ui,
        i: usize,
        names: &[String],
        go: &mut Option<BookmarkView>,
        action: &mut Option<RowAction>,
    ) {
        let b = &self.items[i];
        let kind = match &b.view {
            BookmarkView::Escape { kind, .. } => kind.label().to_string(),
            BookmarkView::Ifs { system, .. } => system.name.clone(),
        };
        let magnification = crate::engine::viewport::DEFAULT_SPAN / b.view.span();
        let label = format!("{}  ({kind})  ×{magnification:.1e}", b.name);
        let here = b.folder.clone();
        ui.horizontal(|ui| {
            if ui.button("Go").clicked() {
                *go = Some(self.items[i].view.clone());
            }
            if ui.button("🗑").on_hover_text("Delete").clicked() {
                *action = Some(RowAction::Delete(i));
            }
            ui.menu_button("📁", |ui| {
                if here.is_some() && ui.button("Unfiled").clicked() {
                    *action = Some(RowAction::Move(i, None));
                    ui.close();
                }
                for f in names {
                    if here.as_ref() != Some(f) && ui.button(f).clicked() {
                        *action = Some(RowAction::Move(i, Some(f.clone())));
                        ui.close();
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.new_folder);
                    let typed = self.typed_folder();
                    if ui
                        .add_enabled(typed.is_some(), egui::Button::new("New folder"))
                        .clicked()
                    {
                        *action = Some(RowAction::Move(i, typed));
                        self.new_folder.clear();
                        ui.close();
                    }
                });
            })
            .response
            .on_hover_text("Move to folder");
            ui.label(label);
        });
    }
}

enum RowAction {
    Delete(usize),
    Move(usize, Option<String>),
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
        ui.add("home".into(), view(), None);
        assert_eq!(ui.items.len(), 1);
        assert!(ui.load_error.is_none());
        assert!(ui.save_error.is_some());
    }

    fn temp_ui() -> (tempfile::TempDir, BookmarksUi) {
        let dir = tempfile::tempdir().unwrap();
        let ui = BookmarksUi::with_store(BookmarkStore::new(dir.path().join("b.json")));
        (dir, ui)
    }

    fn reloaded(dir: &tempfile::TempDir) -> Vec<Bookmark> {
        BookmarkStore::new(dir.path().join("b.json"))
            .load()
            .unwrap()
    }

    #[test]
    fn add_into_folder_persists() {
        let (dir, mut ui) = temp_ui();
        ui.add("home".into(), view(), Some("dive".into()));
        assert_eq!(reloaded(&dir)[0].folder.as_deref(), Some("dive"));
    }

    #[test]
    fn set_folder_moves_and_persists() {
        let (dir, mut ui) = temp_ui();
        ui.add("home".into(), view(), Some("dive".into()));
        ui.set_folder(0, Some("other".into()));
        assert_eq!(reloaded(&dir)[0].folder.as_deref(), Some("other"));
        ui.set_folder(0, None);
        assert_eq!(reloaded(&dir)[0].folder, None);
        assert!(folders(&ui.items).is_empty());
    }

    #[test]
    fn set_rotation_persists() {
        let (dir, mut ui) = temp_ui();
        ui.add("home".into(), view(), Some("dive".into()));
        ui.set_rotation(0, -12.5);
        assert_eq!(reloaded(&dir)[0].rotation, -12.5);
    }
}
