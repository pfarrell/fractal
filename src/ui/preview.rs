//! The movie preview player: one window of small frames, rendered in the
//! background, played and scrubbed inside the movie dialog.

use crate::engine::movie::MovieSpec;
use crate::engine::preview::{
    Fingerprint, PREVIEW_FPS, PreviewMsg, fingerprint, frame_at_time, preview_spec, runs_to_render,
    spawn_preview, window_at,
};
use crate::engine::render::CancelToken;
use eframe::egui;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, channel};

/// The two specs render the same frames (the output path doesn't count).
fn same_frames(a: &MovieSpec, b: &MovieSpec) -> bool {
    (
        a.kind, &a.stops, a.fps, a.seconds, a.width, a.height, a.max_iter, a.color, a.quality,
    ) == (
        b.kind, &b.stops, b.fps, b.seconds, b.width, b.height, b.max_iter, b.color, b.quality,
    )
}

pub struct Preview {
    /// The preview spec the frames were rendered for.
    spec: MovieSpec,
    /// The frames held: `window.start .. window.end` of `spec`.
    pub window: Range<usize>,
    /// Frame `window.start + k` at `[k]`, with the fingerprint it was
    /// rendered for.
    frames: Vec<Option<(Fingerprint, Arc<[u8]>)>>,
    /// The frame shown, as a frame index of `spec` (may lie outside
    /// `window` after scrubbing).
    pub playhead: usize,
    pub playing: bool,
    pub looping: bool,
    /// Fraction of a frame of play time not yet shown.
    clock: f64,
    generation: Arc<AtomicU64>,
    rx: Option<Receiver<PreviewMsg>>,
    /// Frames out of frames to render, for the current job.
    pub progress: (usize, usize),
    /// Why the last render stopped, if not finished or cancelled.
    pub error: Option<String>,
    /// The frame on screen and its texture.
    texture: Option<(usize, egui::TextureHandle)>,
    /// Which frame data the texture holds: (index, data address).
    shown: Option<(usize, usize)>,
}

/// What the player asks the dialog to do (it needs the current settings).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlayerAction {
    None,
    /// Preview the window after this one.
    NextWindow,
    /// Preview the window starting at the playhead.
    FromHere,
    /// Preview this window again with the current settings.
    Repreview,
    /// Drop the preview and its frames.
    Discard,
}

/// "m:ss".
fn clock(t: f64) -> String {
    let s = t.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

const SCRUB_HEIGHT: f32 = 22.0;
const DIM: egui::Color32 = egui::Color32::from_gray(45);
const WINDOW: egui::Color32 = egui::Color32::from_gray(80);
const RENDERED: egui::Color32 = egui::Color32::from_rgb(90, 140, 220);
const STALE: egui::Color32 = egui::Color32::from_rgb(230, 170, 60);

impl Preview {
    /// An empty preview of `final_spec`'s path (nothing rendered yet).
    pub fn new(final_spec: &MovieSpec) -> Self {
        Self {
            spec: preview_spec(final_spec),
            window: 0..0,
            frames: Vec::new(),
            playhead: 0,
            playing: false,
            looping: true,
            clock: 0.0,
            generation: Arc::new(AtomicU64::new(0)),
            rx: None,
            progress: (0, 0),
            error: None,
            texture: None,
            shown: None,
        }
    }

    pub fn spec(&self) -> &MovieSpec {
        &self.spec
    }

    /// Switch to `final_spec`'s preview and the window at `start_frame`,
    /// keeping held frames that land in the new window unchanged; cancels
    /// any running job. Returns the runs of frames still to render.
    pub fn prepare(&mut self, final_spec: &MovieSpec, start_frame: usize) -> Vec<Range<usize>> {
        self.cancel_render();
        let spec = preview_spec(final_spec);
        let window = window_at(&spec, start_frame);
        let mut frames: Vec<Option<(Fingerprint, Arc<[u8]>)>> = vec![None; window.len()];
        for (k, slot) in frames.iter_mut().enumerate() {
            let i = window.start + k;
            if let Some(Some((fp, rgb))) = i
                .checked_sub(self.window.start)
                .filter(|_| self.window.contains(&i))
                .map(|j| &self.frames[j])
                && *fp == fingerprint(&spec, i)
            {
                *slot = Some((*fp, rgb.clone()));
            }
        }
        if !window.contains(&self.playhead) {
            self.playhead = window.start;
        }
        self.spec = spec;
        self.window = window.clone();
        self.frames = frames;
        self.clock = 0.0;
        self.error = None;
        let runs = runs_to_render(window, |i| self.frame_rgb(i).is_some());
        self.progress = (0, runs.iter().map(|r| r.len()).sum());
        runs
    }

    /// [`Self::prepare`], then render the missing runs in the background.
    pub fn start(
        &mut self,
        final_spec: &MovieSpec,
        start_frame: usize,
        threads: usize,
        wake: impl Fn() + Send + 'static,
    ) {
        let runs = self.prepare(final_spec, start_frame);
        if runs.is_empty() {
            return;
        }
        let (tx, rx) = channel();
        let cancel = CancelToken::new(
            self.generation.clone(),
            self.generation.load(Ordering::Relaxed),
        );
        spawn_preview(self.spec.clone(), runs, threads, cancel, tx, wake);
        self.rx = Some(rx);
    }

    /// Take in what the job has sent (current generation only).
    pub fn receive(&mut self) {
        let Some(rx) = &self.rx else { return };
        let now = self.generation.load(Ordering::Relaxed);
        let mut finished = false;
        let msgs: Vec<PreviewMsg> = rx.try_iter().collect();
        for msg in msgs {
            match msg {
                PreviewMsg::Frame {
                    generation,
                    index,
                    rgb,
                } if generation == now && self.window.contains(&index) => {
                    let fp = fingerprint(&self.spec, index);
                    self.frames[index - self.window.start] = Some((fp, rgb));
                }
                PreviewMsg::Progress {
                    generation,
                    done,
                    total,
                } if generation == now => self.progress = (done, total),
                PreviewMsg::Finished { generation, result } if generation == now => {
                    finished = true;
                    if let Err(e) = result
                        && !matches!(e, crate::engine::movie::MovieError::Cancelled { .. })
                    {
                        self.error = Some(e.to_string());
                    }
                }
                _ => {}
            }
        }
        if finished {
            self.rx = None;
        }
    }

    /// Stop the running job, keeping its frames.
    pub fn cancel_render(&mut self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.rx = None;
    }

    pub fn is_rendering(&self) -> bool {
        self.rx.is_some()
    }

    /// The settings no longer describe what was previewed (`None`: they
    /// describe no movie at all).
    pub fn is_stale(&self, final_spec: Option<&MovieSpec>) -> bool {
        final_spec.is_none_or(|f| !same_frames(&preview_spec(f), &self.spec))
    }

    pub fn frame_rgb(&self, i: usize) -> Option<&Arc<[u8]>> {
        let k = i.checked_sub(self.window.start)?;
        self.frames.get(k)?.as_ref().map(|(_, rgb)| rgb)
    }

    /// Every frame of the window is here.
    pub fn complete(&self) -> bool {
        !self.frames.is_empty() && self.frames.iter().all(Option::is_some)
    }

    /// Advance play by `dt` seconds.
    pub fn tick(&mut self, dt: f64) {
        if !self.playing {
            return;
        }
        self.clock += dt * PREVIEW_FPS as f64;
        while self.clock >= 1.0 {
            let next = self.playhead + 1;
            if next >= self.window.end || next < self.window.start {
                if self.looping && self.complete() {
                    self.playhead = self.window.start;
                } else {
                    self.playing = false;
                    self.clock = 0.0;
                    return;
                }
            } else if self.frame_rgb(next).is_none() {
                // Rendering hasn't got here yet: hold the last frame.
                self.clock = 0.0;
                return;
            } else {
                self.playhead = next;
            }
            self.clock -= 1.0;
        }
    }

    /// Move the playhead by `frames`, within the movie.
    pub fn step(&mut self, frames: isize) {
        let last = self.spec.frame_count() as isize - 1;
        self.playhead = (self.playhead as isize + frames).clamp(0, last) as usize;
        self.clock = 0.0;
    }

    /// Put the playhead at movie time `t`.
    pub fn seek_time(&mut self, t: f64) {
        self.playhead = frame_at_time(&self.spec, t);
        self.clock = 0.0;
    }
}

impl Preview {
    /// The playhead's frame, if the texture doesn't hold it yet.
    fn frame_to_upload(&self) -> Option<Arc<[u8]>> {
        let rgb = self.frame_rgb(self.playhead)?;
        let key = (self.playhead, Arc::as_ptr(rgb) as *const u8 as usize);
        (self.shown != Some(key)).then(|| rgb.clone())
    }

    /// The texture now holds `rgb` for the playhead.
    fn mark_uploaded(&mut self, rgb: &Arc<[u8]>) {
        self.shown = Some((self.playhead, Arc::as_ptr(rgb) as *const u8 as usize));
    }

    /// Runs of rendered frames in the window.
    fn rendered_runs(&self) -> Vec<Range<usize>> {
        runs_to_render(self.window.clone(), |i| self.frame_rgb(i).is_none())
    }

    /// Movie time of the playhead.
    pub fn playhead_time(&self) -> f64 {
        self.spec.frame_time(self.playhead)
    }

    /// The player: picture, scrub bar, readout and controls. `current` is
    /// the movie the dialog's settings describe now (for the stale check);
    /// `can_start` says whether a new preview render may begin.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        current: Option<&MovieSpec>,
        can_start: bool,
    ) -> PlayerAction {
        self.receive();
        let mut action = PlayerAction::None;
        let ctx = ui.ctx().clone();
        if ctx.memory(|m| m.focused().is_none()) {
            let (space, left, right) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                )
            });
            if space {
                self.toggle_play();
            }
            if left {
                self.playing = false;
                self.step(-1);
            }
            if right {
                self.playing = false;
                self.step(1);
            }
        }
        self.tick(ui.input(|i| i.stable_dt) as f64);
        if self.playing || self.is_rendering() {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(1.0 / PREVIEW_FPS as f64));
        }

        self.picture(ui);
        self.scrub_bar(ui);
        self.readout(ui);
        ui.horizontal(|ui| {
            let label = if self.playing { "⏸" } else { "▶" };
            if ui.button(label).clicked() {
                self.toggle_play();
            }
            ui.checkbox(&mut self.looping, "loop");
            ui.add_enabled_ui(can_start && current.is_some(), |ui| {
                if self.window.end < self.spec.frame_count() && ui.button("Next 5 min").clicked() {
                    action = PlayerAction::NextWindow;
                }
                if ui.button("Preview from here").clicked() {
                    action = PlayerAction::FromHere;
                }
            });
            if ui.button("Discard preview").clicked() {
                action = PlayerAction::Discard;
            }
        });
        if self.is_rendering() {
            let (done, total) = self.progress;
            let frac = if total > 0 {
                done as f32 / total as f32
            } else {
                0.0
            };
            ui.add(egui::ProgressBar::new(frac).text(format!("previewing {done}/{total} frames")));
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 90, 90), e);
        }
        if self.is_stale(current) {
            ui.horizontal(|ui| {
                ui.colored_label(STALE, "settings changed — re-preview");
                if ui
                    .add_enabled(
                        can_start && current.is_some(),
                        egui::Button::new("Re-preview"),
                    )
                    .clicked()
                {
                    action = PlayerAction::Repreview;
                }
            });
        }
        action
    }

    fn toggle_play(&mut self) {
        if !self.playing
            && (!self.window.contains(&self.playhead) || self.playhead + 1 >= self.window.end)
        {
            self.playhead = self.window.start;
        }
        self.playing = !self.playing;
        self.clock = 0.0;
    }

    fn picture(&mut self, ui: &mut egui::Ui) {
        let (w, h) = (self.spec.width, self.spec.height);
        let width = ui.available_width();
        let size = egui::vec2(width, width * h as f32 / w as f32);
        if let Some(rgb) = self.frame_to_upload() {
            let image = egui::ColorImage::from_rgb([w, h], &rgb);
            match &mut self.texture {
                Some((i, t)) if t.size() == [w, h] => {
                    t.set(image, egui::TextureOptions::LINEAR);
                    *i = self.playhead;
                }
                _ => {
                    let t =
                        ui.ctx()
                            .load_texture("movie_preview", image, egui::TextureOptions::LINEAR);
                    self.texture = Some((self.playhead, t));
                }
            }
            self.mark_uploaded(&rgb);
        }
        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::BLACK);
        let shown = self.frame_rgb(self.playhead).is_some();
        if let Some((_, t)) = &self.texture {
            let tint = if shown {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_gray(70)
            };
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(t.id(), rect, uv, tint);
        }
        if !shown {
            let text = if self.window.contains(&self.playhead) {
                "rendering…"
            } else {
                "not previewed here — Preview from here"
            };
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            );
        }
    }

    fn scrub_bar(&mut self, ui: &mut egui::Ui) {
        let width = ui.available_width();
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(width, SCRUB_HEIGHT + 12.0),
            egui::Sense::click_and_drag(),
        );
        let bar = egui::Rect::from_min_size(rect.min, egui::vec2(width, SCRUB_HEIGHT));
        let n = self.spec.frame_count();
        let x_of = |i: f64| bar.left() + bar.width() * (i / (n - 1).max(1) as f64) as f32;
        let span_rect = |a: usize, b: usize| {
            egui::Rect::from_x_y_ranges(x_of(a as f64)..=x_of(b as f64), bar.y_range())
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(bar, 2.0, DIM);
        if !self.window.is_empty() {
            painter.rect_filled(
                span_rect(self.window.start, self.window.end - 1),
                0.0,
                WINDOW,
            );
        }
        for run in self.rendered_runs() {
            painter.rect_filled(span_rect(run.start, run.end - 1), 0.0, RENDERED);
        }
        let route = self.spec.route();
        for (k, t) in route.step_times().iter().enumerate() {
            let x = x_of(t / self.spec.seconds * (n - 1) as f64);
            painter.line_segment(
                [egui::pos2(x, bar.top()), egui::pos2(x, bar.bottom())],
                egui::Stroke::new(1.0, egui::Color32::WHITE),
            );
            painter.text(
                egui::pos2(x, bar.bottom() + 1.0),
                egui::Align2::CENTER_TOP,
                format!("{}", k + 1),
                egui::FontId::proportional(10.0),
                egui::Color32::LIGHT_GRAY,
            );
        }
        let x = x_of(self.playhead as f64);
        painter.line_segment(
            [
                egui::pos2(x, bar.top() - 2.0),
                egui::pos2(x, bar.bottom() + 2.0),
            ],
            egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 230, 120)),
        );
        if let Some(pos) = resp.interact_pointer_pos()
            && (resp.dragged() || resp.clicked())
        {
            let f = ((pos.x - bar.left()) / bar.width()).clamp(0.0, 1.0) as f64;
            self.seek_time(f * self.spec.seconds);
        }
    }

    fn readout(&self, ui: &mut egui::Ui) {
        let t = self.playhead_time();
        let times = self.spec.route().step_times().to_vec();
        let k = (0..times.len() - 1)
            .find(|&k| t < times[k + 1])
            .unwrap_or(times.len() - 2);
        ui.horizontal(|ui| {
            ui.monospace(format!("{} / {}", clock(t), clock(self.spec.seconds)));
            ui.separator();
            ui.label(format!("step {} → {}", k + 1, k + 2));
            ui.separator();
            ui.label(format!("turn {:+.0} °/s", self.spec.stops[k + 1].rotation));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::color::ColorSettings;
    use crate::engine::escape::FractalKind;
    use crate::engine::keyframe::Quality;
    use crate::engine::route::{JOINT_HALF_WIDTH_S, RotationUnit, Stop};
    use crate::engine::viewport::MaxIter;
    use num_complex::Complex64;
    use std::path::PathBuf;

    fn st(re: f64, im: f64, span: f64, rotation: f64) -> Stop {
        Stop {
            center: Complex64::new(re, im),
            span,
            rotation,
            rotation_unit: RotationUnit::PerSecond,
        }
    }

    fn final_spec(seconds: f64) -> MovieSpec {
        MovieSpec {
            kind: FractalKind::Mandelbrot,
            stops: vec![
                st(-0.5, 0.0, 3.0, 0.0),
                st(-0.75, 0.1, 0.03, 10.0),
                st(-0.745, 0.105, 3e-5, -5.0),
            ],
            fps: 30,
            seconds,
            width: 1280,
            height: 720,
            max_iter: MaxIter::Auto,
            color: ColorSettings::default(),
            output: PathBuf::from("/tmp/x.mp4"),
            quality: Quality::Smooth,
            music: Vec::new(),
        }
    }

    /// Put frame `i` in as if rendered.
    fn insert(p: &mut Preview, i: usize) {
        let fp = fingerprint(&p.spec, i);
        p.frames[i - p.window.start] = Some((fp, Arc::from(vec![0u8; 3])));
    }

    /// A preview of a 100-frame window with `have` rendered.
    fn with_frames(have: Range<usize>) -> Preview {
        let mut p = Preview::new(&final_spec(100.0 / 12.0));
        let runs = p.prepare(&final_spec(100.0 / 12.0), 0);
        assert_eq!(runs, vec![0..100]);
        for i in have {
            insert(&mut p, i);
        }
        p
    }

    #[test]
    fn tick_waits_at_the_rendered_edge() {
        let mut p = with_frames(0..10);
        p.playing = true;
        p.tick(5.0);
        assert_eq!(p.playhead, 9);
        assert!(p.playing);
        for i in 10..20 {
            insert(&mut p, i);
        }
        p.tick(0.5);
        assert_eq!(p.playhead, 15);
    }

    #[test]
    fn tick_loops_only_a_complete_window() {
        let mut p = with_frames(0..100);
        assert!(p.complete());
        (p.playing, p.looping, p.playhead) = (true, true, 99);
        p.tick(1.0 / 12.0);
        assert_eq!(p.playhead, 0);
        (p.looping, p.playhead) = (false, 99);
        p.tick(1.0 / 12.0);
        assert_eq!(p.playhead, 99);
        assert!(!p.playing);
    }

    #[test]
    fn late_frames_from_an_old_generation_are_ignored() {
        let mut p = with_frames(0..0);
        let (tx, rx) = channel();
        p.rx = Some(rx);
        let old = p.generation.load(Ordering::Relaxed).wrapping_sub(1);
        let now = p.generation.load(Ordering::Relaxed);
        for (generation, index) in [(old, 3), (now, 4)] {
            tx.send(PreviewMsg::Frame {
                generation,
                index,
                rgb: Arc::from(vec![1u8; 3]),
            })
            .unwrap();
        }
        p.receive();
        assert!(p.frame_rgb(3).is_none());
        assert!(p.frame_rgb(4).is_some());
    }

    #[test]
    fn restart_keeps_matching_frames() {
        let f = final_spec(20.0);
        let mut p = Preview::new(&f);
        p.prepare(&f, 0);
        for i in p.window.clone() {
            insert(&mut p, i);
        }
        let mut g = f.clone();
        g.stops[2].rotation = 25.0;
        let runs = p.prepare(&g, 0);
        let joint = p.spec.route().step_times()[1];
        let first_changed = runs[0].start;
        assert!(p.spec.frame_time(first_changed) >= joint - JOINT_HALF_WIDTH_S);
        for i in 0..first_changed {
            assert!(p.frame_rgb(i).is_some(), "frame {i} dropped");
        }
        assert_eq!(runs.last().unwrap().end, p.window.end);
    }

    #[test]
    fn stale_when_settings_change_or_spec_invalid() {
        let f = final_spec(20.0);
        let mut p = Preview::new(&f);
        p.prepare(&f, 0);
        assert!(!p.is_stale(Some(&f)));
        let moved = MovieSpec {
            output: PathBuf::from("/elsewhere.mp4"),
            ..f.clone()
        };
        assert!(!p.is_stale(Some(&moved)));
        assert!(p.is_stale(Some(&final_spec(21.0))));
        assert!(p.is_stale(None));
    }

    #[test]
    fn a_rerendered_frame_at_the_playhead_is_uploaded_again() {
        let f = final_spec(100.0 / 12.0);
        let mut p = Preview::new(&f);
        p.prepare(&f, 0);
        insert(&mut p, 5);
        p.playhead = 5;
        let first = p.frame_to_upload().expect("new frame");
        p.mark_uploaded(&first);
        assert!(p.frame_to_upload().is_none());
        // Re-preview with a changed turn: frame 5 comes back re-rendered.
        let mut g = f.clone();
        g.stops[1].rotation = 0.0;
        p.prepare(&g, 0);
        insert(&mut p, 5);
        assert!(p.frame_to_upload().is_some(), "stale picture kept");
    }

    #[test]
    fn rendered_runs_stay_inside_the_window() {
        let mut p = with_frames(0..0);
        for i in [3, 4, 5, 9, 98, 99] {
            insert(&mut p, i);
        }
        assert_eq!(p.rendered_runs(), vec![3..6, 9..10, 98..100]);
    }

    #[test]
    fn seek_and_step_stay_in_the_movie() {
        let mut p = with_frames(0..100);
        p.seek_time(1e9);
        assert_eq!(p.playhead, p.spec.frame_count() - 1);
        p.step(-1000);
        assert_eq!(p.playhead, 0);
        p.step(3);
        assert_eq!(p.playhead, 3);
    }
}
