use fractal_explorer::engine::escape::FractalKind;
use fractal_explorer::engine::jobs::{RenderOutput, RenderRequest, RenderSlot};
use fractal_explorer::engine::render::{F64Sampler, render_serial};
use fractal_explorer::engine::viewport::Viewport;
use num_complex::Complex64;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn counting_slot(threads: usize) -> (RenderSlot, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    (
        RenderSlot::new(
            threads,
            Arc::new(move || {
                c.fetch_add(1, Ordering::SeqCst);
            }),
        ),
        count,
    )
}

fn wait_for(count: &AtomicUsize, n: usize) {
    let start = Instant::now();
    while count.load(Ordering::SeqCst) < n {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "timed out waiting for {n} notifications"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn request(view: Viewport, max_iter: u32, progressive: bool) -> RenderRequest {
    RenderRequest {
        sampler: Arc::new(F64Sampler {
            kind: FractalKind::Mandelbrot,
            max_iter,
        }),
        view,
        progressive,
    }
}

#[test]
fn submit_then_poll_gets_final() {
    let (slot, count) = counting_slot(4);
    let view = Viewport::default_for(200, 100);
    let generation = slot.submit(request(view, 200, false));
    wait_for(&count, 1);
    let out: RenderOutput = slot.poll().expect("output");
    assert!(out.final_pass);
    assert_eq!(out.generation, generation);
    assert_eq!(out.threads, 4);
    let sampler = F64Sampler {
        kind: FractalKind::Mandelbrot,
        max_iter: 200,
    };
    assert!(out.samples == render_serial(&sampler, &view));
}

#[test]
fn progressive_sends_preview_first() {
    let (slot, count) = counting_slot(4);
    slot.submit(request(Viewport::default_for(400, 256), 200, true));
    wait_for(&count, 2);
    let outs = slot.poll_all();
    assert_eq!(outs.len(), 2);
    assert!(!outs[0].final_pass);
    assert_eq!((outs[0].view.width, outs[0].view.height), (100, 64));
    assert!(outs[1].final_pass);
    assert_eq!((outs[1].view.width, outs[1].view.height), (400, 256));
}

#[test]
fn newer_submit_wins() {
    let (slot, count) = counting_slot(4);
    let slow = Viewport::new(Complex64::new(0.0, 0.0), 0.1, 640, 480);
    slot.submit(request(slow, 20_000, false));
    let b = slot.submit(request(Viewport::default_for(64, 64), 100, false));
    wait_for(&count, 1);
    std::thread::sleep(Duration::from_millis(300));
    let outs = slot.poll_all();
    assert!(!outs.is_empty());
    assert!(outs.iter().all(|o| o.generation == b));
}

#[test]
fn set_threads_changes_count() {
    let (mut slot, _) = counting_slot(4);
    assert_eq!(slot.threads(), 4);
    slot.set_threads(2);
    assert_eq!(slot.threads(), 2);
}
