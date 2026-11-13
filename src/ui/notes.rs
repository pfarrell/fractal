//! Short "what am I looking at?" explanations.

use crate::engine::escape::FractalKind;
use crate::engine::newton::NewtonPoly;
use eframe::egui;

/// The note for a fractal type. Exhaustive on purpose: adding a kind
/// without a note won't compile.
pub fn note_for_kind(kind: &FractalKind) -> &'static str {
    match kind {
        FractalKind::Mandelbrot => {
            "Each pixel is a number c. Start at z = 0 and repeat z → z² + c. \
             If z stays small forever, c is in the Mandelbrot set (black). \
             Otherwise the color says how quickly z flew off to infinity."
        }
        FractalKind::Julia { .. } => {
            "Here c is fixed and each pixel is a starting point z. Repeat z → z² + c \
             and color by how fast z escapes. The Julia set is in one connected \
             piece exactly when c lies inside the Mandelbrot set; otherwise it \
             shatters into dust."
        }
        FractalKind::BurningShip => {
            "Like the Mandelbrot set, but the real and imaginary parts are made \
             positive before squaring: z → (|Re z| + i|Im z|)² + c. That one \
             absolute value breaks the mirror symmetry and folds the shape into \
             the 'ship' with its burning masts."
        }
        FractalKind::Tricorn => {
            "Like the Mandelbrot set, but z is flipped (conjugated) before \
             squaring: z → conj(z)² + c. The flip turns the twofold symmetry \
             into threefold, giving three 'horns'."
        }
        FractalKind::Multibrot { .. } => {
            "The Mandelbrot recipe with a higher power: z → zᵈ + c. A power of d \
             gives d − 1 identical lobes arranged in a ring: try d = 3, 4, 5."
        }
        FractalKind::Newton { poly } => match poly {
            NewtonPoly::CycleCubic => {
                "Newton's method for z³ − 2z + 2. Each color is the root Newton's \
                 method reaches from that starting point. The black regions are \
                 where it never converges: it gets trapped bouncing 0 → 1 → 0 forever."
            }
            _ => {
                "Newton's method (z → z − p(z)/p'(z)) homes in on a root of the \
                 polynomial. Each color is the root reached from that starting \
                 point; darker means it took longer. Every boundary point touches \
                 all the basins at once, which forces the infinitely detailed edges."
            }
        },
    }
}

pub const NOTE_JULIA_LINK: &str = "Every point c of the Mandelbrot set has its own Julia set. \
    Move the cursor inside the black region and the Julia set stays in one piece; \
    move outside and it breaks into dust. Near the boundary, the Julia set looks \
    like the Mandelbrot set right where you are pointing.";

pub const NOTE_ORBIT: &str = "The dots are z₀, z₁, z₂, …: the path one point takes \
    as the formula is repeated (red is the start). The pixel's color is just a \
    summary of this path: how many steps until it escaped, if it ever did.";

pub const NOTE_IFS: &str = "The picture is the one shape that equals the union of \
    its shrunken copies, one per row of the table (the colored outlines). The \
    chaos game finds it by starting anywhere and jumping with randomly chosen \
    maps; after a few jumps every point lands on the shape.";

/// A collapsed "What am I looking at?" section.
pub fn note(ui: &mut egui::Ui, id: &str, text: &str) {
    egui::CollapsingHeader::new("What am I looking at?")
        .id_salt(id)
        .default_open(false)
        .show(ui, |ui| {
            ui.label(text);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::escape::FractalKind;
    use crate::engine::newton::NewtonPoly;

    #[test]
    fn every_kind_has_note() {
        let mut kinds = FractalKind::gallery();
        kinds.extend(NewtonPoly::ALL.map(|poly| FractalKind::Newton { poly }));
        for k in kinds {
            assert!(note_for_kind(&k).len() > 40, "{k:?}");
        }
        for n in [NOTE_JULIA_LINK, NOTE_ORBIT, NOTE_IFS] {
            assert!(n.len() > 40);
        }
    }
}
