// TODO: Fix bsp module compilation errors (literal overflow issues)
// pub mod bsp;
pub mod webgl;

pub use webgl::{WebGLRenderer, DOOM_WIDTH, DOOM_HEIGHT, DOOM_FRAMEBUFFER_SIZE, PALETTE_SIZE};
// pub use bsp::BspRenderer;
