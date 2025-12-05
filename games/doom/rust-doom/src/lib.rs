// DOOM in Rust for WebAssembly
// Pure Rust + WebGL2 implementation - NO EMSCRIPTEN

pub mod fixed;
pub mod input;
pub mod renderer;
pub mod wad;

use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, console};

use crate::input::InputManager;
use crate::renderer::webgl::WebGLRenderer;
use crate::wad::Wad;

// Set panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Main DOOM game state
#[wasm_bindgen]
pub struct DoomGame {
    wad: Option<Wad>,
    renderer: Option<WebGLRenderer>,
    input: Option<InputManager>,
    framebuffer: Vec<u8>,
    palette: Vec<u8>,
    running: bool,
}

#[wasm_bindgen]
impl DoomGame {
    /// Create a new DOOM game instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> DoomGame {
        console::log_1(&"DOOM Rust/WASM initializing...".into());
        DoomGame {
            wad: None,
            renderer: None,
            input: None,
            framebuffer: vec![0u8; 320 * 200],
            palette: vec![0u8; 768],
            running: false,
        }
    }

    /// Initialize the renderer with a canvas element
    #[wasm_bindgen]
    pub fn init_renderer(&mut self, canvas: HtmlCanvasElement, crt_effect: bool) -> Result<(), JsValue> {
        console::log_1(&"Initializing WebGL2 renderer...".into());
        self.renderer = Some(WebGLRenderer::new(&canvas, crt_effect)?);
        self.input = Some(InputManager::new(canvas)?);
        console::log_1(&"Renderer initialized!".into());
        Ok(())
    }

    /// Load a WAD file from bytes
    #[wasm_bindgen]
    pub fn load_wad(&mut self, data: Vec<u8>) -> Result<(), JsValue> {
        console::log_1(&format!("Loading WAD ({} bytes)...", data.len()).into());

        match Wad::from_bytes(data) {
            Ok(wad) => {
                // Load palette (PLAYPAL lump)
                if let Some(playpal) = wad.get_lump("PLAYPAL") {
                    if playpal.len() >= 768 {
                        self.palette.copy_from_slice(&playpal[0..768]);
                        if let Some(ref renderer) = self.renderer {
                            renderer.upload_palette(&self.palette);
                        }
                        console::log_1(&"Palette loaded!".into());
                    }
                }

                console::log_1(&format!("WAD loaded: {} lumps", wad.num_lumps()).into());
                self.wad = Some(wad);
                Ok(())
            }
            Err(e) => Err(JsValue::from_str(&format!("WAD error: {:?}", e)))
        }
    }

    /// Start the game
    #[wasm_bindgen]
    pub fn start(&mut self) {
        self.running = true;
        console::log_1(&"DOOM started!".into());
    }

    /// Stop the game
    #[wasm_bindgen]
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Run one frame of the game loop
    #[wasm_bindgen]
    pub fn frame(&mut self) {
        if !self.running {
            return;
        }

        // Process input
        if let Some(ref input) = self.input {
            let events = input.poll_events();
            for event in events {
                // TODO: Handle input events
                match event {
                    input::InputEvent::KeyDown(key) => {
                        console::log_1(&format!("Key down: {}", key).into());
                    }
                    _ => {}
                }
            }
        }

        // TODO: Run game logic (P_Ticker, etc.)

        // TODO: Render frame (R_RenderPlayerView)
        // For now, just show a test pattern
        self.draw_test_pattern();

        // Upload and display
        if let Some(ref renderer) = self.renderer {
            renderer.upload_framebuffer(&self.framebuffer);
            renderer.render();
        }
    }

    /// Draw a test pattern to verify rendering works
    fn draw_test_pattern(&mut self) {
        static mut FRAME: u32 = 0;
        unsafe { FRAME = FRAME.wrapping_add(1); }

        for y in 0..200 {
            for x in 0..320 {
                let idx = y * 320 + x;
                // Animated color bars
                unsafe {
                    self.framebuffer[idx] = ((x + FRAME as usize) % 256) as u8;
                }
            }
        }
    }

    /// Warp to a specific level
    #[wasm_bindgen]
    pub fn warp(&mut self, episode: u32, map: u32) {
        console::log_1(&format!("Warping to E{}M{}...", episode, map).into());
        // TODO: Load level from WAD
    }

    /// Toggle god mode
    #[wasm_bindgen]
    pub fn god_mode(&mut self) {
        console::log_1(&"God mode toggled!".into());
        // TODO: Implement
    }

    /// Give all weapons and ammo
    #[wasm_bindgen]
    pub fn give_all(&mut self) {
        console::log_1(&"IDKFA!".into());
        // TODO: Implement
    }

    /// Toggle no-clip
    #[wasm_bindgen]
    pub fn noclip(&mut self) {
        console::log_1(&"No-clip toggled!".into());
        // TODO: Implement
    }
}

impl Default for DoomGame {
    fn default() -> Self {
        Self::new()
    }
}
