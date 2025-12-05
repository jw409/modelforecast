use wasm_bindgen::JsCast;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader,
    WebGlTexture, WebGlUniformLocation,
};

/// DOOM framebuffer dimensions
pub const DOOM_WIDTH: usize = 320;
pub const DOOM_HEIGHT: usize = 200;
pub const DOOM_FRAMEBUFFER_SIZE: usize = DOOM_WIDTH * DOOM_HEIGHT;

/// Palette size (256 colors * 3 RGB bytes)
pub const PALETTE_SIZE: usize = 256 * 3;

/// WebGL2 renderer for DOOM
pub struct WebGLRenderer {
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    vbo: WebGlBuffer,
    framebuffer_texture: WebGlTexture,
    palette_texture: WebGlTexture,
    u_framebuffer_loc: WebGlUniformLocation,
    u_palette_loc: WebGlUniformLocation,
    enable_crt: bool,
}

impl WebGLRenderer {
    /// Initialize WebGL2 renderer from canvas element
    pub fn new(canvas: &HtmlCanvasElement, enable_crt: bool) -> Result<Self, String> {
        // Get WebGL2 context
        let gl = canvas
            .get_context("webgl2")
            .map_err(|e| format!("Failed to get context: {:?}", e))?
            .ok_or("WebGL2 not supported")?
            .dyn_into::<WebGl2RenderingContext>()
            .map_err(|e| format!("Failed to cast context: {:?}", e))?;

        // Create shader program
        let vert_shader = Self::compile_shader(
            &gl,
            WebGl2RenderingContext::VERTEX_SHADER,
            VERTEX_SHADER_SOURCE,
        )?;

        let frag_shader_source = if enable_crt {
            FRAGMENT_SHADER_CRT_SOURCE
        } else {
            FRAGMENT_SHADER_SOURCE
        };

        let frag_shader = Self::compile_shader(
            &gl,
            WebGl2RenderingContext::FRAGMENT_SHADER,
            frag_shader_source,
        )?;

        let program = Self::link_program(&gl, &vert_shader, &frag_shader)?;

        gl.use_program(Some(&program));

        // Create fullscreen quad vertex buffer
        let vbo = gl
            .create_buffer()
            .ok_or("Failed to create vertex buffer")?;

        #[rustfmt::skip]
        let vertices: [f32; 12] = [
            // Triangle 1
            -1.0,  1.0,  // top-left
            -1.0, -1.0,  // bottom-left
             1.0, -1.0,  // bottom-right
            // Triangle 2
            -1.0,  1.0,  // top-left
             1.0, -1.0,  // bottom-right
             1.0,  1.0,  // top-right
        ];

        gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vbo));

        // Upload vertex data
        unsafe {
            let vertex_array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &vertex_array,
                WebGl2RenderingContext::STATIC_DRAW,
            );
        }

        // Set up vertex attribute
        let a_pos_loc = gl.get_attrib_location(&program, "a_pos") as u32;
        gl.vertex_attrib_pointer_with_i32(a_pos_loc, 2, WebGl2RenderingContext::FLOAT, false, 0, 0);
        gl.enable_vertex_attrib_array(a_pos_loc);

        // Create framebuffer texture (320x200 R8)
        let framebuffer_texture = gl.create_texture().ok_or("Failed to create framebuffer texture")?;
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&framebuffer_texture));

        // Set texture parameters
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );

        // Allocate texture storage
        gl.tex_storage_2d(
            WebGl2RenderingContext::TEXTURE_2D,
            1,
            WebGl2RenderingContext::R8,
            DOOM_WIDTH as i32,
            DOOM_HEIGHT as i32,
        );

        // Create palette texture (256x1 RGB)
        let palette_texture = gl.create_texture().ok_or("Failed to create palette texture")?;
        gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&palette_texture));

        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );

        gl.tex_storage_2d(
            WebGl2RenderingContext::TEXTURE_2D,
            1,
            WebGl2RenderingContext::RGB8,
            256,
            1,
        );

        // Get uniform locations
        let u_framebuffer_loc = gl
            .get_uniform_location(&program, "u_framebuffer")
            .ok_or("Failed to get u_framebuffer uniform location")?;

        let u_palette_loc = gl
            .get_uniform_location(&program, "u_palette")
            .ok_or("Failed to get u_palette uniform location")?;

        // Set texture units
        gl.uniform1i(Some(&u_framebuffer_loc), 0);
        gl.uniform1i(Some(&u_palette_loc), 1);

        // Set viewport
        gl.viewport(0, 0, canvas.width() as i32, canvas.height() as i32);

        // Set clear color
        gl.clear_color(0.0, 0.0, 0.0, 1.0);

        Ok(Self {
            gl,
            program,
            vbo,
            framebuffer_texture,
            palette_texture,
            u_framebuffer_loc,
            u_palette_loc,
            enable_crt,
        })
    }

    /// Upload framebuffer data (320x200 8-bit palette indices)
    pub fn upload_framebuffer(&self, data: &[u8]) {
        assert_eq!(data.len(), DOOM_FRAMEBUFFER_SIZE, "Invalid framebuffer size");

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.framebuffer_texture),
        );

        self.gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                0,
                0,
                DOOM_WIDTH as i32,
                DOOM_HEIGHT as i32,
                WebGl2RenderingContext::RED,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(data),
            )
            .expect("Failed to upload framebuffer data");
    }

    /// Upload palette data (256 RGB colors, 768 bytes total)
    pub fn upload_palette(&self, data: &[u8]) {
        assert_eq!(data.len(), PALETTE_SIZE, "Invalid palette size");

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.palette_texture),
        );

        self.gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                0,
                0,
                256,
                1,
                WebGl2RenderingContext::RGB,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(data),
            )
            .expect("Failed to upload palette data");
    }

    /// Render the frame
    pub fn render(&self) {
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

        self.gl.use_program(Some(&self.program));

        // Bind textures
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.framebuffer_texture),
        );

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.palette_texture),
        );

        // Bind vertex buffer
        self.gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&self.vbo));

        // Draw fullscreen quad
        self.gl.draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, 6);
    }

    /// Resize viewport
    pub fn resize(&self, width: u32, height: u32) {
        self.gl.viewport(0, 0, width as i32, height as i32);
    }

    /// Compile shader
    fn compile_shader(
        gl: &WebGl2RenderingContext,
        shader_type: u32,
        source: &str,
    ) -> Result<WebGlShader, String> {
        let shader = gl
            .create_shader(shader_type)
            .ok_or("Failed to create shader")?;

        gl.shader_source(&shader, source);
        gl.compile_shader(&shader);

        if gl
            .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            Ok(shader)
        } else {
            let log = gl
                .get_shader_info_log(&shader)
                .unwrap_or_else(|| "Unknown error".to_string());
            Err(format!("Shader compilation failed: {}", log))
        }
    }

    /// Link shader program
    fn link_program(
        gl: &WebGl2RenderingContext,
        vert_shader: &WebGlShader,
        frag_shader: &WebGlShader,
    ) -> Result<WebGlProgram, String> {
        let program = gl.create_program().ok_or("Failed to create program")?;

        gl.attach_shader(&program, vert_shader);
        gl.attach_shader(&program, frag_shader);
        gl.link_program(&program);

        if gl
            .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            Ok(program)
        } else {
            let log = gl
                .get_program_info_log(&program)
                .unwrap_or_else(|| "Unknown error".to_string());
            Err(format!("Program linking failed: {}", log))
        }
    }
}

/// Vertex shader source
const VERTEX_SHADER_SOURCE: &str = r#"#version 300 es

in vec2 a_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_pos * 0.5 + 0.5;
}
"#;

/// Fragment shader source (basic)
const FRAGMENT_SHADER_SOURCE: &str = r#"#version 300 es
precision mediump float;

uniform sampler2D u_framebuffer;
uniform sampler2D u_palette;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    // Flip Y coordinate (DOOM renders top-down, WebGL is bottom-up)
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    // Look up palette index
    float idx = texture(u_framebuffer, uv).r;

    // Look up color in palette
    vec3 color = texture(u_palette, vec2(idx, 0.5)).rgb;

    fragColor = vec4(color, 1.0);
}
"#;

/// Fragment shader source with CRT effect
const FRAGMENT_SHADER_CRT_SOURCE: &str = r#"#version 300 es
precision mediump float;

uniform sampler2D u_framebuffer;
uniform sampler2D u_palette;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    // CRT barrel distortion
    vec2 uv = v_uv - 0.5;
    float dist = length(uv);
    float distortion = 0.15;
    uv *= 1.0 + distortion * dist * dist;
    uv += 0.5;

    // Discard pixels outside viewport
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Flip Y coordinate
    uv.y = 1.0 - uv.y;

    // Look up palette index
    float idx = texture(u_framebuffer, uv).r;

    // Look up color in palette
    vec3 color = texture(u_palette, vec2(idx, 0.5)).rgb;

    // Scanline effect
    float scanline = sin(uv.y * float(200) * 3.14159 * 2.0) * 0.04;
    color -= scanline;

    // Vignette effect
    float vignette = smoothstep(0.7, 0.3, dist);
    color *= vignette;

    fragColor = vec4(color, 1.0);
}
"#;
