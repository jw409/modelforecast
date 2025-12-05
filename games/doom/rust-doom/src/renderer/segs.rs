// DOOM Seg Renderer - Wall segment rendering module
// Ported from r_segs.c (linuxdoom-1.10)
//
// This module handles rendering of wall segments (segs) to the framebuffer.
// Segs are parts of linedefs that are visible from the current subsector.
// Each seg is rendered column-by-column with texture mapping and lighting.

use std::cmp::{max, min};

// Constants for rendering
const SCREEN_WIDTH: usize = 320;
const SCREEN_HEIGHT: usize = 200;
const HEIGHTBITS: i32 = 12;
const HEIGHTUNIT: i32 = 1 << HEIGHTBITS;

const LIGHTLEVELS: usize = 16;
const LIGHTSEGSHIFT: i32 = 4;
const MAXLIGHTSCALE: usize = 48;
const LIGHTSCALESHIFT: i32 = 12;

const ANGLETOFINESHIFT: i32 = 19;
const FINEANGLES: usize = 8192;

// Silhouette flags
const SIL_NONE: u8 = 0;
const SIL_BOTTOM: u8 = 1;
const SIL_TOP: u8 = 2;
const SIL_BOTH: u8 = 3;

const MAXDRAWSEGS: usize = 256;

// Fixed-point types (16.16 format)
pub type Fixed = i32;

// Angle type (BAM - Binary Angle Measurement)
pub type Angle = u32;

const ANG90: Angle = 0x40000000;
const ANG180: Angle = 0x80000000;

// Fixed-point math helpers
#[inline]
fn fixed_mul(a: Fixed, b: Fixed) -> Fixed {
    ((a as i64 * b as i64) >> 16) as i32
}

#[inline]
fn fixed_div(a: Fixed, b: Fixed) -> Fixed {
    if b == 0 {
        if a < 0 { i32::MIN } else { i32::MAX }
    } else {
        (((a as i64) << 16) / (b as i64)) as i32
    }
}

const FRACUNIT: Fixed = 1 << 16;
const FRACBITS: i32 = 16;

/// Vertex in 2D space
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub x: Fixed,
    pub y: Fixed,
}

/// Sector (room) data
#[derive(Clone, Debug)]
pub struct Sector {
    pub floor_height: Fixed,
    pub ceiling_height: Fixed,
    pub floor_pic: i16,
    pub ceiling_pic: i16,
    pub light_level: i16,
}

/// Sidedef - texture info for one side of a linedef
#[derive(Clone, Debug)]
pub struct Sidedef {
    pub texture_offset: Fixed,
    pub row_offset: Fixed,
    pub top_texture: i16,
    pub bottom_texture: i16,
    pub mid_texture: i16,
}

/// Linedef flags
const ML_DONTPEGBOTTOM: i16 = 0x0008;
const ML_DONTPEGTOP: i16 = 0x0010;
const ML_MAPPED: i16 = 0x0080;

/// Line definition
#[derive(Clone, Debug)]
pub struct Linedef {
    pub flags: i16,
}

/// Seg - a portion of a linedef visible from current subsector
#[derive(Clone, Debug)]
pub struct Seg {
    pub v1: Vertex,
    pub v2: Vertex,
    pub angle: Angle,
    pub offset: Fixed,
    pub sidedef: Sidedef,
    pub linedef: Linedef,
    pub front_sector: Sector,
    pub back_sector: Option<Sector>,
}

/// Draw segment - bookkeeping for sprite clipping
#[derive(Clone, Debug)]
pub struct DrawSeg {
    pub x1: i32,
    pub x2: i32,
    pub scale1: Fixed,
    pub scale2: Fixed,
    pub scale_step: Fixed,
    pub silhouette: u8,
    pub spr_top_clip: Vec<i16>,
    pub spr_bottom_clip: Vec<i16>,
    pub masked_texture_col: Vec<i16>,
}

/// Main seg renderer state
pub struct SegRenderer {
    // Global state
    view_x: Fixed,
    view_y: Fixed,
    view_z: Fixed,
    view_angle: Angle,
    view_cos: Fixed,
    view_sin: Fixed,

    centerx: i32,
    centery: i32,
    centerx_frac: Fixed,
    centery_frac: Fixed,
    projection: Fixed,

    // Clipping arrays (track floor/ceiling bounds per column)
    ceiling_clip: [i16; SCREEN_WIDTH],
    floor_clip: [i16; SCREEN_WIDTH],

    // Current seg rendering state
    rw_x: i32,
    rw_stopx: i32,
    rw_centerangle: Angle,
    rw_offset: Fixed,
    rw_distance: Fixed,
    rw_scale: Fixed,
    rw_scalestep: Fixed,
    rw_midtexturemid: Fixed,
    rw_toptexturemid: Fixed,
    rw_bottomtexturemid: Fixed,

    world_top: i32,
    world_bottom: i32,
    world_high: i32,
    world_low: i32,

    pixhigh: Fixed,
    pixlow: Fixed,
    pixhighstep: Fixed,
    pixlowstep: Fixed,

    topfrac: Fixed,
    topstep: Fixed,
    bottomfrac: Fixed,
    bottomstep: Fixed,

    // Current seg info
    segtextured: bool,
    markfloor: bool,
    markceiling: bool,
    maskedtexture: bool,

    toptexture: i16,
    bottomtexture: i16,
    midtexture: i16,

    // Lighting
    extra_light: i32,

    // Textures (placeholder - would be actual texture data)
    texture_heights: Vec<i32>,

    // Fine trig tables
    fine_sine: Vec<Fixed>,
    fine_tangent: Vec<Fixed>,

    // Angle-to-X mapping
    xtoviewangle: Vec<Angle>,

    // Draw segments
    draw_segs: Vec<DrawSeg>,

    // Sky flat number
    sky_flat_num: i16,
}

impl SegRenderer {
    pub fn new() -> Self {
        let mut renderer = SegRenderer {
            view_x: 0,
            view_y: 0,
            view_z: 41 * FRACUNIT,
            view_angle: 0,
            view_cos: FRACUNIT,
            view_sin: 0,

            centerx: (SCREEN_WIDTH / 2) as i32,
            centery: (SCREEN_HEIGHT / 2) as i32,
            centerx_frac: ((SCREEN_WIDTH / 2) << FRACBITS) as Fixed,
            centery_frac: ((SCREEN_HEIGHT / 2) << FRACBITS) as Fixed,
            projection: (SCREEN_WIDTH / 2) as Fixed * FRACUNIT,

            ceiling_clip: [0; SCREEN_WIDTH],
            floor_clip: [SCREEN_HEIGHT as i16 - 1; SCREEN_WIDTH],

            rw_x: 0,
            rw_stopx: 0,
            rw_centerangle: 0,
            rw_offset: 0,
            rw_distance: 0,
            rw_scale: 0,
            rw_scalestep: 0,
            rw_midtexturemid: 0,
            rw_toptexturemid: 0,
            rw_bottomtexturemid: 0,

            world_top: 0,
            world_bottom: 0,
            world_high: 0,
            world_low: 0,

            pixhigh: 0,
            pixlow: 0,
            pixhighstep: 0,
            pixlowstep: 0,

            topfrac: 0,
            topstep: 0,
            bottomfrac: 0,
            bottomstep: 0,

            segtextured: false,
            markfloor: false,
            markceiling: false,
            maskedtexture: false,

            toptexture: 0,
            bottomtexture: 0,
            midtexture: 0,

            extra_light: 0,

            texture_heights: vec![128; 256],
            fine_sine: vec![0; FINEANGLES],
            fine_tangent: vec![0; FINEANGLES],
            xtoviewangle: vec![0; SCREEN_WIDTH],

            draw_segs: Vec::with_capacity(MAXDRAWSEGS),

            sky_flat_num: -1,
        };

        // Initialize trig tables (simplified)
        renderer.init_tables();

        renderer
    }

    fn init_tables(&mut self) {
        use std::f64::consts::PI;

        // Initialize sine/tangent tables
        for i in 0..FINEANGLES {
            let angle = (i as f64) * 2.0 * PI / (FINEANGLES as f64);
            self.fine_sine[i] = (angle.sin() * (FRACUNIT as f64)) as Fixed;
            self.fine_tangent[i] = (angle.tan() * (FRACUNIT as f64)) as Fixed;
        }

        // Initialize angle-to-X mapping (simplified)
        for i in 0..SCREEN_WIDTH {
            let tan = ((i as i32 - self.centerx) << FRACBITS) / self.projection;
            self.xtoviewangle[i] = (tan * 10) as Angle; // Simplified
        }
    }

    /// Calculate distance from point to viewpoint
    fn point_to_dist(&self, x: Fixed, y: Fixed) -> Fixed {
        let dx = x - self.view_x;
        let dy = y - self.view_y;

        let dx_abs = dx.abs() as i64;
        let dy_abs = dy.abs() as i64;

        // Approximate distance
        let dist = ((dx_abs * dx_abs + dy_abs * dy_abs) as f64).sqrt() as Fixed;
        max(dist, 1)
    }

    /// Calculate scale from global angle
    fn scale_from_global_angle(&self, angle: Angle) -> Fixed {
        let angle_diff = angle.wrapping_sub(self.view_angle);
        let fineangle = (angle_diff >> ANGLETOFINESHIFT) & (FINEANGLES as u32 - 1);

        let cosine = self.fine_sine[((fineangle + FINEANGLES as u32 / 4) & (FINEANGLES as u32 - 1)) as usize];

        if cosine == 0 {
            return i32::MAX;
        }

        fixed_div(self.projection, fixed_mul(self.rw_distance, cosine))
    }

    /// Main seg rendering loop - draws wall textures column by column
    fn render_seg_loop(&mut self, framebuffer: &mut [u8; SCREEN_WIDTH * SCREEN_HEIGHT]) {
        while self.rw_x < self.rw_stopx {
            // Calculate top of wall in screen space
            let mut yl = ((self.topfrac + HEIGHTUNIT - 1) >> HEIGHTBITS) as i16;

            // Clip to ceiling
            if yl < self.ceiling_clip[self.rw_x as usize] + 1 {
                yl = self.ceiling_clip[self.rw_x as usize] + 1;
            }

            // Mark ceiling area if needed
            if self.markceiling {
                let top = self.ceiling_clip[self.rw_x as usize] + 1;
                let bottom = yl - 1;

                if bottom >= self.floor_clip[self.rw_x as usize] {
                    // Would mark ceiling plane here
                }
            }

            // Calculate bottom of wall in screen space
            let mut yh = (self.bottomfrac >> HEIGHTBITS) as i16;

            // Clip to floor
            if yh >= self.floor_clip[self.rw_x as usize] {
                yh = self.floor_clip[self.rw_x as usize] - 1;
            }

            // Mark floor area if needed
            if self.markfloor {
                let top = yh + 1;
                let bottom = self.floor_clip[self.rw_x as usize] - 1;

                if top <= self.ceiling_clip[self.rw_x as usize] {
                    // Would mark floor plane here
                }
            }

            // Calculate texture column and lighting
            let mut texture_column = 0;
            let mut light_index = 0;

            if self.segtextured {
                // Calculate texture offset
                let angle = self.rw_centerangle.wrapping_add(self.xtoviewangle[self.rw_x as usize]);
                let fineangle = (angle >> ANGLETOFINESHIFT) & (FINEANGLES as u32 - 1);

                texture_column = self.rw_offset - fixed_mul(
                    self.fine_tangent[fineangle as usize],
                    self.rw_distance
                );
                texture_column >>= FRACBITS;

                // Calculate lighting
                light_index = (self.rw_scale >> LIGHTSCALESHIFT) as usize;
                if light_index >= MAXLIGHTSCALE {
                    light_index = MAXLIGHTSCALE - 1;
                }
            }

            // Draw wall tiers
            if self.midtexture != 0 {
                // Single-sided line - draw full wall
                self.draw_column(
                    framebuffer,
                    self.rw_x,
                    yl,
                    yh,
                    self.midtexture,
                    texture_column,
                    self.rw_midtexturemid,
                    light_index,
                );

                // Mark as solid wall
                self.ceiling_clip[self.rw_x as usize] = SCREEN_HEIGHT as i16;
                self.floor_clip[self.rw_x as usize] = -1;
            } else {
                // Two-sided line - draw top/bottom textures

                // Top wall (upper texture)
                if self.toptexture != 0 {
                    let mut mid = (self.pixhigh >> HEIGHTBITS) as i16;
                    self.pixhigh += self.pixhighstep;

                    if mid >= self.floor_clip[self.rw_x as usize] {
                        mid = self.floor_clip[self.rw_x as usize] - 1;
                    }

                    if mid >= yl {
                        self.draw_column(
                            framebuffer,
                            self.rw_x,
                            yl,
                            mid,
                            self.toptexture,
                            texture_column,
                            self.rw_toptexturemid,
                            light_index,
                        );
                        self.ceiling_clip[self.rw_x as usize] = mid;
                    } else {
                        self.ceiling_clip[self.rw_x as usize] = yl - 1;
                    }
                } else if self.markceiling {
                    self.ceiling_clip[self.rw_x as usize] = yl - 1;
                }

                // Bottom wall (lower texture)
                if self.bottomtexture != 0 {
                    let mut mid = ((self.pixlow + HEIGHTUNIT - 1) >> HEIGHTBITS) as i16;
                    self.pixlow += self.pixlowstep;

                    if mid <= self.ceiling_clip[self.rw_x as usize] {
                        mid = self.ceiling_clip[self.rw_x as usize] + 1;
                    }

                    if mid <= yh {
                        self.draw_column(
                            framebuffer,
                            self.rw_x,
                            mid,
                            yh,
                            self.bottomtexture,
                            texture_column,
                            self.rw_bottomtexturemid,
                            light_index,
                        );
                        self.floor_clip[self.rw_x as usize] = mid;
                    } else {
                        self.floor_clip[self.rw_x as usize] = yh + 1;
                    }
                } else if self.markfloor {
                    self.floor_clip[self.rw_x as usize] = yh + 1;
                }
            }

            // Advance to next column
            self.rw_scale += self.rw_scalestep;
            self.topfrac += self.topstep;
            self.bottomfrac += self.bottomstep;
            self.rw_x += 1;
        }
    }

    /// Draw a single texture column to the framebuffer
    fn draw_column(
        &self,
        framebuffer: &mut [u8; SCREEN_WIDTH * SCREEN_HEIGHT],
        x: i32,
        y1: i16,
        y2: i16,
        texture: i16,
        texture_col: i32,
        texture_mid: Fixed,
        light_index: usize,
    ) {
        if x < 0 || x >= SCREEN_WIDTH as i32 {
            return;
        }

        let y1 = max(0, min(y1 as i32, SCREEN_HEIGHT as i32 - 1));
        let y2 = max(0, min(y2 as i32, SCREEN_HEIGHT as i32 - 1));

        if y1 > y2 {
            return;
        }

        // Calculate texture V coordinate
        let scale = if self.rw_scale != 0 {
            self.rw_scale
        } else {
            FRACUNIT
        };

        let iscale = if scale != 0 {
            fixed_div(FRACUNIT, scale)
        } else {
            FRACUNIT
        };

        let sprtopscreen = (self.centery_frac >> 4) - fixed_mul(texture_mid, scale);

        // Draw column pixels
        for y in y1..=y2 {
            // Calculate texture coordinate
            let frac = fixed_mul((y << FRACBITS) - sprtopscreen, iscale);
            let texture_y = ((frac >> FRACBITS) & 127) as usize; // Wrap to 128 pixels

            // Get texture pixel (placeholder - would read from actual texture)
            // For now, use a simple pattern based on texture number and coordinates
            let color = self.get_texture_pixel(texture, texture_col as usize, texture_y, light_index);

            // Write to framebuffer
            let offset = (y as usize) * SCREEN_WIDTH + (x as usize);
            framebuffer[offset] = color;
        }
    }

    /// Get a texture pixel (placeholder implementation)
    fn get_texture_pixel(&self, texture: i16, x: usize, y: usize, light_index: usize) -> u8 {
        // Placeholder: generate a simple pattern
        // Real implementation would look up from texture data
        let pattern = ((x ^ y) + (texture as usize) * 17) & 0xFF;

        // Apply lighting (darken based on light index)
        let light_factor = (MAXLIGHTSCALE - light_index) * 255 / MAXLIGHTSCALE;
        let lit = (pattern * light_factor / 255) as u8;

        lit
    }

    /// Store and render a wall range
    pub fn render_seg(
        &mut self,
        seg: &Seg,
        start: i32,
        stop: i32,
        framebuffer: &mut [u8; SCREEN_WIDTH * SCREEN_HEIGHT],
    ) {
        if start >= SCREEN_WIDTH as i32 || start > stop {
            return;
        }

        // Mark linedef as visible for automap
        // (would modify linedef flags here)

        // Calculate distance to seg
        let normal_angle = seg.angle.wrapping_add(ANG90);
        let offset_angle = normal_angle.wrapping_sub(self.view_angle);

        let hyp = self.point_to_dist(seg.v1.x, seg.v1.y);
        let dist_angle = ANG90.wrapping_sub(offset_angle);
        let fineangle = (dist_angle >> ANGLETOFINESHIFT) & (FINEANGLES as u32 - 1);
        let sineval = self.fine_sine[fineangle as usize];

        self.rw_distance = fixed_mul(hyp, sineval);

        self.rw_x = start;
        self.rw_stopx = stop + 1;

        // Calculate scale at both ends
        self.rw_scale = self.scale_from_global_angle(
            self.view_angle.wrapping_add(self.xtoviewangle[start as usize])
        );

        if stop > start {
            let scale2 = self.scale_from_global_angle(
                self.view_angle.wrapping_add(self.xtoviewangle[stop as usize])
            );
            self.rw_scalestep = (scale2 - self.rw_scale) / (stop - start);
        } else {
            self.rw_scalestep = 0;
        }

        // Calculate texture boundaries
        self.world_top = seg.front_sector.ceiling_height - self.view_z;
        self.world_bottom = seg.front_sector.floor_height - self.view_z;

        self.midtexture = 0;
        self.toptexture = 0;
        self.bottomtexture = 0;
        self.maskedtexture = false;

        if let Some(ref back_sector) = seg.back_sector {
            // Two-sided line
            self.markfloor = back_sector.floor_height != seg.front_sector.floor_height
                || back_sector.floor_pic != seg.front_sector.floor_pic
                || back_sector.light_level != seg.front_sector.light_level;

            self.markceiling = back_sector.ceiling_height != seg.front_sector.ceiling_height
                || back_sector.ceiling_pic != seg.front_sector.ceiling_pic
                || back_sector.light_level != seg.front_sector.light_level;

            self.world_high = back_sector.ceiling_height - self.view_z;
            self.world_low = back_sector.floor_height - self.view_z;

            // Sky hack
            if seg.front_sector.ceiling_pic == self.sky_flat_num
                && back_sector.ceiling_pic == self.sky_flat_num
            {
                self.world_top = self.world_high;
            }

            // Top texture
            if self.world_high < self.world_top {
                self.toptexture = seg.sidedef.top_texture;

                if seg.linedef.flags & ML_DONTPEGTOP != 0 {
                    self.rw_toptexturemid = self.world_top;
                } else {
                    let vtop = back_sector.ceiling_height
                        + self.texture_heights[seg.sidedef.top_texture as usize];
                    self.rw_toptexturemid = vtop - self.view_z;
                }
            }

            // Bottom texture
            if self.world_low > self.world_bottom {
                self.bottomtexture = seg.sidedef.bottom_texture;

                if seg.linedef.flags & ML_DONTPEGBOTTOM != 0 {
                    self.rw_bottomtexturemid = self.world_top;
                } else {
                    self.rw_bottomtexturemid = self.world_low;
                }
            }

            self.rw_toptexturemid += seg.sidedef.row_offset;
            self.rw_bottomtexturemid += seg.sidedef.row_offset;

            // Masked midtexture
            if seg.sidedef.mid_texture != 0 {
                self.maskedtexture = true;
            }
        } else {
            // Single-sided line
            self.midtexture = seg.sidedef.mid_texture;
            self.markfloor = true;
            self.markceiling = true;

            if seg.linedef.flags & ML_DONTPEGBOTTOM != 0 {
                let vtop = seg.front_sector.floor_height
                    + self.texture_heights[seg.sidedef.mid_texture as usize];
                self.rw_midtexturemid = vtop - self.view_z;
            } else {
                self.rw_midtexturemid = self.world_top;
            }

            self.rw_midtexturemid += seg.sidedef.row_offset;
        }

        // Calculate texture offset
        self.segtextured = self.midtexture != 0
            || self.toptexture != 0
            || self.bottomtexture != 0
            || self.maskedtexture;

        if self.segtextured {
            let offset_angle = normal_angle.wrapping_sub(self.view_angle);

            let offset_angle = if offset_angle > ANG180 {
                offset_angle.wrapping_neg()
            } else {
                offset_angle
            };

            let fineangle = (offset_angle >> ANGLETOFINESHIFT) & (FINEANGLES as u32 - 1);
            let sineval = self.fine_sine[fineangle as usize];

            self.rw_offset = fixed_mul(hyp, sineval);

            if normal_angle.wrapping_sub(self.view_angle) < ANG180 {
                self.rw_offset = -self.rw_offset;
            }

            self.rw_offset += seg.sidedef.texture_offset + seg.offset;
            self.rw_centerangle = ANG90.wrapping_add(self.view_angle).wrapping_sub(normal_angle);
        }

        // Don't mark planes if on wrong side of view
        if seg.front_sector.floor_height >= self.view_z {
            self.markfloor = false;
        }

        if seg.front_sector.ceiling_height <= self.view_z
            && seg.front_sector.ceiling_pic != self.sky_flat_num
        {
            self.markceiling = false;
        }

        // Calculate incremental stepping values for texture edges
        self.world_top >>= 4;
        self.world_bottom >>= 4;

        self.topstep = -fixed_mul(self.rw_scalestep, self.world_top);
        self.topfrac = (self.centery_frac >> 4) - fixed_mul(self.world_top, self.rw_scale);

        self.bottomstep = -fixed_mul(self.rw_scalestep, self.world_bottom);
        self.bottomfrac = (self.centery_frac >> 4) - fixed_mul(self.world_bottom, self.rw_scale);

        if seg.back_sector.is_some() {
            self.world_high >>= 4;
            self.world_low >>= 4;

            if self.world_high < self.world_top {
                self.pixhigh = (self.centery_frac >> 4) - fixed_mul(self.world_high, self.rw_scale);
                self.pixhighstep = -fixed_mul(self.rw_scalestep, self.world_high);
            }

            if self.world_low > self.world_bottom {
                self.pixlow = (self.centery_frac >> 4) - fixed_mul(self.world_low, self.rw_scale);
                self.pixlowstep = -fixed_mul(self.rw_scalestep, self.world_low);
            }
        }

        // Render the seg
        self.render_seg_loop(framebuffer);
    }

    /// Set view parameters
    pub fn set_view(&mut self, x: Fixed, y: Fixed, z: Fixed, angle: Angle) {
        self.view_x = x;
        self.view_y = y;
        self.view_z = z;
        self.view_angle = angle;

        // Calculate view sin/cos
        let fineangle = (angle >> ANGLETOFINESHIFT) & (FINEANGLES as u32 - 1);
        self.view_cos = self.fine_sine[((fineangle + FINEANGLES as u32 / 4) & (FINEANGLES as u32 - 1)) as usize];
        self.view_sin = self.fine_sine[fineangle as usize];
    }

    /// Reset clipping arrays for new frame
    pub fn reset_clip_arrays(&mut self) {
        for i in 0..SCREEN_WIDTH {
            self.ceiling_clip[i] = -1;
            self.floor_clip[i] = SCREEN_HEIGHT as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_mul() {
        assert_eq!(fixed_mul(FRACUNIT, FRACUNIT), FRACUNIT);
        assert_eq!(fixed_mul(FRACUNIT * 2, FRACUNIT / 2), FRACUNIT);
    }

    #[test]
    fn test_seg_renderer_new() {
        let renderer = SegRenderer::new();
        assert_eq!(renderer.centerx, (SCREEN_WIDTH / 2) as i32);
        assert_eq!(renderer.centery, (SCREEN_HEIGHT / 2) as i32);
    }
}
