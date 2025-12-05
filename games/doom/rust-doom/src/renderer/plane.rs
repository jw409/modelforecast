//! Floor and ceiling rendering (visplanes)
//!
//! This module implements the visplane system for rendering horizontal surfaces
//! (floors and ceilings) in DOOM. The algorithm works as follows:
//!
//! 1. During BSP traversal, walls set floor/ceiling clip arrays
//! 2. Sectors create visplanes (one per unique height/texture/light combination)
//! 3. After walls are drawn, visplanes are converted to horizontal spans
//! 4. Each span is texture-mapped using perspective-correct math
//!
//! Reference: linuxdoom-1.10/r_plane.c

use std::ptr;

// DOOM fixed-point constants (16.16 format)
pub type Fixed = i32;
pub const FRACBITS: i32 = 16;
pub const FRACUNIT: Fixed = 1 << FRACBITS;

// Angle constants
pub type Angle = u32;
pub const ANGLETOFINESHIFT: u32 = 19;
pub const FINEANGLES: usize = 8192;
const ANGLE_90: Angle = 0x40000000;

// Screen dimensions
pub const SCREENWIDTH: usize = 320;
pub const SCREENHEIGHT: usize = 200;

// Lighting constants
pub const LIGHTLEVELS: usize = 16;
pub const LIGHTSEGSHIFT: i32 = 4;
pub const MAXLIGHTZ: usize = 128;
pub const LIGHTZSHIFT: i32 = 20;

// Visplane limits
const MAXVISPLANES: usize = 128;
const MAXOPENINGS: usize = SCREENWIDTH * 64;

/// A visplane represents a horizontal surface (floor or ceiling) to be rendered.
/// Multiple sectors can share the same visplane if they have matching properties.
#[derive(Clone)]
pub struct VisPlane {
    /// Height of the plane in world coordinates (fixed-point)
    pub height: Fixed,
    /// Texture/flat number to render
    pub picnum: i32,
    /// Light level (0-255)
    pub lightlevel: i32,
    /// Leftmost column with content
    pub minx: i32,
    /// Rightmost column with content
    pub maxx: i32,
    /// Top clip array - tracks ceiling boundaries for each column
    /// 0xff means no content at that column
    pub top: [u8; SCREENWIDTH + 2], // Extra padding for [-1] and [+1] access
    /// Bottom clip array - tracks floor boundaries for each column
    pub bottom: [u8; SCREENWIDTH + 2],
}

impl VisPlane {
    /// Create a new empty visplane
    fn new() -> Self {
        VisPlane {
            height: 0,
            picnum: 0,
            lightlevel: 0,
            minx: SCREENWIDTH as i32,
            maxx: -1,
            top: [0xff; SCREENWIDTH + 2],
            bottom: [0xff; SCREENWIDTH + 2],
        }
    }
}

/// Plane renderer state
pub struct PlaneRenderer {
    // Visplane storage
    visplanes: Vec<VisPlane>,
    last_visplane: usize,

    // Opening storage for clipping
    openings: Vec<i16>,
    last_opening: usize,

    // Clip arrays (set by wall renderer)
    pub floor_clip: [i16; SCREENWIDTH],
    pub ceiling_clip: [i16; SCREENWIDTH],

    // Span tracking for horizontal strips
    span_start: [i32; SCREENHEIGHT],

    // Texture mapping state
    plane_height: Fixed,

    // Precalculated values for perspective
    pub yslope: [Fixed; SCREENHEIGHT],
    pub distscale: [Fixed; SCREENWIDTH],

    // Cached calculations for span drawing
    cached_height: [Fixed; SCREENHEIGHT],
    cached_distance: [Fixed; SCREENHEIGHT],
    cached_xstep: [Fixed; SCREENHEIGHT],
    cached_ystep: [Fixed; SCREENHEIGHT],

    // Texture mapping scales
    basexscale: Fixed,
    baseyscale: Fixed,

    // View parameters (set externally)
    pub viewx: Fixed,
    pub viewy: Fixed,
    pub viewz: Fixed,
    pub viewangle: Angle,
    pub centerxfrac: Fixed,

    // Sky flat number (special case - rendered differently)
    pub sky_flatnum: i32,

    // Extra light from player powerups
    pub extralight: i32,
}

impl PlaneRenderer {
    /// Create a new plane renderer
    pub fn new() -> Self {
        PlaneRenderer {
            visplanes: vec![VisPlane::new(); MAXVISPLANES],
            last_visplane: 0,
            openings: vec![0; MAXOPENINGS],
            last_opening: 0,
            floor_clip: [SCREENHEIGHT as i16; SCREENWIDTH],
            ceiling_clip: [-1; SCREENWIDTH],
            span_start: [0; SCREENHEIGHT],
            plane_height: 0,
            yslope: [0; SCREENHEIGHT],
            distscale: [0; SCREENWIDTH],
            cached_height: [0; SCREENHEIGHT],
            cached_distance: [0; SCREENHEIGHT],
            cached_xstep: [0; SCREENHEIGHT],
            cached_ystep: [0; SCREENHEIGHT],
            basexscale: 0,
            baseyscale: 0,
            viewx: 0,
            viewy: 0,
            viewz: 0,
            viewangle: 0,
            centerxfrac: (SCREENWIDTH / 2) << FRACBITS,
            sky_flatnum: -1,
            extralight: 0,
        }
    }

    /// Initialize for a new frame
    pub fn clear_planes(&mut self, xtoviewangle: &[Angle; SCREENWIDTH], finecosine: &[Fixed], finesine: &[Fixed]) {
        // Reset clipping arrays
        for i in 0..SCREENWIDTH {
            self.floor_clip[i] = SCREENHEIGHT as i16;
            self.ceiling_clip[i] = -1;
        }

        // Reset visplane tracking
        self.last_visplane = 0;
        self.last_opening = 0;

        // Clear cached heights
        self.cached_height.fill(0);

        // Calculate base texture mapping scales
        // This is the scale at the center of the screen
        let angle = self.viewangle.wrapping_sub(ANGLE_90) >> ANGLETOFINESHIFT;
        let angle_idx = (angle & (FINEANGLES as u32 - 1)) as usize;

        self.basexscale = fixed_div(finecosine[angle_idx], self.centerxfrac);
        self.baseyscale = -fixed_div(finesine[angle_idx], self.centerxfrac);
    }

    /// Find or create a visplane for the given properties
    pub fn find_plane(&mut self, height: Fixed, picnum: i32, lightlevel: i32) -> usize {
        let mut height = height;
        let mut lightlevel = lightlevel;

        // Sky flats all share the same visplane (height/light don't matter)
        if picnum == self.sky_flatnum {
            height = 0;
            lightlevel = 0;
        }

        // Search for existing matching visplane
        for i in 0..self.last_visplane {
            let plane = &self.visplanes[i];
            if height == plane.height
                && picnum == plane.picnum
                && lightlevel == plane.lightlevel
            {
                return i;
            }
        }

        // Need a new visplane
        if self.last_visplane >= MAXVISPLANES {
            panic!("R_FindPlane: no more visplanes (max {})", MAXVISPLANES);
        }

        let idx = self.last_visplane;
        self.last_visplane += 1;

        // Initialize the new visplane
        let plane = &mut self.visplanes[idx];
        plane.height = height;
        plane.picnum = picnum;
        plane.lightlevel = lightlevel;
        plane.minx = SCREENWIDTH as i32;
        plane.maxx = -1;
        plane.top.fill(0xff);
        plane.bottom.fill(0xff);

        idx
    }

    /// Check if a visplane can be extended, or create a new one
    pub fn check_plane(&mut self, plane_idx: usize, start: i32, stop: i32) -> usize {
        let plane = &self.visplanes[plane_idx];

        // Calculate intersection and union of ranges
        let (intrl, unionl) = if start < plane.minx {
            (plane.minx, start)
        } else {
            (start, plane.minx)
        };

        let (intrh, unionh) = if stop > plane.maxx {
            (plane.maxx, stop)
        } else {
            (stop, plane.maxx)
        };

        // Check if intersection range is clear
        let mut can_extend = true;
        for x in intrl..=intrh {
            if x >= 0 && (x as usize) < SCREENWIDTH {
                if plane.top[x as usize + 1] != 0xff {
                    can_extend = false;
                    break;
                }
            }
        }

        if can_extend {
            // Extend the existing plane
            let plane = &mut self.visplanes[plane_idx];
            plane.minx = unionl;
            plane.maxx = unionh;
            return plane_idx;
        }

        // Make a new visplane with same properties
        if self.last_visplane >= MAXVISPLANES {
            panic!("R_CheckPlane: no more visplanes");
        }

        let old_plane = &self.visplanes[plane_idx];
        let new_idx = self.last_visplane;
        self.last_visplane += 1;

        let new_plane = &mut self.visplanes[new_idx];
        new_plane.height = old_plane.height;
        new_plane.picnum = old_plane.picnum;
        new_plane.lightlevel = old_plane.lightlevel;
        new_plane.minx = start;
        new_plane.maxx = stop;
        new_plane.top.fill(0xff);
        new_plane.bottom.fill(0xff);

        new_idx
    }

    /// Convert visplane columns to horizontal spans and draw them
    fn make_spans(
        &mut self,
        x: i32,
        t1: i32,
        b1: i32,
        t2: i32,
        b2: i32,
        map_span: &mut dyn FnMut(i32, i32, i32),
    ) {
        let mut t1 = t1;
        let mut b1 = b1;
        let mut t2 = t2;
        let mut b2 = b2;

        // Draw spans that closed between t1 and t2
        while t1 < t2 && t1 <= b1 {
            self.map_plane_span(t1, self.span_start[t1 as usize], x - 1, map_span);
            t1 += 1;
        }

        while b1 > b2 && b1 >= t1 {
            self.map_plane_span(b1, self.span_start[b1 as usize], x - 1, map_span);
            b1 -= 1;
        }

        // Mark new spans that opened
        while t2 < t1 && t2 <= b2 {
            self.span_start[t2 as usize] = x;
            t2 += 1;
        }

        while b2 > b1 && b2 >= t2 {
            self.span_start[b2 as usize] = x;
            b2 -= 1;
        }
    }

    /// Map a single horizontal span (one scanline of floor/ceiling)
    fn map_plane_span(
        &mut self,
        y: i32,
        x1: i32,
        x2: i32,
        draw_span: &mut dyn FnMut(i32, i32, i32),
    ) {
        if x2 < x1 || x1 < 0 || x2 >= SCREENWIDTH as i32 || y < 0 || y >= SCREENHEIGHT as i32 {
            return; // Out of bounds
        }

        let y_idx = y as usize;

        // Calculate distance and step values (with caching)
        if self.plane_height != self.cached_height[y_idx] {
            self.cached_height[y_idx] = self.plane_height;
            let distance = fixed_mul(self.plane_height, self.yslope[y_idx]);
            self.cached_distance[y_idx] = distance;
            self.cached_xstep[y_idx] = fixed_mul(distance, self.basexscale);
            self.cached_ystep[y_idx] = fixed_mul(distance, self.baseyscale);
        }

        // Call the span drawing function with the y coordinate and x range
        draw_span(y, x1, x2);
    }

    /// Draw all accumulated visplanes
    pub fn draw_planes(
        &mut self,
        finecosine: &[Fixed],
        finesine: &[Fixed],
        xtoviewangle: &[Angle; SCREENWIDTH],
        zlight: &[[*const u8; MAXLIGHTZ]; LIGHTLEVELS],
        fixedcolormap: *const u8,
        firstflat: i32,
        flattranslation: &[i32],
        get_flat_data: &mut dyn FnMut(i32) -> *const u8,
        draw_span: &mut dyn FnMut(i32, i32, i32),
    ) {
        for i in 0..self.last_visplane {
            let plane = &self.visplanes[i];

            if plane.minx > plane.maxx {
                continue; // Empty plane
            }

            // Handle sky specially (not implemented here - would draw sky texture)
            if plane.picnum == self.sky_flatnum {
                // Sky rendering would go here
                // Typically draws a tiled sky texture using column renderer
                continue;
            }

            // Get the flat texture data
            let flat_idx = firstflat + flattranslation[plane.picnum as usize];
            let _ds_source = get_flat_data(flat_idx);

            // Calculate plane height and lighting
            self.plane_height = (plane.height - self.viewz).abs();
            let mut light = (plane.lightlevel >> LIGHTSEGSHIFT) + self.extralight;

            if light >= LIGHTLEVELS as i32 {
                light = LIGHTLEVELS as i32 - 1;
            }
            if light < 0 {
                light = 0;
            }

            let _planezlight = &zlight[light as usize];

            // Pad the top array for span generation
            let minx = plane.minx as usize;
            let maxx = plane.maxx as usize;

            // Make a mutable copy for span generation
            let mut top = plane.top;
            let bottom = plane.bottom;

            if maxx + 1 < SCREENWIDTH {
                top[maxx + 2] = 0xff;
            }
            if minx > 0 {
                top[minx] = 0xff;
            }

            // Generate spans across the visplane
            let stop = (plane.maxx + 1) as i32;
            for x in plane.minx..=stop {
                let x_idx = (x as usize).saturating_sub(1) + 1;
                let t_prev = if x > 0 { top[x_idx] as i32 } else { 0xff };
                let b_prev = if x > 0 { bottom[x_idx] as i32 } else { 0xff };
                let t_curr = top[(x as usize) + 1] as i32;
                let b_curr = bottom[(x as usize) + 1] as i32;

                self.make_spans(x, t_prev, b_prev, t_curr, b_curr, draw_span);
            }
        }
    }
}

impl Default for PlaneRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// Fixed-point math helpers

/// Multiply two fixed-point numbers
#[inline]
pub fn fixed_mul(a: Fixed, b: Fixed) -> Fixed {
    ((a as i64 * b as i64) >> FRACBITS) as Fixed
}

/// Divide two fixed-point numbers
#[inline]
pub fn fixed_div(a: Fixed, b: Fixed) -> Fixed {
    if b == 0 {
        if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        (((a as i64) << FRACBITS) / (b as i64)) as Fixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visplane_creation() {
        let mut renderer = PlaneRenderer::new();
        renderer.sky_flatnum = 100;

        // Find a plane
        let idx = renderer.find_plane(100 << FRACBITS, 5, 128);
        assert_eq!(idx, 0);
        assert_eq!(renderer.visplanes[idx].height, 100 << FRACBITS);
        assert_eq!(renderer.visplanes[idx].picnum, 5);

        // Find same plane again - should return same index
        let idx2 = renderer.find_plane(100 << FRACBITS, 5, 128);
        assert_eq!(idx, idx2);

        // Find different plane - should create new
        let idx3 = renderer.find_plane(200 << FRACBITS, 5, 128);
        assert_eq!(idx3, 1);
    }

    #[test]
    fn test_sky_plane_normalization() {
        let mut renderer = PlaneRenderer::new();
        renderer.sky_flatnum = 100;

        // Sky flats should normalize height and light
        let idx1 = renderer.find_plane(50 << FRACBITS, 100, 200);
        let idx2 = renderer.find_plane(999 << FRACBITS, 100, 50);

        // Should be the same plane (both sky)
        assert_eq!(idx1, idx2);
        assert_eq!(renderer.visplanes[idx1].height, 0);
        assert_eq!(renderer.visplanes[idx1].lightlevel, 0);
    }

    #[test]
    fn test_fixed_mul() {
        assert_eq!(fixed_mul(FRACUNIT, FRACUNIT), FRACUNIT); // 1.0 * 1.0 = 1.0
        assert_eq!(fixed_mul(FRACUNIT * 2, FRACUNIT / 2), FRACUNIT); // 2.0 * 0.5 = 1.0
        assert_eq!(fixed_mul(FRACUNIT * 3, FRACUNIT * 4), FRACUNIT * 12); // 3.0 * 4.0 = 12.0
    }

    #[test]
    fn test_fixed_div() {
        assert_eq!(fixed_div(FRACUNIT, FRACUNIT), FRACUNIT); // 1.0 / 1.0 = 1.0
        assert_eq!(fixed_div(FRACUNIT * 4, FRACUNIT * 2), FRACUNIT * 2); // 4.0 / 2.0 = 2.0
        assert_eq!(fixed_div(FRACUNIT, FRACUNIT * 2), FRACUNIT / 2); // 1.0 / 2.0 = 0.5

        // Division by zero
        assert_eq!(fixed_div(FRACUNIT, 0), i32::MAX);
        assert_eq!(fixed_div(-FRACUNIT, 0), i32::MIN);
    }

    #[test]
    fn test_clear_planes() {
        let mut renderer = PlaneRenderer::new();
        let dummy_angles = [0; SCREENWIDTH];
        let dummy_cos = vec![FRACUNIT; FINEANGLES];
        let dummy_sin = vec![0; FINEANGLES];

        // Add some planes
        renderer.find_plane(100 << FRACBITS, 1, 128);
        renderer.find_plane(200 << FRACBITS, 2, 64);
        assert_eq!(renderer.last_visplane, 2);

        // Clear should reset
        renderer.clear_planes(&dummy_angles, &dummy_cos, &dummy_sin);
        assert_eq!(renderer.last_visplane, 0);

        // Clip arrays should be reset
        assert_eq!(renderer.floor_clip[0], SCREENHEIGHT as i16);
        assert_eq!(renderer.ceiling_clip[0], -1);
    }
}
