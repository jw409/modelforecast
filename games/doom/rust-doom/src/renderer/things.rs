// Emacs style mode select   -*- Rust -*-
//-----------------------------------------------------------------------------
//
// Copyright (C) 1993-1996 by id Software, Inc.
// Copyright (C) 2024 Rust DOOM WASM Port
//
// This source is available for distribution and/or modification
// only under the terms of the DOOM Source Code License as
// published by id Software. All rights reserved.
//
// DESCRIPTION:
//  Refresh of things, i.e. objects represented by sprites.
//  Sprite rendering with proper depth sorting, clipping, and transparency.
//
//-----------------------------------------------------------------------------

use std::cmp::{max, min};
use std::ptr;

// Fixed point math (16.16 format)
pub type Fixed = i32;
pub type Angle = u32;

const FRACBITS: i32 = 16;
const FRACUNIT: Fixed = 1 << FRACBITS;

// Minimum Z distance for sprite rendering
const MINZ: Fixed = FRACUNIT * 4;
const BASEYCENTER: i32 = 100;

// Maximum visible sprites per frame
const MAXVISSPRITES: usize = 128;

// Screen dimensions (will be set at runtime)
const SCREENWIDTH: usize = 320;
const SCREENHEIGHT: usize = 200;

// Light scale constants
const LIGHTSCALESHIFT: i32 = 12;
const MAXLIGHTSCALE: usize = 48;
const LIGHTLEVELS: usize = 16;
const LIGHTSEGSHIFT: i32 = 4;

// Sprite frame flags
const FF_FRAMEMASK: u32 = 0x7FFF;
const FF_FULLBRIGHT: u32 = 0x8000;

// Mobj flags
const MF_SHADOW: u32 = 0x00000010;
const MF_TRANSLATION: u32 = 0x04000000;
const MF_TRANSSHIFT: u32 = 26;

// Silhouette flags for clipping
const SIL_NONE: i32 = 0;
const SIL_BOTTOM: i32 = 1;
const SIL_TOP: i32 = 2;
const SIL_BOTH: i32 = 3;

//-----------------------------------------------------------------------------
// Data Structures
//-----------------------------------------------------------------------------

/// A post is a run of non-masked source pixels in a column
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Post {
    pub topdelta: u8,  // 0xFF marks end of column
    pub length: u8,    // Number of pixels in this post
}

/// Column is a list of posts
pub type Column = Post;

/// Patch structure - sprite/texture with column-based format
#[repr(C)]
pub struct Patch {
    pub width: i16,
    pub height: i16,
    pub leftoffset: i16,
    pub topoffset: i16,
    pub columnofs: [i32; 1], // Variable length array - only [width] used
}

/// Sprite frame - one frame of animation with rotations
#[derive(Clone)]
pub struct SpriteFrame {
    pub rotate: bool,      // If false, use lump[0] for all views
    pub lump: [i16; 8],    // Lump indices for 8 rotations (0-7)
    pub flip: [u8; 8],     // Horizontal flip flags
}

impl SpriteFrame {
    pub fn new() -> Self {
        SpriteFrame {
            rotate: false,
            lump: [-1; 8],
            flip: [0; 8],
        }
    }
}

/// Sprite definition - complete sprite with all frames
pub struct SpriteDef {
    pub numframes: usize,
    pub spriteframes: Vec<SpriteFrame>,
}

/// Visible sprite - a thing that will be drawn this frame
#[derive(Clone)]
pub struct VisSprite {
    // Linked list for sorting
    pub prev: Option<usize>,
    pub next: Option<usize>,

    // Screen coordinates
    pub x1: i32,
    pub x2: i32,

    // World position (for line side calculation)
    pub gx: Fixed,
    pub gy: Fixed,

    // Global bottom/top for silhouette clipping
    pub gz: Fixed,
    pub gzt: Fixed,

    // Horizontal position and scale
    pub startfrac: Fixed,
    pub scale: Fixed,
    pub xiscale: Fixed,  // Negative if flipped

    // Texture positioning
    pub texturemid: Fixed,
    pub patch: usize,

    // Lighting and effects
    pub colormap: Option<usize>,  // None = shadow draw
    pub mobjflags: u32,
}

impl VisSprite {
    pub fn new() -> Self {
        VisSprite {
            prev: None,
            next: None,
            x1: 0,
            x2: 0,
            gx: 0,
            gy: 0,
            gz: 0,
            gzt: 0,
            startfrac: 0,
            scale: 0,
            xiscale: 0,
            texturemid: 0,
            patch: 0,
            colormap: None,
            mobjflags: 0,
        }
    }
}

/// Drawseg - wall segment used for sprite clipping
pub struct DrawSeg {
    pub x1: i32,
    pub x2: i32,
    pub scale1: Fixed,
    pub scale2: Fixed,
    pub silhouette: i32,
    pub bsilheight: Fixed,  // Bottom silhouette height
    pub tsilheight: Fixed,  // Top silhouette height
    pub sprtopclip: Vec<i16>,
    pub sprbottomclip: Vec<i16>,
    pub maskedtexturecol: Option<Vec<i16>>,
}

/// Map object (thing) - simplified version
pub struct MapObj {
    pub x: Fixed,
    pub y: Fixed,
    pub z: Fixed,
    pub angle: Angle,
    pub sprite: usize,
    pub frame: u32,
    pub flags: u32,
}

/// Sector - map sector
pub struct Sector {
    pub lightlevel: i16,
    pub validcount: i32,
    pub thinglist: Vec<MapObj>,
}

//-----------------------------------------------------------------------------
// Renderer State
//-----------------------------------------------------------------------------

pub struct ThingRenderer {
    // Sprite definitions
    sprites: Vec<SpriteDef>,
    numsprites: usize,

    // Visible sprites for current frame
    vissprites: Vec<VisSprite>,
    vissprite_count: usize,
    overflow_sprite: VisSprite,

    // Sorted sprite list (indices)
    sorted_sprites: Vec<usize>,

    // Sprite offset/size lookup tables
    spriteoffset: Vec<Fixed>,
    spritewidth: Vec<Fixed>,
    spritetopoffset: Vec<Fixed>,

    // Clipping arrays
    negonearray: [i16; SCREENWIDTH],
    screenheightarray: [i16; SCREENHEIGHT],

    // Current clip arrays (set during drawing)
    mfloorclip: Vec<i16>,
    mceilingclip: Vec<i16>,

    // Drawing state
    spryscale: Fixed,
    sprtopscreen: Fixed,

    // View state
    viewx: Fixed,
    viewy: Fixed,
    viewz: Fixed,
    viewangle: Angle,
    viewcos: Fixed,
    viewsin: Fixed,
    projection: Fixed,
    centerx: i32,
    centerxfrac: Fixed,
    centeryfrac: Fixed,

    // Light tables
    spritelights: Vec<Vec<u8>>,  // Current light level
    scalelight: Vec<Vec<Vec<u8>>>, // [lightlevel][scale]
    colormaps: Vec<u8>,
    fixedcolormap: Option<Vec<u8>>,

    // Extra light (from power-ups, gun flash, etc)
    extralight: i32,

    // Validation counter for BSP traversal
    validcount: i32,

    // Detail level (0 = high, 1 = low)
    detailshift: i32,

    // Drawsegs for clipping
    drawsegs: Vec<DrawSeg>,
}

impl ThingRenderer {
    pub fn new() -> Self {
        let mut negone = [-1i16; SCREENWIDTH];
        let mut screenheight = [0i16; SCREENHEIGHT];

        for i in 0..SCREENWIDTH {
            negone[i] = -1;
        }
        for i in 0..SCREENHEIGHT {
            screenheight[i] = SCREENHEIGHT as i16;
        }

        ThingRenderer {
            sprites: Vec::new(),
            numsprites: 0,
            vissprites: vec![VisSprite::new(); MAXVISSPRITES],
            vissprite_count: 0,
            overflow_sprite: VisSprite::new(),
            sorted_sprites: Vec::new(),
            spriteoffset: Vec::new(),
            spritewidth: Vec::new(),
            spritetopoffset: Vec::new(),
            negonearray: negone,
            screenheightarray: screenheight,
            mfloorclip: vec![0; SCREENWIDTH],
            mceilingclip: vec![0; SCREENWIDTH],
            spryscale: 0,
            sprtopscreen: 0,
            viewx: 0,
            viewy: 0,
            viewz: 0,
            viewangle: 0,
            viewcos: FRACUNIT,
            viewsin: 0,
            projection: 0,
            centerx: SCREENWIDTH as i32 / 2,
            centerxfrac: (SCREENWIDTH as i32 / 2) << FRACBITS,
            centeryfrac: (SCREENHEIGHT as i32 / 2) << FRACBITS,
            spritelights: Vec::new(),
            scalelight: Vec::new(),
            colormaps: Vec::new(),
            fixedcolormap: None,
            extralight: 0,
            validcount: 0,
            detailshift: 0,
            drawsegs: Vec::new(),
        }
    }

    //-------------------------------------------------------------------------
    // Initialization
    //-------------------------------------------------------------------------

    /// Initialize sprite system with sprite name list
    pub fn init_sprites(&mut self, namelist: Vec<String>) {
        self.numsprites = namelist.len();
        self.sprites = vec![SpriteDef { numframes: 0, spriteframes: Vec::new() }; self.numsprites];

        // In real implementation, would scan WAD lumps and build sprite frames
        // This is a placeholder for the initialization logic
    }

    /// Clear sprites at start of frame
    pub fn clear_sprites(&mut self) {
        self.vissprite_count = 0;
    }

    //-------------------------------------------------------------------------
    // Sprite Projection
    //-------------------------------------------------------------------------

    /// Project a 3D thing to screen space and create a vissprite
    pub fn project_sprite(&mut self, thing: &MapObj) {
        // Transform thing position to view space
        let tr_x = thing.x - self.viewx;
        let tr_y = thing.y - self.viewy;

        let gxt = self.fixed_mul(tr_x, self.viewcos);
        let gyt = -self.fixed_mul(tr_y, self.viewsin);
        let tz = gxt - gyt;

        // Thing is behind view plane?
        if tz < MINZ {
            return;
        }

        let xscale = self.fixed_div(self.projection, tz);

        let gxt = -self.fixed_mul(tr_x, self.viewsin);
        let gyt = self.fixed_mul(tr_y, self.viewcos);
        let tx = -(gyt + gxt);

        // Too far off the side?
        if tx.abs() > (tz << 2) {
            return;
        }

        // Get sprite definition
        if thing.sprite >= self.numsprites {
            return; // Invalid sprite
        }

        let sprdef = &self.sprites[thing.sprite];
        let frame = (thing.frame & FF_FRAMEMASK) as usize;

        if frame >= sprdef.numframes {
            return; // Invalid frame
        }

        let sprframe = &sprdef.spriteframes[frame];

        // Determine which rotation to use
        let (lump, flip) = if sprframe.rotate {
            // Choose rotation based on player view angle
            let ang = self.point_to_angle(thing.x, thing.y);
            let rot = ((ang.wrapping_sub(thing.angle).wrapping_add(
                (0x20000000u32) * 9)) >> 29) as usize;
            (sprframe.lump[rot] as usize, sprframe.flip[rot] != 0)
        } else {
            // Use single rotation for all views
            (sprframe.lump[0] as usize, sprframe.flip[0] != 0)
        };

        // Calculate screen edges
        let mut tx = tx;
        tx -= self.spriteoffset[lump];
        let x1 = ((self.centerxfrac + self.fixed_mul(tx, xscale)) >> FRACBITS) as i32;

        // Off the right side?
        if x1 > SCREENWIDTH as i32 {
            return;
        }

        tx += self.spritewidth[lump];
        let x2 = (((self.centerxfrac + self.fixed_mul(tx, xscale)) >> FRACBITS) - 1) as i32;

        // Off the left side?
        if x2 < 0 {
            return;
        }

        // Allocate a vissprite
        let vis = self.new_vissprite();
        if vis >= MAXVISSPRITES {
            return; // Overflow
        }

        // Fill in vissprite information
        let vissprite = &mut self.vissprites[vis];
        vissprite.mobjflags = thing.flags;
        vissprite.scale = xscale << self.detailshift;
        vissprite.gx = thing.x;
        vissprite.gy = thing.y;
        vissprite.gz = thing.z;
        vissprite.gzt = thing.z + self.spritetopoffset[lump];
        vissprite.texturemid = vissprite.gzt - self.viewz;
        vissprite.x1 = if x1 < 0 { 0 } else { x1 };
        vissprite.x2 = if x2 >= SCREENWIDTH as i32 { SCREENWIDTH as i32 - 1 } else { x2 };

        let iscale = self.fixed_div(FRACUNIT, xscale);

        if flip {
            vissprite.startfrac = self.spritewidth[lump] - 1;
            vissprite.xiscale = -iscale;
        } else {
            vissprite.startfrac = 0;
            vissprite.xiscale = iscale;
        }

        if vissprite.x1 > x1 {
            vissprite.startfrac += vissprite.xiscale * (vissprite.x1 - x1);
        }

        vissprite.patch = lump;

        // Get light level
        if thing.flags & MF_SHADOW != 0 {
            // Shadow draw
            vissprite.colormap = None;
        } else if self.fixedcolormap.is_some() {
            // Fixed colormap (invulnerability, etc)
            vissprite.colormap = Some(0); // Index into fixedcolormap
        } else if thing.frame & FF_FULLBRIGHT != 0 {
            // Full bright
            vissprite.colormap = Some(0); // Index into colormaps
        } else {
            // Diminished light based on distance
            let mut index = (xscale >> (LIGHTSCALESHIFT - self.detailshift)) as usize;
            if index >= MAXLIGHTSCALE {
                index = MAXLIGHTSCALE - 1;
            }
            vissprite.colormap = Some(index);
        }
    }

    /// Add all sprites in a sector to the visible sprite list
    pub fn add_sprites(&mut self, sec: &Sector) {
        // Check if sector already added
        if sec.validcount == self.validcount {
            return;
        }

        // Calculate light level for this sector
        let mut lightnum = ((sec.lightlevel as i32) >> LIGHTSEGSHIFT) + self.extralight;

        if lightnum < 0 {
            lightnum = 0;
        } else if lightnum >= LIGHTLEVELS as i32 {
            lightnum = LIGHTLEVELS as i32 - 1;
        }

        // Set current sprite lights
        if lightnum < self.scalelight.len() as i32 {
            // self.spritelights = &self.scalelight[lightnum as usize];
        }

        // Project all things in sector
        for thing in &sec.thinglist {
            self.project_sprite(thing);
        }
    }

    //-------------------------------------------------------------------------
    // Sprite Sorting
    //-------------------------------------------------------------------------

    /// Sort visible sprites back to front (painter's algorithm)
    pub fn sort_vissprites(&mut self) {
        if self.vissprite_count == 0 {
            return;
        }

        // Simple insertion sort by scale (smaller scale = farther = draw first)
        self.sorted_sprites.clear();

        for i in 0..self.vissprite_count {
            self.sorted_sprites.push(i);
        }

        // Sort by scale (ascending - farthest first)
        self.sorted_sprites.sort_by(|&a, &b| {
            self.vissprites[a].scale.cmp(&self.vissprites[b].scale)
        });
    }

    //-------------------------------------------------------------------------
    // Sprite Drawing
    //-------------------------------------------------------------------------

    /// Draw a masked column (sprite column with transparency)
    pub fn draw_masked_column(&mut self, column_data: &[u8], dc_x: i32) {
        let mut offset = 0;

        loop {
            // Read post header
            if offset >= column_data.len() {
                break;
            }

            let topdelta = column_data[offset];
            if topdelta == 0xFF {
                break; // End of column
            }

            offset += 1;
            if offset >= column_data.len() {
                break;
            }

            let length = column_data[offset];
            offset += 1;

            // Calculate screen coordinates for this post
            let topscreen = self.sprtopscreen + self.fixed_mul(self.spryscale, topdelta as i32);
            let bottomscreen = topscreen + self.fixed_mul(self.spryscale, length as i32);

            let mut dc_yl = ((topscreen + FRACUNIT - 1) >> FRACBITS) as i32;
            let mut dc_yh = ((bottomscreen - 1) >> FRACBITS) as i32;

            // Clip to floor/ceiling
            if dc_yh >= self.mfloorclip[dc_x as usize] as i32 {
                dc_yh = self.mfloorclip[dc_x as usize] as i32 - 1;
            }
            if dc_yl <= self.mceilingclip[dc_x as usize] as i32 {
                dc_yl = self.mceilingclip[dc_x as usize] as i32 + 1;
            }

            // Draw the column segment
            if dc_yl <= dc_yh {
                offset += 1; // Skip padding byte

                // Here we would call the actual column drawing function
                // self.draw_column(dc_x, dc_yl, dc_yh, &column_data[offset..]);

                offset += length as usize;
            } else {
                offset += 1 + length as usize;
            }

            offset += 1; // Skip padding byte after column data
        }
    }

    /// Draw a single visible sprite
    pub fn draw_vissprite(&mut self, vis_index: usize) {
        let vis = self.vissprites[vis_index].clone();

        // Set up drawing state
        self.spryscale = vis.scale;
        self.sprtopscreen = self.centeryfrac - self.fixed_mul(vis.texturemid, self.spryscale);

        // In a real implementation, we would:
        // 1. Load the patch from WAD
        // 2. For each column from x1 to x2
        // 3. Get the column data
        // 4. Draw it with draw_masked_column

        // Pseudo-code:
        // let patch = load_patch(vis.patch);
        // let mut frac = vis.startfrac;
        // for dc_x in vis.x1..=vis.x2 {
        //     let texturecolumn = (frac >> FRACBITS) as usize;
        //     let column = get_patch_column(patch, texturecolumn);
        //     self.draw_masked_column(column, dc_x);
        //     frac += vis.xiscale;
        // }
    }

    /// Draw a sprite with clipping against walls
    pub fn draw_sprite(&mut self, spr_index: usize) {
        let spr = self.vissprites[spr_index].clone();

        // Initialize clip arrays
        let mut clipbot = vec![-2i16; SCREENWIDTH];
        let mut cliptop = vec![-2i16; SCREENWIDTH];

        for x in spr.x1 as usize..=spr.x2 as usize {
            clipbot[x] = -2;
            cliptop[x] = -2;
        }

        // Scan drawsegs to find clipping boundaries
        for ds in self.drawsegs.iter().rev() {
            // Check if drawseg overlaps sprite horizontally
            if ds.x2 < spr.x1 || ds.x1 > spr.x2 {
                continue;
            }

            if ds.silhouette == SIL_NONE && ds.maskedtexturecol.is_none() {
                continue;
            }

            let r1 = max(ds.x1, spr.x1);
            let r2 = min(ds.x2, spr.x2);

            let (scale, lowscale) = if ds.scale1 > ds.scale2 {
                (ds.scale1, ds.scale2)
            } else {
                (ds.scale2, ds.scale1)
            };

            // Check if sprite is in front of this seg
            if scale < spr.scale ||
               (lowscale < spr.scale && !self.point_on_seg_side(spr.gx, spr.gy)) {
                // Seg is behind sprite
                continue;
            }

            // Clip sprite to this seg
            let mut silhouette = ds.silhouette;

            if spr.gz >= ds.bsilheight {
                silhouette &= !SIL_BOTTOM;
            }
            if spr.gzt <= ds.tsilheight {
                silhouette &= !SIL_TOP;
            }

            if silhouette & SIL_BOTTOM != 0 {
                for x in r1 as usize..=r2 as usize {
                    if clipbot[x] == -2 {
                        clipbot[x] = ds.sprbottomclip[x];
                    }
                }
            }

            if silhouette & SIL_TOP != 0 {
                for x in r1 as usize..=r2 as usize {
                    if cliptop[x] == -2 {
                        cliptop[x] = ds.sprtopclip[x];
                    }
                }
            }
        }

        // Fill in unclipped columns
        for x in spr.x1 as usize..=spr.x2 as usize {
            if clipbot[x] == -2 {
                clipbot[x] = SCREENHEIGHT as i16;
            }
            if cliptop[x] == -2 {
                cliptop[x] = -1;
            }
        }

        // Set clip arrays and draw
        self.mfloorclip = clipbot;
        self.mceilingclip = cliptop;
        self.draw_vissprite(spr_index);
    }

    /// Main entry point - draw all masked elements (sprites + masked walls)
    pub fn draw_masked(&mut self) {
        // Sort sprites by distance
        self.sort_vissprites();

        // Draw all sprites back to front
        for &spr_index in &self.sorted_sprites {
            self.draw_sprite(spr_index);
        }

        // Draw any remaining masked mid textures
        // (In real implementation, would iterate drawsegs)
    }

    //-------------------------------------------------------------------------
    // Player Weapon Sprites (psprites)
    //-------------------------------------------------------------------------

    /// Draw player weapon sprites (always on top)
    pub fn draw_player_sprites(&mut self) {
        // Set up clipping for psprites
        self.mfloorclip = self.screenheightarray.to_vec();
        self.mceilingclip = self.negonearray.to_vec();

        // In real implementation:
        // - Get player's psprite states
        // - For each active psprite (weapon, flash, etc)
        // - Calculate screen position
        // - Draw with draw_vissprite
    }

    //-------------------------------------------------------------------------
    // Helper Functions
    //-------------------------------------------------------------------------

    fn new_vissprite(&mut self) -> usize {
        if self.vissprite_count >= MAXVISSPRITES {
            return MAXVISSPRITES; // Overflow
        }

        let index = self.vissprite_count;
        self.vissprite_count += 1;
        index
    }

    fn fixed_mul(&self, a: Fixed, b: Fixed) -> Fixed {
        ((a as i64 * b as i64) >> FRACBITS) as Fixed
    }

    fn fixed_div(&self, a: Fixed, b: Fixed) -> Fixed {
        if b == 0 {
            return if a >= 0 { i32::MAX } else { i32::MIN };
        }
        ((a as i64) << FRACBITS) / (b as i64) as Fixed
    }

    fn point_to_angle(&self, x: Fixed, y: Fixed) -> Angle {
        // Simplified angle calculation
        // In real implementation, use atan2 table
        let dx = x - self.viewx;
        let dy = y - self.viewy;

        // This is a placeholder - real DOOM uses a lookup table
        let angle_rad = (dy as f64).atan2(dx as f64);
        (angle_rad * 2147483648.0 / std::f64::consts::PI) as Angle
    }

    fn point_on_seg_side(&self, x: Fixed, y: Fixed) -> bool {
        // Simplified side check
        // In real implementation, checks which side of a line segment
        false // Placeholder
    }
}

//-----------------------------------------------------------------------------
// Public API
//-----------------------------------------------------------------------------

impl ThingRenderer {
    /// Initialize at program start
    pub fn init(&mut self, sprite_names: Vec<String>) {
        self.init_sprites(sprite_names);
    }

    /// Clear at frame start
    pub fn begin_frame(&mut self) {
        self.clear_sprites();
    }

    /// Project and add sprites from a sector
    pub fn add_sector_sprites(&mut self, sector: &Sector) {
        self.add_sprites(sector);
    }

    /// Render all sprites (called after walls/floors)
    pub fn render(&mut self) {
        self.draw_masked();
    }

    /// Set view parameters
    pub fn set_view(&mut self, x: Fixed, y: Fixed, z: Fixed, angle: Angle) {
        self.viewx = x;
        self.viewy = y;
        self.viewz = z;
        self.viewangle = angle;

        // Calculate view cos/sin
        // In real implementation, use lookup table
        let angle_rad = (angle as f64) * std::f64::consts::PI / 2147483648.0;
        self.viewcos = (angle_rad.cos() * FRACUNIT as f64) as Fixed;
        self.viewsin = (angle_rad.sin() * FRACUNIT as f64) as Fixed;
    }
}

//-----------------------------------------------------------------------------
// Tests
//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vissprite_creation() {
        let vis = VisSprite::new();
        assert_eq!(vis.x1, 0);
        assert_eq!(vis.x2, 0);
        assert_eq!(vis.scale, 0);
    }

    #[test]
    fn test_renderer_init() {
        let mut renderer = ThingRenderer::new();
        renderer.init(vec![]);
        assert_eq!(renderer.vissprite_count, 0);
    }

    #[test]
    fn test_fixed_mul() {
        let renderer = ThingRenderer::new();
        let result = renderer.fixed_mul(FRACUNIT * 2, FRACUNIT * 3);
        assert_eq!(result, FRACUNIT * 6);
    }

    #[test]
    fn test_fixed_div() {
        let renderer = ThingRenderer::new();
        let result = renderer.fixed_div(FRACUNIT * 6, FRACUNIT * 2);
        assert_eq!(result, FRACUNIT * 3);
    }
}
