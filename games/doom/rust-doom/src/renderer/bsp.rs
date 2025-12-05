//! BSP tree traversal for the DOOM renderer.
//!
//! This module handles the Binary Space Partitioning tree traversal that determines
//! visible walls in front-to-back order. The BSP tree divides the level geometry
//! into convex subsectors, and we traverse it recursively based on the view position.
//!
//! Key concepts:
//! - Nodes: Interior nodes with partition lines and bounding boxes
//! - Subsectors: Leaf nodes containing lists of segs (wall segments)
//! - Segs: Wall segments that reference linedefs, sidedefs, and sectors
//! - Clip ranges: Track which screen columns have been drawn (solidsegs)
//!
//! Reference: r_bsp.c from linuxdoom-1.10

// Fixed-point math (16.16 format)
pub type Fixed = i32;
pub const FRACBITS: i32 = 16;
pub const FRACUNIT: Fixed = 1 << FRACBITS;

// Binary Angle Measurement (BAM) - unsigned 32-bit angles
pub type Angle = u32;
pub const ANG45: Angle = 0x20000000;
pub const ANG90: Angle = 0x40000000;
pub const ANG180: Angle = 0x80000000;
pub const ANG270: Angle = 0xc0000000;

// Fine angle constants
pub const FINEANGLES: usize = 8192;
pub const FINEMASK: usize = FINEANGLES - 1;
pub const ANGLETOFINESHIFT: u32 = 19;

// Screen dimensions
pub const SCREENWIDTH: usize = 320;
pub const SCREENHEIGHT: usize = 200;

// Maximum number of draw segs and clip ranges
pub const MAXDRAWSEGS: usize = 256;
pub const MAXSEGS: usize = 32;

// BSP node flags
pub const NF_SUBSECTOR: i32 = 0x8000;

// Bounding box indices
pub const BOXTOP: usize = 0;
pub const BOXBOTTOM: usize = 1;
pub const BOXLEFT: usize = 2;
pub const BOXRIGHT: usize = 3;

/// A vertex in the map
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub x: Fixed,
    pub y: Fixed,
}

/// A sector (room/area) with floor and ceiling
#[derive(Debug, Clone)]
pub struct Sector {
    pub floor_height: Fixed,
    pub ceiling_height: Fixed,
    pub floor_pic: i16,
    pub ceiling_pic: i16,
    pub light_level: i16,
    pub special: i16,
    pub tag: i16,
}

/// A sidedef (side of a linedef)
#[derive(Debug, Clone)]
pub struct SideDef {
    pub texture_offset: Fixed,
    pub row_offset: Fixed,
    pub top_texture: i16,
    pub bottom_texture: i16,
    pub mid_texture: i16,
    pub sector: usize, // Index into sectors array
}

/// A linedef (wall definition)
#[derive(Debug, Clone)]
pub struct LineDef {
    pub v1: usize,     // Vertex index
    pub v2: usize,     // Vertex index
    pub dx: Fixed,     // Precalculated v2.x - v1.x
    pub dy: Fixed,     // Precalculated v2.y - v1.y
    pub flags: i16,
    pub special: i16,
    pub tag: i16,
    pub sidenum: [i16; 2], // [0] = front, [1] = back (-1 if one-sided)
    pub bbox: [Fixed; 4],
    pub front_sector: Option<usize>,
    pub back_sector: Option<usize>,
}

/// A seg (wall segment for rendering)
#[derive(Debug, Clone)]
pub struct Seg {
    pub v1: usize,           // Vertex index
    pub v2: usize,           // Vertex index
    pub offset: Fixed,       // Texture offset along wall
    pub angle: Angle,        // Angle for texture mapping
    pub sidedef: usize,      // SideDef index
    pub linedef: usize,      // LineDef index
    pub front_sector: usize, // Sector index
    pub back_sector: Option<usize>, // None for one-sided walls
}

/// A subsector (leaf of BSP tree)
#[derive(Debug, Clone)]
pub struct SubSector {
    pub sector: usize,   // Sector index
    pub num_lines: i16,  // Number of segs
    pub first_line: i16, // Index into segs array
}

/// A BSP node (interior node of BSP tree)
#[derive(Debug, Clone)]
pub struct Node {
    pub x: Fixed,           // Partition line start
    pub y: Fixed,
    pub dx: Fixed,          // Partition line direction
    pub dy: Fixed,
    pub bbox: [[Fixed; 4]; 2], // Bounding boxes [left child][right child]
    pub children: [u16; 2], // Child indices (MSB set = subsector)
}

/// A clip range for tracking drawn columns
#[derive(Debug, Clone, Copy)]
pub struct ClipRange {
    pub first: i32,
    pub last: i32,
}

/// A draw segment (stored info about rendered wall)
#[derive(Debug, Clone)]
pub struct DrawSeg {
    pub curline: usize,  // Seg index
    pub x1: i32,
    pub x2: i32,
    pub scale1: Fixed,
    pub scale2: Fixed,
    pub scale_step: Fixed,
    pub silhouette: i32, // 0=none, 1=bottom, 2=top, 3=both
    pub bsil_height: Fixed, // Bottom silhouette height
    pub tsil_height: Fixed, // Top silhouette height
}

/// The BSP renderer state
pub struct BspRenderer {
    // Level geometry
    pub vertices: Vec<Vertex>,
    pub sectors: Vec<Sector>,
    pub sidedefs: Vec<SideDef>,
    pub linedefs: Vec<LineDef>,
    pub segs: Vec<Seg>,
    pub subsectors: Vec<SubSector>,
    pub nodes: Vec<Node>,

    // View state
    pub view_x: Fixed,
    pub view_y: Fixed,
    pub view_z: Fixed,
    pub view_angle: Angle,
    pub view_cos: Fixed,
    pub view_sin: Fixed,
    pub clip_angle: Angle,

    // Screen projection
    pub view_angle_to_x: [i32; FINEANGLES / 2],
    pub view_width: i32,

    // Clip ranges (solidsegs)
    pub solid_segs: Vec<ClipRange>,
    pub solid_segs_end: usize,

    // Draw segments
    pub draw_segs: Vec<DrawSeg>,
    pub draw_segs_count: usize,

    // Current seg being processed
    pub cur_line: Option<usize>,
    pub front_sector: Option<usize>,
    pub back_sector: Option<usize>,

    // Lookup tables (would be loaded from tables.c)
    pub fine_sine: Vec<Fixed>,
    pub fine_tangent: Vec<Fixed>,

    // Statistics
    pub subsector_count: i32,
}

impl BspRenderer {
    /// Create a new BSP renderer
    pub fn new() -> Self {
        let mut renderer = Self {
            vertices: Vec::new(),
            sectors: Vec::new(),
            sidedefs: Vec::new(),
            linedefs: Vec::new(),
            segs: Vec::new(),
            subsectors: Vec::new(),
            nodes: Vec::new(),

            view_x: 0,
            view_y: 0,
            view_z: 0,
            view_angle: 0,
            view_cos: FRACUNIT,
            view_sin: 0,
            clip_angle: ANG90, // Default 90-degree FOV

            view_angle_to_x: [0; FINEANGLES / 2],
            view_width: SCREENWIDTH as i32,

            solid_segs: vec![ClipRange { first: 0, last: 0 }; MAXSEGS],
            solid_segs_end: 0,

            draw_segs: vec![DrawSeg {
                curline: 0,
                x1: 0,
                x2: 0,
                scale1: 0,
                scale2: 0,
                scale_step: 0,
                silhouette: 0,
                bsil_height: 0,
                tsil_height: 0,
            }; MAXDRAWSEGS],
            draw_segs_count: 0,

            cur_line: None,
            front_sector: None,
            back_sector: None,

            fine_sine: vec![0; 5 * FINEANGLES / 4],
            fine_tangent: vec![0; FINEANGLES / 2],

            subsector_count: 0,
        };

        renderer.init_tables();
        renderer
    }

    /// Initialize trigonometric lookup tables
    fn init_tables(&mut self) {
        // This would load from precomputed tables in real implementation
        // For now, just allocate space
        // In actual DOOM, these are loaded from tables.c
    }

    /// Clear clip segs for a new frame
    pub fn clear_clip_segs(&mut self) {
        // Initialize with sentinel values at edges
        self.solid_segs[0].first = -0x7fffffff;
        self.solid_segs[0].last = -1;
        self.solid_segs[1].first = self.view_width;
        self.solid_segs[1].last = 0x7fffffff;
        self.solid_segs_end = 2;
    }

    /// Clear draw segs for a new frame
    pub fn clear_draw_segs(&mut self) {
        self.draw_segs_count = 0;
    }

    /// Point-on-side test for BSP traversal
    /// Returns 0 for front, 1 for back
    pub fn point_on_side(&self, x: Fixed, y: Fixed, node: &Node) -> usize {
        if node.dx == 0 {
            if x <= node.x {
                return if node.dy > 0 { 1 } else { 0 };
            }
            return if node.dy < 0 { 1 } else { 0 };
        }

        if node.dy == 0 {
            if y <= node.y {
                return if node.dx < 0 { 1 } else { 0 };
            }
            return if node.dx > 0 { 1 } else { 0 };
        }

        let dx = x - node.x;
        let dy = y - node.y;

        // Try to quickly decide by looking at sign bits
        if ((node.dy ^ node.dx ^ dx ^ dy) & 0x80000000u32 as i32) != 0 {
            if ((node.dy ^ dx) & 0x80000000u32 as i32) != 0 {
                return 1; // Left is negative
            }
            return 0;
        }

        // Must do full cross product
        let left = (node.dy as i64 >> FRACBITS) * (dx as i64 >> FRACBITS);
        let right = (dy as i64 >> FRACBITS) * (node.dx as i64 >> FRACBITS);

        if right < left {
            return 0; // Front side
        }
        1 // Back side
    }

    /// Check if a bounding box is potentially visible
    pub fn check_bbox(&self, bbox: &[Fixed; 4]) -> bool {
        // Determine which corners of the bbox to test based on view position
        let boxx = if self.view_x <= bbox[BOXLEFT] {
            0
        } else if self.view_x < bbox[BOXRIGHT] {
            1
        } else {
            2
        };

        let boxy = if self.view_y >= bbox[BOXTOP] {
            0
        } else if self.view_y > bbox[BOXBOTTOM] {
            1
        } else {
            2
        };

        let boxpos = (boxy << 2) + boxx;

        // If viewer is inside bbox, it's visible
        if boxpos == 5 {
            return true;
        }

        // Lookup table for which corners to test
        const CHECK_COORD: [[usize; 4]; 12] = [
            [3, 0, 2, 1],
            [3, 0, 2, 0],
            [3, 1, 2, 0],
            [0, 0, 0, 0],
            [2, 0, 2, 1],
            [0, 0, 0, 0],
            [3, 1, 3, 0],
            [0, 0, 0, 0],
            [2, 0, 3, 1],
            [2, 1, 3, 1],
            [2, 1, 3, 0],
            [0, 0, 0, 0],
        ];

        let check = &CHECK_COORD[boxpos];
        let _x1 = bbox[check[0]];
        let _y1 = bbox[check[1]];
        let _x2 = bbox[check[2]];
        let _y2 = bbox[check[3]];

        // Check if the bbox corners are in view frustum
        // This is simplified - full version would use R_PointToAngle and check against clipsegs
        true // Conservative: assume visible for now
    }

    /// Render a subsector
    pub fn render_subsector(&mut self, subsector_idx: usize) {
        self.subsector_count += 1;

        let subsector = &self.subsectors[subsector_idx];
        self.front_sector = Some(subsector.sector);

        // Determine floor/ceiling planes for this sector
        // This would call into r_plane.rs in full implementation

        // Add sprites from this sector
        // This would call into r_things.rs in full implementation

        // Render each seg in the subsector
        let first_line = subsector.first_line as usize;
        let num_lines = subsector.num_lines as usize;

        for seg_idx in first_line..(first_line + num_lines) {
            self.add_line(seg_idx);
        }
    }

    /// Add a seg to the rendering list (with clipping)
    pub fn add_line(&mut self, seg_idx: usize) {
        // This is simplified - full version would:
        // 1. Calculate angles to seg endpoints
        // 2. Perform backface culling
        // 3. Clip to view frustum
        // 4. Project to screen X coordinates
        // 5. Call clip_solid_wall_segment or clip_pass_wall_segment
        // 6. Eventually call store_wall_range (in r_segs.rs)

        self.cur_line = Some(seg_idx);

        // Would implement full R_AddLine logic here
        // For now, this is a stub that shows the structure
    }

    /// Clip a solid wall segment (e.g., one-sided walls)
    pub fn clip_solid_wall_segment(&mut self, first: i32, last: i32) {
        // Find the first range that touches this range
        let mut start_idx = 0;
        while start_idx < self.solid_segs_end &&
              self.solid_segs[start_idx].last < first - 1 {
            start_idx += 1;
        }

        if start_idx >= self.solid_segs_end {
            return;
        }

        let first = first;
        let last = last;

        // Check if there's a fragment before start
        if first < self.solid_segs[start_idx].first {
            if last < self.solid_segs[start_idx].first - 1 {
                // Post is entirely visible before start
                self.store_wall_range(first, last);

                // Insert new clippost
                if self.solid_segs_end < MAXSEGS {
                    self.solid_segs.copy_within(start_idx..self.solid_segs_end, start_idx + 1);
                    self.solid_segs[start_idx] = ClipRange { first, last };
                    self.solid_segs_end += 1;
                }
                return;
            }

            // Fragment before start
            self.store_wall_range(first, self.solid_segs[start_idx].first - 1);
            self.solid_segs[start_idx].first = first;
        }

        // Bottom contained in start?
        if last <= self.solid_segs[start_idx].last {
            return;
        }

        // Handle fragments between multiple clip ranges
        let mut next_idx = start_idx;
        while next_idx + 1 < self.solid_segs_end &&
              last >= self.solid_segs[next_idx + 1].first - 1 {
            // Fragment between two posts
            self.store_wall_range(
                self.solid_segs[next_idx].last + 1,
                self.solid_segs[next_idx + 1].first - 1
            );
            next_idx += 1;

            if last <= self.solid_segs[next_idx].last {
                self.solid_segs[start_idx].last = self.solid_segs[next_idx].last;
                // Crunch: remove intermediate ranges
                if next_idx > start_idx && next_idx < self.solid_segs_end {
                    self.solid_segs.copy_within((next_idx + 1)..self.solid_segs_end, start_idx + 1);
                    self.solid_segs_end -= next_idx - start_idx;
                }
                return;
            }
        }

        // Fragment after next
        self.store_wall_range(self.solid_segs[next_idx].last + 1, last);
        self.solid_segs[start_idx].last = last;

        // Crunch
        if next_idx > start_idx && next_idx < self.solid_segs_end {
            self.solid_segs.copy_within((next_idx + 1)..self.solid_segs_end, start_idx + 1);
            self.solid_segs_end -= next_idx - start_idx;
        }
    }

    /// Clip a pass wall segment (e.g., windows with upper/lower textures)
    pub fn clip_pass_wall_segment(&mut self, first: i32, last: i32) {
        // Find the first range that touches this range
        let mut start_idx = 0;
        while start_idx < self.solid_segs_end &&
              self.solid_segs[start_idx].last < first - 1 {
            start_idx += 1;
        }

        if start_idx >= self.solid_segs_end {
            return;
        }

        let first = first;
        let last = last;

        // Check if there's a fragment before start
        if first < self.solid_segs[start_idx].first {
            if last < self.solid_segs[start_idx].first - 1 {
                // Post is entirely visible
                self.store_wall_range(first, last);
                return;
            }

            // Fragment before start
            self.store_wall_range(first, self.solid_segs[start_idx].first - 1);
        }

        // Bottom contained in start?
        if last <= self.solid_segs[start_idx].last {
            return;
        }

        // Handle fragments between clip ranges
        while start_idx + 1 < self.solid_segs_end &&
              last >= self.solid_segs[start_idx + 1].first - 1 {
            // Fragment between two posts
            self.store_wall_range(
                self.solid_segs[start_idx].last + 1,
                self.solid_segs[start_idx + 1].first - 1
            );
            start_idx += 1;

            if last <= self.solid_segs[start_idx].last {
                return;
            }
        }

        // Fragment after next
        self.store_wall_range(self.solid_segs[start_idx].last + 1, last);
    }

    /// Store a wall range for drawing
    /// This would be implemented in r_segs.rs in full implementation
    pub fn store_wall_range(&mut self, _start: i32, _stop: i32) {
        // This would:
        // 1. Calculate texture mapping
        // 2. Calculate wall scale/height
        // 3. Store in draw_segs for later rendering
        // 4. Mark floor/ceiling spans
        // 5. Set up sprite clipping

        if self.draw_segs_count < MAXDRAWSEGS {
            // Would populate draw_segs here
            self.draw_segs_count += 1;
        }
    }

    /// Recursively render BSP node
    pub fn render_bsp_node(&mut self, bsp_num: i32) {
        // Check if this is a subsector (leaf)
        if (bsp_num & NF_SUBSECTOR) != 0 {
            let subsector_idx = if bsp_num == -1 {
                0
            } else {
                (bsp_num & !NF_SUBSECTOR) as usize
            };

            if subsector_idx < self.subsectors.len() {
                self.render_subsector(subsector_idx);
            }
            return;
        }

        // Interior node
        let bsp_num = bsp_num as usize;
        if bsp_num >= self.nodes.len() {
            return;
        }

        let node = self.nodes[bsp_num].clone();

        // Decide which side the view point is on
        let side = self.point_on_side(self.view_x, self.view_y, &node);

        // Recursively divide front space (near to far)
        let front_child = node.children[side] as i32;
        self.render_bsp_node(front_child);

        // Check if back space is potentially visible
        let back_side = side ^ 1;
        if self.check_bbox(&node.bbox[back_side]) {
            let back_child = node.children[back_side] as i32;
            self.render_bsp_node(back_child);
        }
    }

    /// Main entry point: render the entire BSP tree
    pub fn render_bsp(&mut self, root_node: i32) {
        self.clear_clip_segs();
        self.clear_draw_segs();
        self.subsector_count = 0;

        self.render_bsp_node(root_node);
    }
}

impl Default for BspRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_range_initialization() {
        let renderer = BspRenderer::new();
        assert_eq!(renderer.solid_segs.len(), MAXSEGS);
        assert_eq!(renderer.draw_segs.len(), MAXDRAWSEGS);
    }

    #[test]
    fn test_clear_clip_segs() {
        let mut renderer = BspRenderer::new();
        renderer.clear_clip_segs();

        assert_eq!(renderer.solid_segs_end, 2);
        assert_eq!(renderer.solid_segs[0].last, -1);
        assert_eq!(renderer.solid_segs[1].first, SCREENWIDTH as i32);
    }

    #[test]
    fn test_point_on_side_vertical() {
        let renderer = BspRenderer::new();
        let node = Node {
            x: 100 << FRACBITS,
            y: 0,
            dx: 0,
            dy: 1 << FRACBITS,
            bbox: [[0; 4]; 2],
            children: [0; 2],
        };

        // Point to left of vertical line
        let side = renderer.point_on_side(50 << FRACBITS, 0, &node);
        assert_eq!(side, 1);

        // Point to right of vertical line
        let side = renderer.point_on_side(150 << FRACBITS, 0, &node);
        assert_eq!(side, 0);
    }

    #[test]
    fn test_subsector_flag() {
        assert_eq!(NF_SUBSECTOR, 0x8000);
        assert_eq!(0x8000 & NF_SUBSECTOR, NF_SUBSECTOR);
        assert_eq!(0x0000 & NF_SUBSECTOR, 0);
    }
}
