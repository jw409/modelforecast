// WAD File Parser for DOOM
// Based on linuxdoom-1.10/w_wad.c
//
// WAD (Where's All the Data) is DOOM's archive format containing
// game assets (levels, textures, sprites, sounds, etc.)

use std::collections::HashMap;

/// Errors that can occur when loading a WAD file
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WadError {
    /// WAD file is too small to contain a valid header
    TooSmall,
    /// Invalid magic bytes (must be "IWAD" or "PWAD")
    InvalidMagic,
    /// Lump directory offset is out of bounds
    InvalidDirectory,
    /// Lump data extends beyond the file
    InvalidLump,
}

impl std::fmt::Display for WadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WadError::TooSmall => write!(f, "WAD file too small"),
            WadError::InvalidMagic => write!(f, "Invalid WAD magic (must be IWAD or PWAD)"),
            WadError::InvalidDirectory => write!(f, "Lump directory offset out of bounds"),
            WadError::InvalidLump => write!(f, "Lump data extends beyond file"),
        }
    }
}

impl std::error::Error for WadError {}

/// Information about a single lump in the WAD
#[derive(Debug, Clone, PartialEq)]
struct LumpInfo {
    /// 8-byte name (ASCII, may be null-padded, not null-terminated)
    name: [u8; 8],
    /// Offset from start of file
    offset: usize,
    /// Size in bytes
    size: usize,
}

/// A loaded WAD file with indexed lumps
#[derive(Debug, Clone, PartialEq)]
pub struct Wad {
    /// Complete WAD file data
    data: Vec<u8>,
    /// Lump directory (indexed by lump number)
    lumps: Vec<LumpInfo>,
    /// Name lookup cache (uppercase name -> lump index)
    /// Stores the LAST occurrence of each name (DOOM searches backwards)
    name_cache: HashMap<[u8; 8], usize>,
}

impl Wad {
    /// Load a WAD file from a byte slice
    ///
    /// # Arguments
    /// * `data` - Complete WAD file contents
    ///
    /// # Returns
    /// * `Ok(Wad)` - Successfully parsed WAD
    /// * `Err(WadError)` - Parse error
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, WadError> {
        // WAD header is 12 bytes: magic(4) + numlumps(4) + diroffset(4)
        if data.len() < 12 {
            return Err(WadError::TooSmall);
        }

        // Parse header
        let magic = &data[0..4];
        if magic != b"IWAD" && magic != b"PWAD" {
            return Err(WadError::InvalidMagic);
        }

        let num_lumps = i32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let dir_offset = i32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        // Validate directory offset
        let dir_size = num_lumps * 16; // Each filelump_t is 16 bytes
        if dir_offset + dir_size > data.len() {
            return Err(WadError::InvalidDirectory);
        }

        // Parse lump directory
        let mut lumps = Vec::with_capacity(num_lumps);
        let mut name_cache = HashMap::new();

        for i in 0..num_lumps {
            let entry_offset = dir_offset + i * 16;

            // Parse filelump_t: filepos(4) + size(4) + name(8)
            let filepos = i32::from_le_bytes([
                data[entry_offset],
                data[entry_offset + 1],
                data[entry_offset + 2],
                data[entry_offset + 3],
            ]) as usize;

            let size = i32::from_le_bytes([
                data[entry_offset + 4],
                data[entry_offset + 5],
                data[entry_offset + 6],
                data[entry_offset + 7],
            ]) as usize;

            let mut name = [0u8; 8];
            name.copy_from_slice(&data[entry_offset + 8..entry_offset + 16]);

            // Validate lump bounds
            if filepos + size > data.len() {
                return Err(WadError::InvalidLump);
            }

            let lump = LumpInfo {
                name,
                offset: filepos,
                size,
            };

            // Build name cache (uppercase, last occurrence wins)
            let upper_name = uppercase_name(&name);
            name_cache.insert(upper_name, i);

            lumps.push(lump);
        }

        Ok(Wad {
            data,
            lumps,
            name_cache,
        })
    }

    /// Get the total number of lumps
    pub fn num_lumps(&self) -> usize {
        self.lumps.len()
    }

    /// Get a lump by name (case-insensitive)
    ///
    /// Returns the LAST lump with this name (DOOM behavior: later files override)
    ///
    /// # Arguments
    /// * `name` - Lump name (up to 8 ASCII characters, case-insensitive)
    ///
    /// # Returns
    /// * `Some(&[u8])` - Lump data slice
    /// * `None` - Lump not found
    pub fn get_lump(&self, name: &str) -> Option<&[u8]> {
        // Convert name to uppercase 8-byte array
        let mut name_bytes = [0u8; 8];
        for (i, &byte) in name.as_bytes().iter().enumerate().take(8) {
            name_bytes[i] = byte.to_ascii_uppercase();
        }

        // Look up in cache
        let index = *self.name_cache.get(&name_bytes)?;
        self.get_lump_by_num(index)
    }

    /// Get a lump by index
    ///
    /// # Arguments
    /// * `num` - Lump index (0-based)
    ///
    /// # Returns
    /// * `Some(&[u8])` - Lump data slice
    /// * `None` - Index out of bounds
    pub fn get_lump_by_num(&self, num: usize) -> Option<&[u8]> {
        let lump = self.lumps.get(num)?;
        Some(&self.data[lump.offset..lump.offset + lump.size])
    }

    /// Get lump size by name
    ///
    /// # Arguments
    /// * `name` - Lump name (up to 8 ASCII characters, case-insensitive)
    ///
    /// # Returns
    /// * `Some(usize)` - Lump size in bytes
    /// * `None` - Lump not found
    pub fn get_lump_size(&self, name: &str) -> Option<usize> {
        let mut name_bytes = [0u8; 8];
        for (i, &byte) in name.as_bytes().iter().enumerate().take(8) {
            name_bytes[i] = byte.to_ascii_uppercase();
        }

        let index = *self.name_cache.get(&name_bytes)?;
        Some(self.lumps[index].size)
    }

    /// Get lump size by index
    ///
    /// # Arguments
    /// * `num` - Lump index (0-based)
    ///
    /// # Returns
    /// * `Some(usize)` - Lump size in bytes
    /// * `None` - Index out of bounds
    pub fn get_lump_size_by_num(&self, num: usize) -> Option<usize> {
        Some(self.lumps.get(num)?.size)
    }

    /// Find lump index by name (case-insensitive)
    ///
    /// Returns the LAST lump with this name (DOOM behavior)
    ///
    /// # Arguments
    /// * `name` - Lump name (up to 8 ASCII characters, case-insensitive)
    ///
    /// # Returns
    /// * `Some(usize)` - Lump index
    /// * `None` - Lump not found
    pub fn find_lump(&self, name: &str) -> Option<usize> {
        let mut name_bytes = [0u8; 8];
        for (i, &byte) in name.as_bytes().iter().enumerate().take(8) {
            name_bytes[i] = byte.to_ascii_uppercase();
        }

        self.name_cache.get(&name_bytes).copied()
    }

    /// Get lump name by index
    ///
    /// # Arguments
    /// * `num` - Lump index (0-based)
    ///
    /// # Returns
    /// * `Some(String)` - Lump name (trimmed, uppercase)
    /// * `None` - Index out of bounds
    pub fn get_lump_name(&self, num: usize) -> Option<String> {
        let lump = self.lumps.get(num)?;

        // Find the actual length (stop at first null byte)
        let len = lump.name.iter().position(|&b| b == 0).unwrap_or(8);

        // Convert to UTF-8 string (DOOM names are ASCII)
        String::from_utf8(lump.name[..len].to_vec()).ok()
    }

    /// Iterate over all lumps with their names
    pub fn iter_lumps(&self) -> impl Iterator<Item = (usize, String, &[u8])> {
        self.lumps.iter().enumerate().filter_map(|(idx, lump)| {
            let len = lump.name.iter().position(|&b| b == 0).unwrap_or(8);
            let name = String::from_utf8(lump.name[..len].to_vec()).ok()?;
            let data = &self.data[lump.offset..lump.offset + lump.size];
            Some((idx, name, data))
        })
    }
}

/// Convert a name to uppercase (ASCII only)
fn uppercase_name(name: &[u8; 8]) -> [u8; 8] {
    let mut result = [0u8; 8];
    for (i, &byte) in name.iter().enumerate() {
        result[i] = byte.to_ascii_uppercase();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_wad() -> Vec<u8> {
        let mut wad = Vec::new();

        // Header: "IWAD" + 2 lumps + directory at offset 28
        wad.extend_from_slice(b"IWAD");
        wad.extend_from_slice(&2i32.to_le_bytes()); // numlumps
        wad.extend_from_slice(&28i32.to_le_bytes()); // infotableofs

        // Lump 0 data: "HELLO" at offset 12
        wad.extend_from_slice(b"HELLO");

        // Lump 1 data: "WORLD" at offset 17
        wad.extend_from_slice(b"WORLD");

        // Padding to reach offset 28
        wad.extend_from_slice(&[0, 0, 0, 0, 0, 0]);

        // Directory starts at offset 28
        // Lump 0: filepos=12, size=5, name="TEST1"
        wad.extend_from_slice(&12i32.to_le_bytes());
        wad.extend_from_slice(&5i32.to_le_bytes());
        wad.extend_from_slice(b"TEST1\0\0\0");

        // Lump 1: filepos=17, size=5, name="TEST2"
        wad.extend_from_slice(&17i32.to_le_bytes());
        wad.extend_from_slice(&5i32.to_le_bytes());
        wad.extend_from_slice(b"TEST2\0\0\0");

        wad
    }

    #[test]
    fn test_load_valid_wad() {
        let data = create_test_wad();
        let wad = Wad::from_bytes(data).expect("Failed to load WAD");
        assert_eq!(wad.num_lumps(), 2);
    }

    #[test]
    fn test_get_lump_by_name() {
        let data = create_test_wad();
        let wad = Wad::from_bytes(data).expect("Failed to load WAD");

        let lump1 = wad.get_lump("TEST1").expect("Lump TEST1 not found");
        assert_eq!(lump1, b"HELLO");

        let lump2 = wad.get_lump("test2").expect("Lump test2 not found"); // case insensitive
        assert_eq!(lump2, b"WORLD");
    }

    #[test]
    fn test_get_lump_by_num() {
        let data = create_test_wad();
        let wad = Wad::from_bytes(data).expect("Failed to load WAD");

        let lump0 = wad.get_lump_by_num(0).expect("Lump 0 not found");
        assert_eq!(lump0, b"HELLO");

        let lump1 = wad.get_lump_by_num(1).expect("Lump 1 not found");
        assert_eq!(lump1, b"WORLD");
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = create_test_wad();
        data[0..4].copy_from_slice(b"NOPE");

        let result = Wad::from_bytes(data);
        assert_eq!(result, Err(WadError::InvalidMagic));
    }

    #[test]
    fn test_too_small() {
        let data = vec![0, 1, 2, 3, 4]; // Less than 12 bytes
        let result = Wad::from_bytes(data);
        assert_eq!(result, Err(WadError::TooSmall));
    }

    #[test]
    fn test_get_lump_name() {
        let data = create_test_wad();
        let wad = Wad::from_bytes(data).expect("Failed to load WAD");

        assert_eq!(wad.get_lump_name(0), Some("TEST1".to_string()));
        assert_eq!(wad.get_lump_name(1), Some("TEST2".to_string()));
        assert_eq!(wad.get_lump_name(999), None);
    }

    #[test]
    fn test_find_lump() {
        let data = create_test_wad();
        let wad = Wad::from_bytes(data).expect("Failed to load WAD");

        assert_eq!(wad.find_lump("TEST1"), Some(0));
        assert_eq!(wad.find_lump("test2"), Some(1)); // case insensitive
        assert_eq!(wad.find_lump("NOTFOUND"), None);
    }

    #[test]
    fn test_lump_size() {
        let data = create_test_wad();
        let wad = Wad::from_bytes(data).expect("Failed to load WAD");

        assert_eq!(wad.get_lump_size("TEST1"), Some(5));
        assert_eq!(wad.get_lump_size_by_num(1), Some(5));
        assert_eq!(wad.get_lump_size("NOTFOUND"), None);
    }
}
