// DOOM 16.16 fixed-point math

use std::ops::{Add, Sub, Mul, Div, Neg};

pub const FRACBITS: i32 = 16;
pub const FRACUNIT: i32 = 1 << FRACBITS;

pub const FINEANGLES: usize = 8192;
pub const FINEMASK: usize = FINEANGLES - 1;
pub const ANGLETOFINESHIFT: i32 = 19;

// Binary angles
pub const ANG45: u32 = 0x20000000;
pub const ANG90: u32 = 0x40000000;
pub const ANG180: u32 = 0x80000000;
pub const ANG270: u32 = 0xc0000000;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Fixed(pub i32);

impl Fixed {
    pub const ZERO: Fixed = Fixed(0);
    pub const ONE: Fixed = Fixed(FRACUNIT);

    #[inline]
    pub fn from_i32(n: i32) -> Fixed {
        Fixed(n << FRACBITS)
    }

    #[inline]
    pub fn to_i32(self) -> i32 {
        self.0 >> FRACBITS
    }

    #[inline]
    pub fn from_f32(f: f32) -> Fixed {
        Fixed((f * FRACUNIT as f32) as i32)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / FRACUNIT as f32
    }

    #[inline]
    pub fn bits(self) -> i32 {
        self.0
    }

    #[inline]
    pub fn from_bits(bits: i32) -> Fixed {
        Fixed(bits)
    }

    #[inline]
    pub fn abs(self) -> Fixed {
        Fixed(self.0.abs())
    }
}

impl Add for Fixed {
    type Output = Fixed;
    #[inline]
    fn add(self, rhs: Fixed) -> Fixed {
        Fixed(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Fixed {
    type Output = Fixed;
    #[inline]
    fn sub(self, rhs: Fixed) -> Fixed {
        Fixed(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul for Fixed {
    type Output = Fixed;
    #[inline]
    fn mul(self, rhs: Fixed) -> Fixed {
        Fixed(((self.0 as i64 * rhs.0 as i64) >> FRACBITS) as i32)
    }
}

impl Div for Fixed {
    type Output = Fixed;
    #[inline]
    fn div(self, rhs: Fixed) -> Fixed {
        if rhs.0 == 0 {
            return if self.0 >= 0 { Fixed(i32::MAX) } else { Fixed(i32::MIN) };
        }
        Fixed((((self.0 as i64) << FRACBITS) / rhs.0 as i64) as i32)
    }
}

impl Neg for Fixed {
    type Output = Fixed;
    #[inline]
    fn neg(self) -> Fixed {
        Fixed(-self.0)
    }
}

// C-style functions for compatibility
#[inline]
pub fn fixed_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> FRACBITS) as i32
}

#[inline]
pub fn fixed_div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return if a >= 0 { i32::MAX } else { i32::MIN };
    }
    (((a as i64) << FRACBITS) / b as i64) as i32
}

// Runtime sine calculation (no large static tables needed)
pub fn finesin(angle: usize) -> i32 {
    let angle = angle & FINEMASK;
    let radians = (angle as f64) * std::f64::consts::PI * 2.0 / (FINEANGLES as f64);
    (radians.sin() * FRACUNIT as f64) as i32
}

pub fn finecos(angle: usize) -> i32 {
    finesin(angle + FINEANGLES / 4)
}
