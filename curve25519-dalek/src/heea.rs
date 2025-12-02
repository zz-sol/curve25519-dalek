//! Implementation of the TCHES 2025 paper
//!
//! This module implements Algorithm 4 (hEEA_approx_q) from the paper, which generates
//! half-size scalars for faster EdDSA verification.
//!
//! For verification sB = R + hA, we find rho, tau such that rho = tau*h (mod ell)
use crate::scalar::Scalar;

/// A signed 256-bit integer represented as 4 u64 limbs (little-endian)
/// Used for the half-extended Euclidean algorithm
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct I256 {
    /// Limbs in little-endian order: [low, ..., high]
    limbs: [u64; 4],
    /// Sign: true = negative, false = non-negative
    negative: bool,
}

impl I256 {
    const ZERO: Self = I256 {
        limbs: [0, 0, 0, 0],
        negative: false,
    };

    /// Create from words (high, low as u128 each)
    const fn from_words(high: u128, low: u128) -> Self {
        I256 {
            limbs: [
                low as u64,
                (low >> 64) as u64,
                high as u64,
                (high >> 64) as u64,
            ],
            negative: false,
        }
    }

    /// Create from little-endian bytes
    fn from_le_bytes(bytes: [u8; 32]) -> Self {
        I256 {
            limbs: unsafe { core::mem::transmute::<[u8; 32], [u64; 4]>(bytes) },
            negative: false,
        }
    }

    /// Convert to little-endian bytes
    fn to_le_bytes(self) -> [u8; 32] {
        unsafe { core::mem::transmute::<[u64; 4], [u8; 32]>(self.limbs) }
    }

    /// Check if negative (< 0)
    fn is_negative(&self) -> bool {
        self.negative && !self.is_zero()
    }

    /// Check if zero
    fn is_zero(&self) -> bool {
        self.limbs[0] == 0 && self.limbs[1] == 0 && self.limbs[2] == 0 && self.limbs[3] == 0
    }

    /// Count leading zeros of the magnitude
    fn leading_zeros(&self) -> u32 {
        if self.is_zero() {
            return 256;
        }

        for i in (0..4).rev() {
            if self.limbs[i] != 0 {
                return self.limbs[i].leading_zeros() + (3 - i) as u32 * 64;
            }
        }
        256
    }

    /// Wrapping negation (two's complement)
    fn wrapping_neg(self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }
        I256 {
            limbs: self.limbs,
            negative: !self.negative,
        }
    }

    /// Wrapping addition
    fn wrapping_add(self, rhs: Self) -> Self {
        // If signs are the same, add magnitudes
        if self.negative == rhs.negative {
            let (limbs, _overflow) = add_limbs(&self.limbs, &rhs.limbs);
            I256 {
                limbs,
                negative: self.negative,
            }
        } else {
            // Different signs: subtract the smaller magnitude from larger
            let cmp = cmp_magnitude(&self.limbs, &rhs.limbs);
            match cmp {
                core::cmp::Ordering::Greater => {
                    let (limbs, _underflow) = sub_limbs(&self.limbs, &rhs.limbs);
                    I256 {
                        limbs,
                        negative: self.negative,
                    }
                }
                core::cmp::Ordering::Less => {
                    let (limbs, _underflow) = sub_limbs(&rhs.limbs, &self.limbs);
                    I256 {
                        limbs,
                        negative: rhs.negative,
                    }
                }
                core::cmp::Ordering::Equal => Self::ZERO,
            }
        }
    }

    /// Wrapping subtraction
    fn wrapping_sub(self, rhs: Self) -> Self {
        self.wrapping_add(rhs.wrapping_neg())
    }

    /// Left shift
    fn wrapping_shl(self, shift: u32) -> Self {
        if shift >= 256 {
            return Self::ZERO;
        }
        if shift == 0 {
            return self;
        }

        let limb_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;

        let mut result = [0u64; 4];

        if bit_shift == 0 {
            // Simple limb shift
            for i in limb_shift..4 {
                result[i] = self.limbs[i - limb_shift];
            }
        } else {
            // Shift with carry between limbs
            for i in limb_shift..4 {
                let src_idx = i - limb_shift;
                result[i] = self.limbs[src_idx] << bit_shift;
                if src_idx > 0 {
                    result[i] |= self.limbs[src_idx - 1] >> (64 - bit_shift);
                }
            }
        }

        I256 {
            limbs: result,
            negative: self.negative,
        }
    }
}

// Helper: Add two magnitude arrays, returns (result, overflow_occurred)
fn add_limbs(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let mut carry;

    (result[0], carry) = a[0].overflowing_add(b[0]);
    (result[1], carry) = a[1].carrying_add(b[1], carry);
    (result[2], carry) = a[2].carrying_add(b[2], carry);
    (result[3], carry) = a[3].carrying_add(b[3], carry);

    (result, carry)
}

// Helper: Subtract b from a, returns (result, underflow)
// If underflow is true, then a < b and the result is the two's complement
fn sub_limbs(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let mut borrow;

    (result[0], borrow) = a[0].overflowing_sub(b[0]);
    (result[1], borrow) = a[1].borrowing_sub(b[1], borrow);
    (result[2], borrow) = a[2].borrowing_sub(b[2], borrow);
    (result[3], borrow) = a[3].borrowing_sub(b[3], borrow);

    (result, borrow)
}

// Helper: Compare magnitudes of two limb arrays
fn cmp_magnitude(a: &[u64; 4], b: &[u64; 4]) -> core::cmp::Ordering {
    for i in (0..4).rev() {
        match a[i].cmp(&b[i]) {
            core::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    core::cmp::Ordering::Equal
}

impl PartialOrd for I256 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I256 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => cmp_magnitude(&self.limbs, &other.limbs),
            (true, true) => cmp_magnitude(&other.limbs, &self.limbs),
        }
    }
}

impl core::ops::Shl<u32> for I256 {
    type Output = Self;
    fn shl(self, rhs: u32) -> Self {
        self.wrapping_shl(rhs)
    }
}

/// Ed25519 group order L = 2^252 + 27742317777372353535851937790883648493
/// = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed
/// High 128 bits: 0x10000000000000000000000000000000
/// Low 128 bits:  0x14def9dea2f79cd65812631a5cf5d3ed
const L: I256 = I256::from_words(
    0x1000000000000000_0000000000000000,
    0x14def9dea2f79cd6_5812631a5cf5d3ed,
);

/// Implement curve25519_hEEA_vartime algorithm
/// Returns (rho, tau) such that rho ≡ tau * v (mod L)
pub(crate) fn curve25519_heea_vartime(v: I256) -> (I256, i128) {
    let mut r0 = L;
    let mut r1 = v;
    let mut t0: i128 = 0;
    let mut t1: i128 = 1;

    let mut bl_r0 = 253u32; // bit_length(L) = 253
    let mut bl_r1 = bit_length_i256(r1);

    // Main loop - continue until r1 is approximately half-size (~127 bits)
    loop {
        // Stop when r1 is small enough (half-size)
        if bl_r1 <= 127 {
            return (r1, t1);
        }

        // Compute shift amount
        let s = bl_r0 - bl_r1;

        // Perform the shift-and-add/sub operation
        let mut r = r0;
        let mut t = t0;

        // Check if signs are the same
        let sign_r0 = r0.is_negative();
        let sign_r1 = r1.is_negative();

        if sign_r0 == sign_r1 {
            // Same sign: subtract
            r = r.wrapping_sub(r1 << s);
            // For t, handle shift carefully - if s >= 128, result would overflow
            if s < 128 {
                t = t.wrapping_sub(t1.wrapping_shl(s));
            }
        } else {
            // Different sign: add
            r = r.wrapping_add(r1 << s);
            // For t, handle shift carefully - if s >= 128, result would overflow
            if s < 128 {
                t = t.wrapping_add(t1.wrapping_shl(s));
            }
        }

        let bl_r = bit_length_i256(r);

        if bl_r > bl_r1 {
            // r grew, so keep it in r0
            r0 = r;
            t0 = t;
            bl_r0 = bl_r;
        } else {
            // r shrunk, swap
            r0 = r1;
            r1 = r;
            t0 = t1;
            t1 = t;
            bl_r0 = bl_r1;
            bl_r1 = bl_r;
        }
    }
}

/// Compute bit length of I256 (magnitude, not including sign)
#[inline]
fn bit_length_i256(val: I256) -> u32 {
    if val.is_zero() {
        return 1;
    }

    256 - val.leading_zeros()
}

/// Convert I256 to Scalar
#[inline]
fn i256_to_scalar(val: I256) -> Scalar {
    if val.is_negative() {
        // For negative numbers, compute absolute value and negate
        let abs_val = val.wrapping_neg();
        let bytes = abs_val.to_le_bytes();
        let positive = Scalar::from_canonical_bytes(bytes).unwrap();
        -positive
    } else {
        let bytes = val.to_le_bytes();
        Scalar::from_canonical_bytes(bytes).unwrap()
    }
}

/// Convert i128 to Scalar (for tau)
#[inline]
fn i128_to_scalar(val: i128) -> Scalar {
    if val < 0 {
        // For negative, we want: Scalar representing (ell + val) = ell - |val|
        // Since val is negative, -val = |val|
        let abs_val = (-val) as u128;
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&abs_val.to_le_bytes());
        let positive_scalar = Scalar::from_canonical_bytes(bytes).unwrap();
        // Return -|val| mod ell, which is ell - |val|
        -positive_scalar
    } else {
        // For positive, just convert directly
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&(val as u128).to_le_bytes());
        Scalar::from_canonical_bytes(bytes).unwrap()
    }
}

/// Convert Scalar to I256
#[inline]
fn scalar_to_i256(s: &Scalar) -> I256 {
    // Scalar is always positive and less than L, so treat as unsigned
    I256::from_le_bytes(*s.as_bytes())
}

/// Generate half-size scalars (rho, tau) for a given hash value h
///
/// This function takes the hash value h from the signature verification equation
/// and produces two half-size scalars rho and tau such that rho = tau*h (mod ell).
///
/// And a flag indicating if rho is negative in its signed representation.
pub fn generate_half_size_scalars(h: &Scalar) -> (Scalar, Scalar, bool) {
    // Convert h to I256
    let v = scalar_to_i256(h);

    // Run the algorithm
    let (rho_i256, tau_i128) = curve25519_heea_vartime(v);

    // Convert back to Scalars
    let rho = i256_to_scalar(rho_i256);
    let tau = i128_to_scalar(tau_i128);

    // Check if rho is negative
    let rho_is_negative = rho_i256.is_negative();
    let tau_is_negative = tau_i128 < 0;

    let (rho, tau, flip_h) = match (rho_is_negative, tau_is_negative) {
        (true, true) => (-rho, -tau, false),
        (true, false) => (-rho, tau, true),
        (false, true) => (rho, -tau, true),
        (false, false) => (rho, tau, false),
    };

    (rho, tau, flip_h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::Update;
    use rand::RngCore;

    #[test]
    fn test_generate_half_size_scalars() {
        use rand::rng;
        use sha2::{Digest, Sha512};

        let mut rng = rng();

        // Test with multiple random scalars
        for _ in 0..1000 {
            // Generate a random scalar by hashing random bytes
            let mut random_bytes = [0u8; 64];
            for byte in &mut random_bytes {
                *byte = (rng.next_u32() & 0xff) as u8;
            }
            let h = Scalar::from_hash(Sha512::new().chain(random_bytes));

            // Convert h to I256 to see the actual output
            let h_i256 = scalar_to_i256(&h);
            let (rho_i256, tau_i128) = curve25519_heea_vartime(h_i256);

            // Check the magnitude of rho and tau in their signed representation
            let rho_magnitude_bits = bit_length_i256(rho_i256);

            // For tau (i128), compute bit length
            let tau_magnitude_bits = if tau_i128 < 0 {
                let abs_val = tau_i128.wrapping_neg();
                128 - abs_val.leading_zeros()
            } else {
                128 - tau_i128.leading_zeros()
            };

            // Now convert to Scalars and verify the equation
            let (rho, tau, flip) = generate_half_size_scalars(&h);

            // Verify that rho = tau * h (mod ell)
            let computed_rho = tau * h;
            let computed_rho = if flip { -computed_rho } else { computed_rho };
            assert_eq!(rho, computed_rho, "rho should equal tau * h");

            // Check that they are non-zero
            assert_ne!(rho, Scalar::ZERO, "rho should be non-zero");
            assert_ne!(tau, Scalar::ZERO, "tau should be non-zero");

            // Both magnitudes should be approximately half-size (~127 bits)
            assert!(
                rho_magnitude_bits <= 128,
                "rho magnitude should be approximately half-size, got {} bits",
                rho_magnitude_bits
            );
            assert!(
                tau_magnitude_bits <= 128,
                "tau magnitude should be approximately half-size, got {} bits",
                tau_magnitude_bits
            );
        }
    }
}
