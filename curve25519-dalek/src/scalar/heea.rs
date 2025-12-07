//! Implementation of the paper "Accelerating EdDSA Signature Verification
//! with Faster Scalar Size Halving" (TCHES 2025).
//!
//! This module implements Algorithm 4 (hEEA_approx_q) from the paper, which generates
//! half-size scalars for faster EdDSA verification.
//!
//! For verification sB = R + hA, we find rho, tau such that rho = tau*h (mod ell)
use core::ops::Neg;

use crate::constants;

/// A signed 256-bit integer represented as 4 u64 limbs (little-endian)
/// Used for the half-extended Euclidean algorithm
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct I256 {
    /// Limbs in little-endian order: [low, ..., high]
    limbs: [u64; 4],
    /// Sign: true = negative, false = non-negative
    negative: bool,
}

pub(crate) const HEEA_MAX_INDEX: usize = 129;

/// Implement curve25519_hEEA_vartime algorithm
/// Returns (rho, tau) such that rho ≡ tau * v (mod L)
/// where L is the Ed25519 group order (2^252 + 27742317777372353535851937790883648493)
pub(crate) fn curve25519_heea_vartime(v: I256) -> (I256, I256) {
    // Get L from the existing BASEPOINT_ORDER constant
    let mut r0: I256 = (&constants::BASEPOINT_ORDER).into();
    let mut r1 = v;

    debug_assert!(r1.abs() < r0, "Input v must be less than the group order L");

    let mut t0: I256 = I256::ZERO;
    let mut t1: I256 = I256::ONE;

    let mut bl_r0 = 253u32; // bit_length(L) = 253
    let mut bl_r1 = bit_length_i256(r1);

    // Main loop - continue until r1 is approximately half-size (~127 bits)
    while bl_r1 > 127 {
        // Compute shift amount
        let s = bl_r0 - bl_r1;

        // Check if signs are the same (cheap flag check)
        let sign_r0 = r0.is_negative();
        let sign_r1 = r1.is_negative();

        let (r, t) = if sign_r0 == sign_r1 {
            (r0.wrapping_sub(r1 << s), t0.wrapping_sub(t1 << s))
        } else {
            (r0.wrapping_add(r1 << s), t0.wrapping_add(t1 << s))
        };

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

    (r1, t1)
}

/// Compute bit length of I256 (magnitude, not including sign)
#[inline(always)]
fn bit_length_i256(v: I256) -> u32 {
    for i in (0..4).rev() {
        let limb = v.limbs[i];
        if limb != 0 {
            return (i as u32) * 64 + (64 - limb.leading_zeros());
        }
    }

    0
}

impl I256 {
    pub(crate) const ZERO: Self = I256 {
        limbs: [0, 0, 0, 0],
        negative: false,
    };

    const ONE: Self = I256 {
        limbs: [1, 0, 0, 0],
        negative: false,
    };

    #[inline(always)]
    const fn abs(&self) -> Self {
        I256 {
            limbs: self.limbs,
            negative: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn new(a: i128) -> Self {
        let mag = a.unsigned_abs() as u128;
        I256 {
            limbs: [
                mag as u64,
                (mag >> 64) as u64,
                0,
                0,
            ],
            negative: a.is_negative(),
        }
    }

    /// Create from little-endian bytes
    pub(crate) fn from_le_bytes(bytes: [u8; 32]) -> Self {
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            let start = i * 8;
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&bytes[start..start + 8]);
            limbs[i] = u64::from_le_bytes(chunk);
        }
        I256 {
            limbs,
            negative: false,
        }
    }

    /// Convert to little-endian bytes
    pub(crate) fn to_le_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.limbs[i].to_le_bytes());
        }
        out
    }

    /// Check if zero
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.limbs[0] == 0 && self.limbs[1] == 0 && self.limbs[2] == 0 && self.limbs[3] == 0
    }

    /// Check if negative (< 0)
    #[inline(always)]
    fn is_negative(&self) -> bool {
        self.negative && !self.is_zero()
    }

    /// Wrapping negation (two's complement)
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn wrapping_sub(self, rhs: Self) -> Self {
        self.wrapping_add(rhs.wrapping_neg())
    }

    /// Left shift
    #[inline(always)]
    fn wrapping_shl(self, shift: u32) -> Self {
        if shift == 0 {
            return self;
        }
        if shift >= 256 {
            return Self::ZERO;
        }

        let limb_shift = (shift / 64) as usize;
        let bit_shift = shift & 63;
        let l = self.limbs;
        let mut result = [0u64; 4];

        if bit_shift == 0 {
            match limb_shift {
                0 => return self,
                1 => {
                    result[1] = l[0];
                    result[2] = l[1];
                    result[3] = l[2];
                }
                2 => {
                    result[2] = l[0];
                    result[3] = l[1];
                }
                3 => {
                    result[3] = l[0];
                }
                _ => {}
            }
        } else {
            let carry = 64 - bit_shift;
            match limb_shift {
                0 => {
                    result[0] = l[0] << bit_shift;
                    result[1] = (l[1] << bit_shift) | (l[0] >> carry);
                    result[2] = (l[2] << bit_shift) | (l[1] >> carry);
                    result[3] = (l[3] << bit_shift) | (l[2] >> carry);
                }
                1 => {
                    result[1] = l[0] << bit_shift;
                    result[2] = (l[1] << bit_shift) | (l[0] >> carry);
                    result[3] = (l[2] << bit_shift) | (l[1] >> carry);
                }
                2 => {
                    result[2] = l[0] << bit_shift;
                    result[3] = (l[1] << bit_shift) | (l[0] >> carry);
                }
                3 => {
                    result[3] = l[0] << bit_shift;
                }
                _ => {}
            }
        }

        I256 {
            limbs: result,
            negative: self.negative,
        }
    }
}

impl Neg for I256 {
    type Output = Self;
    fn neg(self) -> <Self as Neg>::Output {
        self.wrapping_neg()
    }
}

// Helper: Add two magnitude arrays, returns (result, overflow_occurred)
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scalar, digest::Update, traits::HEEADecomposition};

    #[cfg(feature = "rand_core")]
    use rand::RngCore;

    #[test]
    #[cfg(feature = "rand_core")]
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
            let h_i256 = (&h).into();
            let (rho_i256, tau_i256) = curve25519_heea_vartime(h_i256);

            // Check the magnitude of rho and tau in their signed representation
            let rho_magnitude_bits = bit_length_i256(rho_i256);
            let tau_magnitude_bits = bit_length_i256(tau_i256);

            // Now convert to Scalars and verify the equation
            let (rho, tau, flip) = h.heea_decompose();

            // Verify that rho = tau * h (mod ell)
            let computed_rho = tau * h;
            let computed_rho = if flip { -computed_rho } else { computed_rho };
            assert_eq!(rho, computed_rho, "rho should equal tau * h");

            // Check that they are non-zero
            assert_ne!(rho, Scalar::ZERO, "rho should be non-zero");
            assert_ne!(tau, Scalar::ZERO, "tau should be non-zero");

            // Both magnitudes should be approximately half-size (~127 bits)
            assert!(
                rho_magnitude_bits <= 127,
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

    #[test]
    fn test_heea_edge_cases() {
        // Test Zero: v = 0
        let v_zero = I256::ZERO;
        let (rho, tau) = curve25519_heea_vartime(v_zero);
        // When v = 0, the algorithm should return (0, 1) since we start with r1=0, t1=1
        // and the loop never executes (bl_r1 would be small)
        assert_eq!(rho, I256::ZERO, "rho should be zero when v is zero");
        assert_eq!(tau, I256::ONE, "tau should be 1 when v is zero");

        // Test One: v = 1
        let v_one = I256::ONE;
        let (rho, tau) = curve25519_heea_vartime(v_one);
        assert_eq!(rho, I256::ONE, "rho should be 1 when v is 1");
        assert_eq!(tau, I256::ONE, "tau should be 1 when v is 1");

        // Test Minus One: v = -1
        let v_minus_one = -I256::ONE;
        let (rho, tau) = curve25519_heea_vartime(v_minus_one);
        assert_eq!(rho, -I256::ONE, "rho should be -1 when v is -1");
        assert_eq!(tau, I256::ONE, "tau should be 1 when v is -1");

        // Test Max i128 Boundary: v = 2^127 - 1
        // This tests whether we handle values near the i128 boundary correctly
        // i128::MAX = 2^127 - 1
        let v_max_i128 = I256::new(i128::MAX);
        let (rho, tau) = curve25519_heea_vartime(v_max_i128);
        let rho_bits = bit_length_i256(rho);
        assert!(
            rho_bits <= 128,
            "rho should be half-size for 2^127-1, got {} bits",
            rho_bits
        );

        // Verify tau magnitude
        let tau_bits = bit_length_i256(tau);
        assert!(
            tau_bits <= 128,
            "tau should be approximately half-size, got {} bits",
            tau_bits
        );

        // test v = 2^252
        let v_252 = I256::ONE << 252;
        let (rho, tau) = curve25519_heea_vartime(v_252);
        let rho_bits = bit_length_i256(rho);
        assert!(
            rho_bits <= 128,
            "rho should be half-size for 2^252, got {} bits",
            rho_bits
        );
        assert_eq!(tau, -I256::ONE, "tau should be -1 for 2^252");
    }
}
