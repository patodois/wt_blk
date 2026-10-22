//! Decoder for meshoptimizer's index sequence codec (`meshopt_decodeIndexSequence`, MIT licensed),
//! which Dagor uses for packed index buffers (`VDATA_PACKED_IB`).
//!
//! Layout: one header byte `0xD0 | version`, then one variable-length integer per index, then a
//! 4-byte tail. Each integer holds, from the lowest bit up: which of the two previous indices it is a
//! delta to, then the zigzag-encoded delta.

use crate::dxp_and_grp::error::DxpGrpError;

const SEQUENCE_HEADER: u8 = 0xD0;
const MAX_VERSION: u8 = 1;

/// Decodes `count` indices; `wide` selects 32-bit indices, otherwise they wrap to 16 bits
pub fn decode_index_sequence(
	data: &[u8],
	count: usize,
	wide: bool,
) -> Result<Vec<u32>, DxpGrpError> {
	let err = |what: &str| DxpGrpError::Model(format!("packed index buffer: {what}"));
	// At least one byte per index, between the header byte and the tail
	if data.len() < 1 + 4 || data.len() - (1 + 4) < count {
		return Err(err("too short"));
	}
	if data[0] & 0xF0 != SEQUENCE_HEADER || data[0] & 0x0F > MAX_VERSION {
		return Err(err("unknown header"));
	}

	let end = data.len() - 4;
	let mut pos = 1;
	let mut last = [0_u32; 2];
	let mut out = Vec::with_capacity(count);
	for _ in 0..count {
		if pos >= end {
			return Err(err("truncated"));
		}
		// Variable-length integer: 7 bits per byte, up to 5 bytes; the tail keeps reads in bounds
		let mut v = u32::from(data[pos] & 0x7F);
		let mut more = data[pos] & 0x80 != 0;
		pos += 1;
		let mut shift = 7;
		while more && shift < 35 {
			let byte = data[pos];
			pos += 1;
			v |= u32::from(byte & 0x7F) << shift;
			shift += 7;
			more = byte & 0x80 != 0;
		}

		let baseline = (v & 1) as usize;
		let v = v >> 1;
		let delta = (v >> 1) ^ (v & 1).wrapping_neg();
		let index = last[baseline].wrapping_add(delta);
		last[baseline] = index;
		out.push(if wide { index } else { index & 0xFFFF });
	}
	if pos != end {
		return Err(err("trailing data"));
	}
	Ok(out)
}

#[cfg(test)]
mod test {
	use super::*;

	/// Encodes like meshopt_encodeIndexSequence, to round-trip against the decoder
	fn encode(indices: &[u32]) -> Vec<u8> {
		let mut out = vec![SEQUENCE_HEADER | 1];
		let mut last = [0_u32; 2];
		for &index in indices {
			// Prefer the baseline closest to the index, as the encoder does
			let d0 = index.wrapping_sub(last[0]) as i32;
			let d1 = index.wrapping_sub(last[1]) as i32;
			let current = usize::from(d1.unsigned_abs() < d0.unsigned_abs());
			let d = index.wrapping_sub(last[current]) as i32;
			let zigzag = ((d << 1) ^ (d >> 31)) as u32;
			let mut v = (zigzag << 1) | current as u32;
			loop {
				let byte = (v & 0x7F) as u8;
				v >>= 7;
				if v == 0 {
					out.push(byte);
					break;
				}
				out.push(byte | 0x80);
			}
			last[current] = index;
		}
		out.extend_from_slice(&[0; 4]);
		out
	}

	#[test]
	fn round_trip() {
		let indices = [0, 1, 2, 2, 1, 3, 100, 5, 70_000, 69_999, 4, 0];
		let encoded = encode(&indices);
		assert_eq!(
			decode_index_sequence(&encoded, indices.len(), true).unwrap(),
			indices
		);
		let narrow: Vec<u32> = indices.iter().map(|i| i & 0xFFFF).collect();
		assert_eq!(
			decode_index_sequence(&encoded, indices.len(), false).unwrap(),
			narrow
		);
	}

	#[test]
	fn rejects_bad_input() {
		let encoded = encode(&[0, 1, 2]);
		assert!(decode_index_sequence(&encoded[..encoded.len() - 1], 3, true).is_err());
		assert!(decode_index_sequence(&encoded, 2, true).is_err());
		let mut bad_header = encoded.clone();
		bad_header[0] = 0xE1;
		assert!(decode_index_sequence(&bad_header, 3, true).is_err());
	}

	#[test]
	fn huge_counts_are_rejected() {
		// The length check used to overflow on `1 + count + 4`
		let encoded = encode(&[0, 1, 2]);
		for count in [usize::MAX, usize::MAX - 4, usize::MAX / 2] {
			assert!(decode_index_sequence(&encoded, count, true).is_err());
		}
		assert!(decode_index_sequence(&[], 0, true).is_err());
	}
}
