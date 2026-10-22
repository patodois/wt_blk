//! Decompression with output sizes taken from file headers, which may lie

use std::{
	io::Read,
	panic::{self, AssertUnwindSafe},
};

use crate::dxp_and_grp::error::DxpGrpError;

/// Oodle spends at least 6 bytes on each 256 KiB block it unpacks: a block header, then a quantum
/// header and the byte a memset quantum fills with
const OODLE_BLOCK: usize = 0x4_0000;
const OODLE_MIN_BLOCK_BYTES: usize = 6;
/// Output of the first decoding attempt, see [`oodle`]
const OODLE_FIRST_TRY: usize = 16 * OODLE_BLOCK;

/// Unpacks an Oodle stream to exactly `size` bytes. Oodle needs its whole output buffer up front, so
/// sizes the stream cannot reach are rejected first, and large outputs are decoded in growing
/// prefixes: a stream that breaks early then fails before the full claimed size is allocated.
/// Blocks start every 256 KiB of output, so prefixes of whole blocks decode as in the full stream
pub(crate) fn oodle(packed: &[u8], size: usize) -> Result<Vec<u8>, DxpGrpError> {
	if size.div_ceil(OODLE_BLOCK) > packed.len() / OODLE_MIN_BLOCK_BYTES {
		return Err(DxpGrpError::Decompression(format!(
			"{} bytes of Oodle data cannot unpack to {size} bytes",
			packed.len()
		)));
	}
	let mut len = size.min(OODLE_FIRST_TRY);
	loop {
		let mut out = vec![0_u8; len];
		// oozextract 0.5.5 panics on some malformed streams (slicing out of range in its LZNA and
		// Bitknit decoders, an assertion when building Huffman tables): report those as errors
		// rather than taking the caller down
		let written = panic::catch_unwind(AssertUnwindSafe(|| {
			oozextract::Extractor::new().read_from_slice(packed, &mut out)
		}))
		.map_err(|_| DxpGrpError::Decompression("Oodle decoder panicked".to_owned()))?
		.map_err(|e| DxpGrpError::Decompression(format!("{e:?}")))?;
		if written != len {
			return Err(DxpGrpError::Decompression(format!(
				"Oodle produced {written} bytes, expected {size}"
			)));
		}
		if len == size {
			return Ok(out);
		}
		len = size.min(len * 2);
	}
}

/// Unpacks zstd data of at most `max` bytes, growing the output as data comes out instead of
/// allocating `max` bytes up front
pub(crate) fn zstd(packed: &[u8], max: usize) -> Result<Vec<u8>, DxpGrpError> {
	let mut out = Vec::new();
	zstd::stream::read::Decoder::new(packed)
		.and_then(|d| d.take(max as u64 + 1).read_to_end(&mut out))
		.map_err(|e| DxpGrpError::Decompression(e.to_string()))?;
	if out.len() > max {
		return Err(DxpGrpError::Decompression(format!(
			"zstd data unpacks to more than {max} bytes"
		)));
	}
	Ok(out)
}

#[cfg(test)]
mod test {
	use super::*;

	/// Oodle stream of stored (uncompressed) Kraken blocks
	fn stored_blocks(data: &[u8]) -> Vec<u8> {
		data.chunks(OODLE_BLOCK)
			.flat_map(|block| [&[0x4C, 0x06][..], block].concat())
			.collect()
	}

	#[test]
	fn oodle_output_grows_with_the_stream() {
		// 20 blocks: decoded as a 16-block prefix first, then in full
		let data: Vec<u8> = (0..20 * OODLE_BLOCK + 100).map(|i| (i / 7) as u8).collect();
		let packed = stored_blocks(&data);
		assert_eq!(oodle(&packed, data.len()).unwrap(), data);
		assert_eq!(oodle(&packed, 3).unwrap(), data[..3]);

		// A stream that ends after 17 blocks while claiming 40 fails, with a 32-block buffer at most
		let short = &packed[..17 * (OODLE_BLOCK + 2)];
		assert!(oodle(short, 40 * OODLE_BLOCK).is_err());
		// Sizes beyond 256 KiB per 6 bytes of stream are rejected before decoding
		assert!(oodle(&[0x4C, 0x06, 0, 0, 0, 0], OODLE_BLOCK + 1).is_err());
		assert!(oodle(&[0x8C, 0x06, 0, 0, 0, 0], u32::MAX as usize).is_err());
	}

	#[test]
	fn oodle_decoder_panics_become_errors() {
		// Streams found by fuzzing: a few header bytes, then zeros apart from some single bytes
		let stream = |len: usize, head: &[u8], pokes: &[(usize, u8)]| {
			let mut s = vec![0_u8; len];
			s[..head.len()].copy_from_slice(head);
			for &(at, b) in pokes {
				s[at] = b;
			}
			s
		};
		// LZNA: slice end out of range
		let lzna = stream(63, &[0x8C, 0x85, 0, 0x3A], &[(13, 0x91), (23, 0x85)]);
		assert!(oodle(&lzna, 40).is_err());
		// Bitknit: subtraction overflow (debug builds), then slice end out of range
		let bitknit = stream(70, &[0x8C, 0x8B, 0, 0x41, 0, 0, 0, 1], &[(33, 0xFF)]);
		assert!(oodle(&bitknit, 262_184).is_err());
		// Mermaid: assertion failed while building a Huffman table
		let mermaid = stream(
			218,
			&[0x0C, 0x0A, 0, 0, 0xD4, 0x40, 0x30, 0, 0, 0x80, 0x3F],
			&[],
		);
		assert!(oodle(&mermaid, 46_376).is_err());
	}
}
