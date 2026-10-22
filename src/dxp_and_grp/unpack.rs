//! Decompression with output sizes taken from file headers, which may lie

use std::io::Read;

use crate::dxp_and_grp::error::DxpGrpError;

/// Oodle spends at least 6 bytes on each 256 KiB block it unpacks: a block header, then a quantum
/// header and the byte a memset quantum fills with
const OODLE_BLOCK: usize = 0x4_0000;
const OODLE_MIN_BLOCK_BYTES: usize = 6;

/// Unpacks an Oodle stream to exactly `size` bytes. Oodle needs the whole output buffer up front, so
/// sizes the stream cannot reach are rejected before allocating it
pub(crate) fn oodle(packed: &[u8], size: usize) -> Result<Vec<u8>, DxpGrpError> {
	if size.div_ceil(OODLE_BLOCK) > packed.len() / OODLE_MIN_BLOCK_BYTES {
		return Err(DxpGrpError::Decompression(format!(
			"{} bytes of Oodle data cannot unpack to {size} bytes",
			packed.len()
		)));
	}
	let mut out = vec![0_u8; size];
	let written = oozextract::Extractor::new()
		.read_from_slice(packed, &mut out)
		.map_err(|e| DxpGrpError::Decompression(format!("{e:?}")))?;
	if written != size {
		return Err(DxpGrpError::Decompression(format!(
			"Oodle produced {written} bytes, expected {size}"
		)));
	}
	Ok(out)
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
