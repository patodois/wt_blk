use crate::dxp_and_grp::error::{DxpGrpError, DxpGrpError::IndexingFileOutOfBounds};

/// Bounds-checked little-endian reader over a byte slice
#[derive(Clone)]
pub(crate) struct Cursor<'a> {
	buf: &'a [u8],
	pos: usize,
}

impl<'a> Cursor<'a> {
	pub(crate) fn new(buf: &'a [u8]) -> Self {
		Self { buf, pos: 0 }
	}

	pub(crate) fn at(buf: &'a [u8], pos: usize) -> Result<Self, DxpGrpError> {
		if pos > buf.len() {
			return Err(IndexingFileOutOfBounds {
				current_ptr: pos,
				file_size:   buf.len(),
			});
		}
		Ok(Self { buf, pos })
	}

	pub(crate) fn pos(&self) -> usize {
		self.pos
	}

	pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8], DxpGrpError> {
		let out = self
			.pos
			.checked_add(n)
			.and_then(|end| self.buf.get(self.pos..end))
			.ok_or(IndexingFileOutOfBounds {
				current_ptr: self.pos.saturating_add(n),
				file_size:   self.buf.len(),
			})?;
		self.pos += n;
		Ok(out)
	}

	pub(crate) fn rest(&mut self) -> &'a [u8] {
		let out = &self.buf[self.pos..];
		self.pos = self.buf.len();
		out
	}

	pub(crate) fn skip(&mut self, n: usize) -> Result<(), DxpGrpError> {
		self.bytes(n).map(|_| ())
	}

	pub(crate) fn u16(&mut self) -> Result<u16, DxpGrpError> {
		let b = self.bytes(2)?;
		Ok(u16::from_le_bytes([b[0], b[1]]))
	}

	pub(crate) fn u32(&mut self) -> Result<u32, DxpGrpError> {
		let b = self.bytes(4)?;
		Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
	}

	/// 32-bit offset or size stored in a 64-bit field (Dagor's patchable pointers in dumps)
	pub(crate) fn u64_usize(&mut self) -> Result<usize, DxpGrpError> {
		let b = self.bytes(8)?;
		let v = u64::from_le_bytes(b.try_into().expect("Infallible"));
		usize::try_from(v)
			.map_err(|_| DxpGrpError::Model(format!("offset {v} does not fit in memory")))
	}

	pub(crate) fn f32(&mut self) -> Result<f32, DxpGrpError> {
		Ok(f32::from_bits(self.u32()?))
	}

	/// NUL-terminated string
	pub(crate) fn cstr(&mut self) -> Result<String, DxpGrpError> {
		let rest = &self.buf[self.pos..];
		let len = rest
			.iter()
			.position(|&b| b == 0)
			.ok_or(DxpGrpError::Model("unterminated string".to_owned()))?;
		let s = String::from_utf8_lossy(&rest[..len]).into_owned();
		self.pos += len + 1;
		Ok(s)
	}
}

/// Checks that a table of `count` records of `size` bytes at `ofs` lies inside `buf`, before a count
/// read from a file sizes an allocation or a loop. Empty tables may point anywhere
pub(crate) fn check_table(
	buf: &[u8],
	ofs: usize,
	count: usize,
	size: usize,
) -> Result<(), DxpGrpError> {
	let end = count.checked_mul(size).and_then(|n| n.checked_add(ofs));
	match end {
		_ if count == 0 => Ok(()),
		Some(end) if end <= buf.len() => Ok(()),
		_ => Err(IndexingFileOutOfBounds {
			current_ptr: end.unwrap_or(usize::MAX),
			file_size:   buf.len(),
		}),
	}
}
