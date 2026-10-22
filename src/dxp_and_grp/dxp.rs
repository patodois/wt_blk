use std::{ffi::CStr, mem::size_of};

use crate::{
	blk::util::bytes_to_offset,
	dxp_and_grp::{
		cursor::{NameBudget, check_table},
		ddsx::{DdsxHeader, Texture},
		dxp::DxpGrpError::{FileTooShort, InvalidHeader},
		error::{DxpGrpError, DxpGrpError::IndexingFileOutOfBounds},
	},
};

/// This function yields the names from a DXP file

// IF YOU WISH TO SEE THE CONTENTS OF DXP FILES, YOU SHOULD OPEN AN ISSUE AND I WILL ADD FUNCTIONALITY FOR IT
// The data section is pretty much similar to the inner-vromf container
// It has a section of offsets + sizes, followed by the data region indexed by said offsets and sizes

pub fn parse_dxp(file: &[u8]) -> Result<Vec<String>, DxpGrpError> {
	// Return empty names for empty file
	if file.len() == 0 {
		return Ok(vec![]);
	}
	if file.len() < 0x48 {
		return Err(FileTooShort { len: file.len() });
	}

	let dxp_header = String::from_utf8(file[0..4].to_owned()).map_err(|e| e.utf8_error())?;
	if dxp_header != "DxP2" {
		return Err(InvalidHeader { found: dxp_header });
	}

	// Fixed offset at 0x8
	let file_count = bytes_to_offset(&file.get(0x8..(0x8 + size_of::<u32>())).ok_or(
		IndexingFileOutOfBounds {
			current_ptr: 0x8,
			file_size:   file.len(),
		},
	)?)
	.expect("Infallible");

	// Names begin at 0x48, usual CString sequence
	let mut ptr: usize = 0x48;
	let mut names = Vec::with_capacity(file_count);
	for _ in 0..file_count {
		let str = CStr::from_bytes_until_nul(&file.get(ptr..).ok_or(IndexingFileOutOfBounds {
			current_ptr: ptr,
			file_size:   file.len(),
		})?)?
		.to_str()?
		.to_owned();
		//              +1 for null
		ptr += str.len() + 1;
		names.push(str);
	}
	Ok(names)
}

/// A DXP texture pack with its texture headers, see `DDSxTexturePack2` in Dagor's `texPackMgr2.cpp`
///
/// After the 16-byte file header (`DxP2`, version, texture count, size of the table dump) comes a dump
/// whose offsets are relative to its own start (file offset 0x10):
///
/// |Dump offset|Content|
/// |---|---|
/// |0x00|Name table: u32 offset, u32 count, 8 bytes padding. Points at u64 offsets of C-strings|
/// |0x10|DDSx header table: u32 offset, u32 count, padding. 32 bytes per texture|
/// |0x20|Record table: u32 offset, u32 count, padding. 24 bytes per texture, holding the file offset (+0x0C) and packed size (+0x10) of its data|
pub struct DxpPack<'a> {
	file:        &'a [u8],
	pub entries: Vec<DxpEntry>,
}

#[derive(Debug, Clone)]
pub struct DxpEntry {
	pub name:   String,
	pub header: DdsxHeader,
	offset:     usize,
	size:       usize,
}

impl<'a> DxpPack<'a> {
	pub fn parse(file: &'a [u8]) -> Result<Self, DxpGrpError> {
		let oob = |ptr: usize| IndexingFileOutOfBounds {
			current_ptr: ptr,
			file_size:   file.len(),
		};
		let u32_at = |buf: &[u8], ofs: usize| -> Result<usize, DxpGrpError> {
			let b = buf.get(ofs..ofs + 4).ok_or(oob(ofs))?;
			Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
		};

		if file.get(..4) != Some(b"DxP2".as_slice()) {
			return Err(InvalidHeader {
				found: String::from_utf8_lossy(file.get(..4).unwrap_or(file)).into_owned(),
			});
		}
		let version = u32_at(file, 4)?;
		if version != 2 {
			return Err(DxpGrpError::UnsupportedTexture(format!(
				"DXP version {version}"
			)));
		}
		let dump_size = u32_at(file, 0xC)?;
		let dump = file
			.get(0x10..0x10 + dump_size)
			.ok_or(oob(0x10 + dump_size))?;

		let (names_ofs, count) = (u32_at(dump, 0x00)?, u32_at(dump, 0x04)?);
		let (headers_ofs, headers_count) = (u32_at(dump, 0x10)?, u32_at(dump, 0x14)?);
		let (records_ofs, records_count) = (u32_at(dump, 0x20)?, u32_at(dump, 0x24)?);
		if headers_count != count || records_count != count {
			return Err(DxpGrpError::UnsupportedTexture(format!(
				"inconsistent DXP tables: {count} names, {headers_count} headers, {records_count} records"
			)));
		}

		check_table(dump, names_ofs, count, 8)?;
		check_table(dump, headers_ofs, count, DdsxHeader::SIZE)?;
		check_table(dump, records_ofs, count, 24)?;

		let budget = NameBudget::new(dump);
		let mut entries = Vec::with_capacity(count);
		for i in 0..count {
			let name_ofs = u32_at(dump, names_ofs + i * 8)?;
			let name = budget.take(
				CStr::from_bytes_until_nul(dump.get(name_ofs..).ok_or(oob(name_ofs))?)?
					.to_str()?
					.to_owned(),
			)?;
			let header_ofs = headers_ofs + i * DdsxHeader::SIZE;
			let header = DdsxHeader::parse(dump.get(header_ofs..).ok_or(oob(header_ofs))?)?;
			let record = records_ofs + i * 24;
			let offset = u32_at(dump, record + 0x0C)?;
			let packed = u32_at(dump, record + 0x10)?;
			let size = if packed > 0 {
				packed
			} else {
				header.mem_size as usize
			};
			entries.push(DxpEntry {
				name,
				header,
				offset,
				size,
			});
		}
		Ok(Self { file, entries })
	}

	/// Decompresses one texture into DDS mip order
	pub fn texture(&self, index: usize) -> Result<Texture, DxpGrpError> {
		let entry = self.entries.get(index).ok_or(IndexingFileOutOfBounds {
			current_ptr: index,
			file_size:   self.entries.len(),
		})?;
		let raw = self
			.file
			.get(entry.offset..entry.offset + entry.size)
			.ok_or(IndexingFileOutOfBounds {
				current_ptr: entry.offset + entry.size,
				file_size:   self.file.len(),
			})?;
		Texture::from_ddsx(entry.header, raw)
	}
}
