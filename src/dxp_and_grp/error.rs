use core::ffi::FromBytesUntilNulError;
use std::str::Utf8Error;

#[derive(Debug, thiserror::Error)]
pub enum DxpGrpError {
	#[error(
		"The files header that was found: {found}, is not the expected header \"DxP2\" or \"GRP2\""
	)]
	InvalidHeader { found: String },

	#[error(transparent)]
	CStringError(#[from] FromBytesUntilNulError),

	#[error(transparent)]
	Utf8Error(#[from] Utf8Error),

	#[error("current ptr {current_ptr} is out of bounds for buffer: {file_size} bytes")]
	IndexingFileOutOfBounds {
		current_ptr: usize,
		file_size:   usize,
	},

	#[error(
		"The file was a valid, but cut short before the names section, minimum bytes are 0x48, but the file was only {len:X}"
	)]
	FileTooShort { len: usize },

	#[error(transparent)]
	IoError(#[from] std::io::Error),

	#[error("Texture is not supported: {0}")]
	UnsupportedTexture(String),

	#[error("Failed to decompress data: {0}")]
	Decompression(String),

	#[error("Failed to encode texture: {0}")]
	Encoding(String),

	#[error("Invalid or unsupported model data: {0}")]
	Model(String),

	#[error("Names add up to more than the {size} bytes they are stored in")]
	NamesTooLong { size: usize },
}
