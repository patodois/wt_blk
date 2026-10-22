//! DDSx, the texture container stored inside DXP packs
//!
//! Every texture starts with a 32-byte header, followed by its (optionally compressed) surface data.
//! The layout mirrors `ddsx::Header` in Dagor's `prog/engine/sharedInclude/3d/ddsxTex.h`:
//!
//! |Offset|Type|Field|
//! |---|---|---|
//! |0x00|u32|Label, always `DDSx`|
//! |0x04|u32|D3D format, either a FourCC like `DXT5` or a numeric `D3DFMT_*`|
//! |0x08|u32|Flags, see below|
//! |0x0C|u16, u16|Width, height|
//! |0x10|u8, u8|Mip levels, levels stored in the separate HQ part|
//! |0x12|u16|Depth (volume textures)|
//! |0x14|u16|Bits per pixel (uncompressed formats)|
//! |0x16|u8|Low nibble: low / medium quality mip, high nibble ignored|
//! |0x17|u8|Low nibble: block size as `log2(bytes per 4x4 block)`, 0 for uncompressed formats|
//! |0x18|u32|Size of the surface data once decompressed|
//! |0x1C|u32|Size of the compressed data, 0 when stored|
//!
//! Notable flags: `0x800` cubemap, `0x1000` volume, `0x6000` mips are generated on load (only the top mip is stored),
//! `0x2_0000` data is a delta against a paired base texture, `0x4_0000` mips are stored smallest first.
//! The top three bits select the compression: `0x2` zstd, `0x3` Oodle, `0x4` zlib, `0x2 << 29` 7zip.

use crate::dxp_and_grp::error::DxpGrpError;

pub const DDSX_LABEL: u32 = u32::from_le_bytes(*b"DDSx");

const FLG_CUBTEX: u32 = 0x0800;
const FLG_VOLTEX: u32 = 0x1000;
const FLG_GENMIP: u32 = 0x2000 | 0x4000;
const FLG_NEED_PAIRED_BASETEX: u32 = 0x0002_0000;
const FLG_REV_MIP_ORDER: u32 = 0x0004_0000;
const FLG_ARRTEX: u32 = 0x0020_0000;
const FLG_COMPR_MASK: u32 = 0xE000_0000;
const FLG_ZSTD: u32 = 0x2000_0000;
const FLG_7ZIP: u32 = 0x4000_0000;
const FLG_OODLE: u32 = 0x6000_0000;
const FLG_ZLIB: u32 = 0x8000_0000;

const D3DFMT_A8R8G8B8: u32 = 21;
const D3DFMT_A8: u32 = 28;
const D3DFMT_A8B8G8R8: u32 = 32;
const D3DFMT_L8: u32 = 50;
const D3DFMT_R16F: u32 = 111;
const D3DFMT_A16B16G16R16F: u32 = 113;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdsxHeader {
	pub d3d_format:     u32,
	pub flags:          u32,
	pub width:          u16,
	pub height:         u16,
	pub levels:         u8,
	pub hq_part_levels: u8,
	pub depth:          u16,
	pub bits_per_pixel: u16,
	/// `log2` of the bytes per 4x4 block, 0 for uncompressed formats
	pub dxt_shift:      u8,
	pub mem_size:       u32,
	pub packed_size:    u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
	None,
	Zstd,
	Oodle,
	Zlib,
	SevenZip,
}

impl DdsxHeader {
	pub const SIZE: usize = 32;

	pub fn parse(bytes: &[u8]) -> Result<Self, DxpGrpError> {
		let b = bytes.get(..Self::SIZE).ok_or(DxpGrpError::IndexingFileOutOfBounds {
			current_ptr: Self::SIZE,
			file_size:   bytes.len(),
		})?;
		let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
		let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
		let label = u32_at(0);
		if label != DDSX_LABEL {
			return Err(DxpGrpError::InvalidHeader {
				found: String::from_utf8_lossy(&b[..4]).into_owned(),
			});
		}
		Ok(Self {
			d3d_format:     u32_at(0x04),
			flags:          u32_at(0x08),
			width:          u16_at(0x0C),
			height:         u16_at(0x0E),
			levels:         b[0x10],
			hq_part_levels: b[0x11],
			depth:          u16_at(0x12),
			bits_per_pixel: u16_at(0x14),
			dxt_shift:      b[0x17] & 0xF,
			mem_size:       u32_at(0x18),
			packed_size:    u32_at(0x1C),
		})
	}

	/// Placeholder entries carry nothing but the label
	pub fn is_empty(&self) -> bool {
		self.d3d_format == 0 && self.width == 0 && self.height == 0 && self.mem_size == 0
	}

	pub fn compression(&self) -> Compression {
		match self.flags & FLG_COMPR_MASK {
			FLG_ZSTD => Compression::Zstd,
			FLG_OODLE => Compression::Oodle,
			FLG_ZLIB => Compression::Zlib,
			FLG_7ZIP => Compression::SevenZip,
			_ => Compression::None,
		}
	}

	pub fn is_cubemap(&self) -> bool {
		self.flags & FLG_CUBTEX != 0
	}

	pub fn is_volume(&self) -> bool {
		self.flags & FLG_VOLTEX != 0
	}

	/// Only the top mip is stored, the game generates the rest when loading
	pub fn generates_mips(&self) -> bool {
		self.flags & FLG_GENMIP != 0
	}

	/// The FourCC as text if it is one, e.g. `DXT5`, otherwise the numeric `D3DFMT_*` value
	pub fn format_name(&self) -> String {
		let bytes = self.d3d_format.to_le_bytes();
		if bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
			String::from_utf8_lossy(&bytes).trim_end().to_owned()
		} else {
			format!("D3DFMT_{}", self.d3d_format)
		}
	}

	fn surface_size(&self, level: u32) -> usize {
		let w = (u32::from(self.width) >> level).max(1) as usize;
		let h = (u32::from(self.height) >> level).max(1) as usize;
		if self.dxt_shift == 0 {
			w * h * usize::from(self.bits_per_pixel) / 8
		} else {
			((w >> 2).max(1) * (h >> 2).max(1)) << self.dxt_shift
		}
	}

	fn level_size(&self, level: u32) -> usize {
		let depth = if self.is_volume() {
			(u32::from(self.depth) >> level).max(1) as usize
		} else {
			1
		};
		self.surface_size(level) * depth
	}

	/// Levels present in the decoded data
	fn stored_levels(&self) -> u32 {
		if self.generates_mips() {
			1
		} else {
			u32::from(self.levels.max(1))
		}
	}

	fn faces(&self) -> usize {
		if self.is_cubemap() { 6 } else { 1 }
	}

	fn expected_size(&self) -> usize {
		(0..self.stored_levels())
			.map(|l| self.level_size(l))
			.sum::<usize>()
			* self.faces()
	}
}

/// A decoded texture: surface data uncompressed and in DDS order (largest mip first, faces outermost for cubemaps)
#[derive(Debug, Clone)]
pub struct Texture {
	pub header: DdsxHeader,
	pub data:   Vec<u8>,
}

impl Texture {
	/// Decodes the raw bytes that follow a DDSx header
	pub fn from_ddsx(header: DdsxHeader, raw: &[u8]) -> Result<Self, DxpGrpError> {
		let unsupported = |what: &str| DxpGrpError::UnsupportedTexture(what.to_owned());
		if header.is_empty() {
			return Err(unsupported("empty placeholder entry"));
		}
		if header.flags & FLG_NEED_PAIRED_BASETEX != 0 {
			return Err(unsupported(
				"stored as a difference to a paired base texture",
			));
		}
		if header.flags & FLG_ARRTEX != 0 {
			return Err(unsupported("array textures"));
		}

		let mem_size = header.mem_size as usize;
		let mut data = match header.compression() {
			Compression::None => {
				raw.get(..mem_size)
					.ok_or(DxpGrpError::IndexingFileOutOfBounds {
						current_ptr: mem_size,
						file_size:   raw.len(),
					})?
					.to_vec()
			},
			Compression::Zstd => {
				let packed = packed_slice(&header, raw)?;
				zstd::bulk::decompress(packed, mem_size)
					.map_err(|e| DxpGrpError::Decompression(e.to_string()))?
			},
			Compression::Oodle => {
				let packed = packed_slice(&header, raw)?;
				let mut out = vec![0_u8; mem_size];
				let written = oozextract::Extractor::new()
					.read_from_slice(packed, &mut out)
					.map_err(|e| DxpGrpError::Decompression(format!("{e:?}")))?;
				if written != mem_size {
					return Err(DxpGrpError::Decompression(format!(
						"Oodle produced {written} bytes, expected {mem_size}"
					)));
				}
				out
			},
			Compression::Zlib => return Err(unsupported("zlib compression")),
			Compression::SevenZip => return Err(unsupported("7zip compression")),
		};

		let expected = header.expected_size();
		if data.len() < expected {
			return Err(DxpGrpError::Decompression(format!(
				"{} bytes of surface data, but the header describes {expected}",
				data.len()
			)));
		}
		// Textures with generated mips carry a few trailing bytes after the top mip
		data.truncate(expected);

		if header.flags & FLG_REV_MIP_ORDER != 0 && !header.generates_mips() {
			data = forward_mips(&header, &data);
		}
		Ok(Self { header, data })
	}

	pub fn width(&self) -> u32 {
		u32::from(self.header.width)
	}

	pub fn height(&self) -> u32 {
		u32::from(self.header.height)
	}

	pub fn mip_levels(&self) -> u32 {
		self.header.stored_levels()
	}

	/// Serializes the texture as a DDS file, keeping the original pixel format and all stored mips
	pub fn to_dds(&self) -> Result<Vec<u8>, DxpGrpError> {
		let h = &self.header;
		let format = DdsFormat::of(h)?;

		const DDSD_CAPS: u32 = 0x1;
		const DDSD_HEIGHT: u32 = 0x2;
		const DDSD_WIDTH: u32 = 0x4;
		const DDSD_PIXELFORMAT: u32 = 0x1000;
		const DDSD_MIPMAPCOUNT: u32 = 0x2_0000;
		const DDSD_DEPTH: u32 = 0x80_0000;
		const DDSCAPS_COMPLEX: u32 = 0x8;
		const DDSCAPS_TEXTURE: u32 = 0x1000;
		const DDSCAPS_MIPMAP: u32 = 0x40_0000;
		const DDSCAPS2_CUBEMAP_ALL_FACES: u32 = 0x200 | 0xFC00;
		const DDSCAPS2_VOLUME: u32 = 0x20_0000;

		let mut flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT;
		let mut caps = DDSCAPS_TEXTURE;
		let mut caps2 = 0;
		if self.mip_levels() > 1 {
			caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP;
		}
		if h.is_cubemap() {
			caps |= DDSCAPS_COMPLEX;
			caps2 |= DDSCAPS2_CUBEMAP_ALL_FACES;
		}
		if h.is_volume() {
			flags |= DDSD_DEPTH;
			caps |= DDSCAPS_COMPLEX;
			caps2 |= DDSCAPS2_VOLUME;
		}

		let mut out = Vec::with_capacity(4 + 124 + 20 + self.data.len());
		let mut put = |v: u32| out.extend_from_slice(&v.to_le_bytes());
		put(u32::from_le_bytes(*b"DDS "));
		put(124);
		put(flags);
		put(self.height());
		put(self.width());
		put(0); // pitch or linear size
		put(if h.is_volume() { u32::from(h.depth) } else { 0 });
		put(self.mip_levels());
		for _ in 0..11 {
			put(0);
		}
		// Pixel format
		put(32);
		put(format.flags);
		put(format.four_cc);
		for v in format.masks {
			put(v);
		}
		put(caps);
		put(caps2);
		for _ in 0..3 {
			put(0);
		}
		if let Some(dxgi_format) = format.dxgi_format {
			const DIMENSION_TEXTURE2D: u32 = 3;
			const DIMENSION_TEXTURE3D: u32 = 4;
			const MISC_TEXTURECUBE: u32 = 0x4;
			put(dxgi_format);
			put(if h.is_volume() {
				DIMENSION_TEXTURE3D
			} else {
				DIMENSION_TEXTURE2D
			});
			put(if h.is_cubemap() { MISC_TEXTURECUBE } else { 0 });
			put(1); // array size
			put(0);
		}
		out.extend_from_slice(&self.data);
		Ok(out)
	}
}

fn packed_slice<'a>(header: &DdsxHeader, raw: &'a [u8]) -> Result<&'a [u8], DxpGrpError> {
	let packed = header.packed_size as usize;
	raw.get(..packed).ok_or(DxpGrpError::IndexingFileOutOfBounds {
		current_ptr: packed,
		file_size:   raw.len(),
	})
}

/// Reorders mips stored smallest-first into DDS order, see `ddsx_forward_mips_inplace` in Dagor
fn forward_mips(header: &DdsxHeader, data: &[u8]) -> Vec<u8> {
	let levels = header.stored_levels();
	let mut out = vec![0_u8; data.len()];
	if header.is_cubemap() {
		// Stored: for each mip from smallest, all six faces. DDS wants each face with its full mip chain
		let mip_offsets: Vec<usize> = (0..levels)
			.scan(0, |acc, l| {
				let ofs = *acc;
				*acc += header.surface_size(l);
				Some(ofs)
			})
			.collect();
		let face_size: usize = (0..levels).map(|l| header.surface_size(l)).sum();
		let mut src = 0;
		for level in (0..levels).rev() {
			let size = header.surface_size(level);
			for face in 0..6 {
				let dst = face_size * face + mip_offsets[level as usize];
				out[dst..dst + size].copy_from_slice(&data[src..src + size]);
				src += size;
			}
		}
	} else {
		let mut src = 0;
		let mut dst_end = out.len();
		for level in (0..levels).rev() {
			let size = header.level_size(level);
			out[dst_end - size..dst_end].copy_from_slice(&data[src..src + size]);
			src += size;
			dst_end -= size;
		}
	}
	out
}

struct DdsFormat {
	flags:       u32,
	four_cc:     u32,
	/// Bit count, then R, G, B and A masks
	masks:       [u32; 5],
	dxgi_format: Option<u32>,
}

impl DdsFormat {
	fn of(h: &DdsxHeader) -> Result<Self, DxpGrpError> {
		const DDPF_ALPHAPIXELS: u32 = 0x1;
		const DDPF_ALPHA: u32 = 0x2;
		const DDPF_FOURCC: u32 = 0x4;
		const DDPF_RGB: u32 = 0x40;
		const DDPF_LUMINANCE: u32 = 0x2_0000;
		const DXGI_FORMAT_BC7_UNORM: u32 = 98;

		let masked = |flags, masks| Self {
			flags,
			four_cc: 0,
			masks,
			dxgi_format: None,
		};
		let four_cc = |four_cc| Self {
			flags: DDPF_FOURCC,
			four_cc,
			masks: [0; 5],
			dxgi_format: None,
		};
		Ok(match h.d3d_format {
			D3DFMT_A8R8G8B8 => masked(DDPF_RGB | DDPF_ALPHAPIXELS, [
				32,
				0x00FF_0000,
				0x0000_FF00,
				0x0000_00FF,
				0xFF00_0000,
			]),
			D3DFMT_A8B8G8R8 => masked(DDPF_RGB | DDPF_ALPHAPIXELS, [
				32,
				0x0000_00FF,
				0x0000_FF00,
				0x00FF_0000,
				0xFF00_0000,
			]),
			D3DFMT_L8 => masked(DDPF_LUMINANCE, [8, 0xFF, 0, 0, 0]),
			D3DFMT_A8 => masked(DDPF_ALPHA, [8, 0, 0, 0, 0xFF]),
			f if f == u32::from_le_bytes(*b"BC7 ") => Self {
				dxgi_format: Some(DXGI_FORMAT_BC7_UNORM),
				..four_cc(u32::from_le_bytes(*b"DX10"))
			},
			// Text FourCCs (DXTn, ATIn) are valid as-is
			f if f > 0xFF => four_cc(f),
			// So are the numeric codes DDS readers accept in the FourCC field:
			// A16B16G16R16, Q16W16V16U16, R16F, G16R16F, A16B16G16R16F, R32F, G32R32F, A32B32G32R32F
			f @ (36 | 110 | 111 | 112 | 113 | 114 | 115 | 116) => four_cc(f),
			_ => {
				return Err(DxpGrpError::UnsupportedTexture(format!(
					"pixel format {}",
					h.format_name()
				)));
			},
		})
	}
}

#[cfg(feature = "png")]
impl Texture {
	/// Decodes the top mip (first face or slice) to 8-bit RGBA
	pub fn to_rgba(&self) -> Result<Vec<u8>, DxpGrpError> {
		let (w, h) = (self.width() as usize, self.height() as usize);
		let top = &self.data[..self.header.surface_size(0)];
		let fourcc = |s: &[u8; 4]| u32::from_le_bytes(*s);
		let decoder: Option<fn(&[u8], usize, usize, &mut [u32]) -> Result<(), &'static str>> =
			match self.header.d3d_format {
				f if f == fourcc(b"DXT1") => Some(texture2ddecoder::decode_bc1),
				f if f == fourcc(b"DXT3") => Some(texture2ddecoder::decode_bc2),
				f if f == fourcc(b"DXT5") => Some(texture2ddecoder::decode_bc3),
				f if f == fourcc(b"ATI1") => Some(texture2ddecoder::decode_bc4),
				f if f == fourcc(b"ATI2") => Some(texture2ddecoder::decode_bc5),
				f if f == fourcc(b"BC7 ") => Some(texture2ddecoder::decode_bc7),
				_ => None,
			};

		let mut rgba = Vec::with_capacity(w * h * 4);
		if let Some(decode) = decoder {
			let mut pixels = vec![0_u32; w * h];
			decode(top, w, h, &mut pixels).map_err(|e| DxpGrpError::Decompression(e.to_owned()))?;
			// The decoder yields BGRA in little-endian order
			for p in pixels {
				let [b, g, r, a] = p.to_le_bytes();
				rgba.extend_from_slice(&[r, g, b, a]);
			}
			return Ok(rgba);
		}
		match self.header.d3d_format {
			D3DFMT_A8R8G8B8 => {
				for px in top.chunks_exact(4) {
					rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
				}
			},
			D3DFMT_A8B8G8R8 => rgba.extend_from_slice(top),
			D3DFMT_L8 => {
				for &l in top {
					rgba.extend_from_slice(&[l, l, l, 0xFF]);
				}
			},
			D3DFMT_A8 => {
				for &a in top {
					rgba.extend_from_slice(&[0xFF, 0xFF, 0xFF, a]);
				}
			},
			// HDR formats are clamped to [0, 1]
			D3DFMT_A16B16G16R16F => {
				for px in top.chunks_exact(8) {
					for c in px.chunks_exact(2) {
						rgba.push(unorm8(f16_to_f32(u16::from_le_bytes([c[0], c[1]]))));
					}
				}
			},
			D3DFMT_R16F => {
				for c in top.chunks_exact(2) {
					let r = unorm8(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
					rgba.extend_from_slice(&[r, r, r, 0xFF]);
				}
			},
			_ => {
				return Err(DxpGrpError::UnsupportedTexture(format!(
					"PNG conversion of {}",
					self.header.format_name()
				)));
			},
		}
		Ok(rgba)
	}

	/// Encodes the top mip as a PNG (HDR formats are clamped)
	pub fn to_png(&self) -> Result<Vec<u8>, DxpGrpError> {
		let rgba = self.to_rgba()?;
		let mut out = Vec::new();
		let mut encoder = png::Encoder::new(&mut out, self.width(), self.height());
		encoder.set_color(png::ColorType::Rgba);
		encoder.set_depth(png::BitDepth::Eight);
		let mut writer = encoder
			.write_header()
			.map_err(|e| DxpGrpError::Encoding(e.to_string()))?;
		writer
			.write_image_data(&rgba)
			.map_err(|e| DxpGrpError::Encoding(e.to_string()))?;
		writer
			.finish()
			.map_err(|e| DxpGrpError::Encoding(e.to_string()))?;
		Ok(out)
	}
}

#[cfg(feature = "png")]
fn unorm8(v: f32) -> u8 {
	(v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// IEEE 754 half to single precision
#[cfg(feature = "png")]
fn f16_to_f32(h: u16) -> f32 {
	let sign = u32::from(h >> 15) << 31;
	let exp = u32::from((h >> 10) & 0x1F);
	let mant = u32::from(h & 0x3FF);
	let bits = match (exp, mant) {
		(0, 0) => sign,
		// Subnormal: renormalise the mantissa
		(0, _) => {
			let shift = mant.leading_zeros() - 21;
			sign | ((113 - shift) << 23) | ((mant << shift) & 0x3FF) << 13
		},
		(0x1F, _) => sign | 0x7F80_0000 | (mant << 13),
		_ => sign | ((exp + 112) << 23) | (mant << 13),
	};
	f32::from_bits(bits)
}

#[cfg(test)]
mod test {
	use super::*;

	/// DXT1 header for a `size`x`size` texture
	fn dxt1(size: u16, levels: u8, flags: u32, mem_size: u32) -> DdsxHeader {
		DdsxHeader {
			d3d_format: u32::from_le_bytes(*b"DXT1"),
			flags,
			width: size,
			height: size,
			levels,
			hq_part_levels: 0,
			depth: 0,
			bits_per_pixel: 0,
			dxt_shift: 3,
			mem_size,
			packed_size: 0,
		}
	}

	#[test]
	fn header_roundtrip() {
		let mut raw = [0_u8; 32];
		raw[..4].copy_from_slice(b"DDSx");
		raw[4..8].copy_from_slice(b"DXT5");
		raw[8..12].copy_from_slice(&0x6004_0011_u32.to_le_bytes());
		raw[12..14].copy_from_slice(&512_u16.to_le_bytes());
		raw[14..16].copy_from_slice(&256_u16.to_le_bytes());
		raw[16] = 10;
		raw[0x17] = 4;
		raw[0x18..0x1C].copy_from_slice(&174_776_u32.to_le_bytes());
		raw[0x1C..0x20].copy_from_slice(&110_032_u32.to_le_bytes());
		let h = DdsxHeader::parse(&raw).unwrap();
		assert_eq!((h.width, h.height, h.levels, h.dxt_shift), (512, 256, 10, 4));
		assert_eq!(h.compression(), Compression::Oodle);
		assert_eq!(h.format_name(), "DXT5");
		assert!(DdsxHeader::parse(b"DDSy and more than thirty two bytes").is_err());
	}

	#[test]
	fn reversed_mips_are_put_in_dds_order() {
		// 8x8 DXT1: mip0 is 32 bytes, mip1 (4x4) 8 bytes, stored smallest first
		let mut stored = vec![1_u8; 8];
		stored.extend_from_slice(&[0_u8; 32]);
		let tex = Texture::from_ddsx(dxt1(8, 2, FLG_REV_MIP_ORDER, 40), &stored).unwrap();
		assert_eq!(&tex.data[..32], &[0_u8; 32]);
		assert_eq!(&tex.data[32..], &[1_u8; 8]);
	}

	#[test]
	fn reversed_cubemap_mips_become_face_major() {
		// 4x4 DXT1 cubemap with 2 levels of 8 bytes each, stored as (mip1 face0..5), (mip0 face0..5)
		let stored: Vec<u8> = (0..12_u8).flat_map(|i| [i; 8]).collect();
		let tex = Texture::from_ddsx(dxt1(4, 2, FLG_REV_MIP_ORDER | FLG_CUBTEX, 96), &stored).unwrap();
		for face in 0..6 {
			let chain = &tex.data[face * 16..face * 16 + 16];
			assert_eq!(&chain[..8], &[6 + face as u8; 8], "face {face} mip0");
			assert_eq!(&chain[8..], &[face as u8; 8], "face {face} mip1");
		}
	}

	#[test]
	fn generated_mips_keep_only_the_top_level() {
		// The game stores the top mip plus 12 trailing bytes and builds the chain on load
		let tex = Texture::from_ddsx(dxt1(8, 4, 0x4000, 44), &[7_u8; 44]).unwrap();
		assert_eq!(tex.data.len(), 32);
		assert_eq!(tex.mip_levels(), 1);
		let dds = tex.to_dds().unwrap();
		assert_eq!(u32::from_le_bytes(dds[28..32].try_into().unwrap()), 1);
	}

	#[test]
	fn rejects_unusable_entries() {
		let empty = DdsxHeader {
			d3d_format: 0,
			width: 0,
			height: 0,
			mem_size: 0,
			..dxt1(0, 0, 0, 0)
		};
		assert!(Texture::from_ddsx(empty, &[]).is_err());
		assert!(Texture::from_ddsx(dxt1(8, 1, FLG_NEED_PAIRED_BASETEX, 32), &[0; 32]).is_err());
		assert!(Texture::from_ddsx(dxt1(8, 2, 0, 40), &[0; 39]).is_err());
	}

	#[test]
	fn unknown_numeric_formats_are_rejected() {
		let header = DdsxHeader {
			d3d_format: 23, // D3DFMT_R5G6B5, not representable without bit masks here
			dxt_shift: 0,
			bits_per_pixel: 16,
			..dxt1(1, 1, 0, 2)
		};
		let tex = Texture::from_ddsx(header, &[0; 2]).unwrap();
		assert!(tex.to_dds().is_err());
	}

	#[test]
	fn bc7_uses_a_dx10_header() {
		let header = DdsxHeader {
			d3d_format: u32::from_le_bytes(*b"BC7 "),
			dxt_shift: 4,
			..dxt1(4, 1, 0, 16)
		};
		let dds = Texture::from_ddsx(header, &[0; 16]).unwrap().to_dds().unwrap();
		assert_eq!(&dds[84..88], b"DX10");
		assert_eq!(u32::from_le_bytes(dds[128..132].try_into().unwrap()), 98);
		assert_eq!(dds.len(), 148 + 16);
	}

	#[cfg(feature = "png")]
	#[test]
	fn bgra_is_swizzled_to_rgba() {
		let header = DdsxHeader {
			d3d_format: D3DFMT_A8R8G8B8,
			dxt_shift: 0,
			bits_per_pixel: 32,
			..dxt1(1, 1, 0, 4)
		};
		let tex = Texture::from_ddsx(header, &[1, 2, 3, 4]).unwrap();
		assert_eq!(tex.to_rgba().unwrap(), [3, 2, 1, 4]);
		assert_eq!(&tex.to_png().unwrap()[1..4], b"PNG");
	}

	#[cfg(feature = "png")]
	#[test]
	fn half_floats() {
		assert_eq!(f16_to_f32(0x3C00), 1.0);
		assert_eq!(f16_to_f32(0xC000), -2.0);
		assert_eq!(f16_to_f32(0x3800), 0.5);
		assert_eq!(f16_to_f32(0x0001), 2.0_f32.powi(-24));
		assert!(f16_to_f32(0x7C00).is_infinite());

		// 1.0, 0.5, -1.0, 2.0 -> clamped to 255, 128, 0, 255
		let header = DdsxHeader {
			d3d_format: D3DFMT_A16B16G16R16F,
			dxt_shift: 0,
			bits_per_pixel: 64,
			..dxt1(1, 1, 0, 8)
		};
		let raw: Vec<u8> = [0x3C00_u16, 0x3800, 0xBC00, 0x4000]
			.iter()
			.flat_map(|h| h.to_le_bytes())
			.collect();
		let tex = Texture::from_ddsx(header, &raw).unwrap();
		assert_eq!(tex.to_rgba().unwrap(), [255, 128, 0, 255]);
	}
}
