//! DynModel resources (Dagor's `DynamicRenderableSceneLodsResource`): animated models such as
//! vehicles, with LODs made of rigid parts placed by a [`crate::dxp_and_grp::geom_node_tree`] and
//! skinned meshes (tracks, cables).
//!
//! Stream, as read by `DynamicRenderableSceneLodsResource::loadResource`
//! (`prog/engine/shaders/dynSceneRes.cpp`):
//!
//! ```text
//! u32 lods dump size
//! u32 texture count, material count   both 0xFFFFFFFF: materials are described in dynModelDesc.bin
//! u32 vdata count, u32 header size    the size's top bits select the compression (see below)
//! vertex data                         ShaderMatVdata (matVdataLoad.cpp)
//! lods dump                           LOD table, bounding box, position unpack offset (+0x28) and
//!                                     scale (+0x38); LOD records {u64 scene size, f32 range, f32}
//! u32 size, node name map dump        RoNameMapEx: sorted names plus their node ids
//! per LOD: scene dump                 rigids {u64 mesh size, f32x4 sphere, i32 node, u32} and skins
//!          ShaderMesh dump per rigid  48-byte elements: material, vdata, index range, base vertex
//! skinned meshes (if any LOD has them)
//! ```
//!
//! Vertex data: a header with one 32-byte entry per vertex buffer (vertex count, stride, index sizes,
//! flags, and a declaration of its channels), followed by each buffer's vertices and indices. Index
//! buffers flagged `0x200` are meshoptimizer index sequences. When the header size has the bits
//! `0b110` on top, header and buffers sit in a block compressed with zstd or Oodle.

use std::io::Read;

use crate::dxp_and_grp::{
	cursor::{Cursor, NameBudget, check_table},
	error::DxpGrpError,
	meshopt::decode_index_sequence,
	unpack,
};

// Vertex channel types (VSDT_*) and usages (SCUSAGE_*), from Dagor's dag_consts_base.h and dag_shaderCommon.h
pub mod channel {
	pub const FLOAT2: u32 = 1;
	pub const FLOAT3: u32 = 2;
	pub const FLOAT4: u32 = 3;
	pub const E3DCOLOR: u32 = 4;
	pub const UBYTE4: u32 = 5;
	pub const SHORT2: u32 = 6;
	pub const SHORT4: u32 = 7;
	pub const SHORT2N: u32 = 9;
	pub const SHORT4N: u32 = 10;
	pub const HALF2: u32 = 15;
	pub const HALF4: u32 = 16;

	pub const POSITION: u32 = 0;
	pub const NORMAL: u32 = 1;
	pub const COLOR: u32 = 2;
	pub const TEXCOORD: u32 = 3;
}
use channel::*;

const VDATA_I16: u32 = 0x20;
const VDATA_I32: u32 = 0x40;
const VDATA_PACKED_IB: u32 = 0x200;
/// Texture coordinates are stored as SHORT2 in 1/4096 units
const TEXCOORD_SCALE: f32 = 1.0 / 4096.0;

fn model_err(what: impl Into<String>) -> DxpGrpError {
	DxpGrpError::Model(what.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Channel {
	/// `VSDT_*` data type, see [`channel`]
	pub ty:     u32,
	/// `SCUSAGE_*` usage, see [`channel`]
	pub usage:  u32,
	pub index:  u16,
	/// Byte offset inside a vertex
	pub offset: usize,
}

/// One vertex buffer with its index buffer
#[derive(Debug, Clone)]
pub struct VData {
	pub stride:   usize,
	pub vertices: Vec<u8>,
	pub indices:  Vec<u32>,
	pub channels: Vec<Channel>,
}

impl VData {
	pub fn vertex_count(&self) -> usize {
		self.vertices.len() / self.stride.max(1)
	}

	/// Whether vertices carry a second texture coordinate set (used for camouflage and occlusion maps)
	pub fn has_uv1(&self) -> bool {
		self.channels
			.iter()
			.any(|c| c.usage == TEXCOORD && c.index == 1)
	}
}

/// A draw call: a range of triangles from one vertex buffer with one material
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Elem {
	/// Index into the model's material list (`matR` / `matS` in dynModelDesc.bin)
	pub material:    usize,
	/// Index into [`DynModel::vdata`] (rigid parts) or [`DynModel::skin_vdata`] (skins)
	pub vdata:       usize,
	pub first_index: usize,
	pub triangles:   usize,
	/// Added to every index of the range
	pub base_vertex: usize,
}

/// A rigid part, placed by the skeleton node of the same name
#[derive(Debug, Clone)]
pub struct Rigid {
	pub node:  String,
	pub elems: Vec<Elem>,
}

/// A skinned mesh (tracks, cables), stored in bind pose in model space
#[derive(Debug, Clone)]
pub struct Skin {
	pub node:  String,
	pub elems: Vec<Elem>,
}

#[derive(Debug, Clone)]
pub struct Lod {
	/// Distance up to which this LOD is used
	pub range:  f32,
	pub rigids: Vec<Rigid>,
	pub skins:  Vec<Skin>,
}

#[derive(Debug, Clone)]
pub struct DynModel {
	pub vdata:      Vec<VData>,
	pub skin_vdata: Vec<VData>,
	/// Packed positions are `short4n * pos_scale + pos_offset`
	pub pos_offset: [f32; 3],
	pub pos_scale:  [f32; 3],
	pub lods:       Vec<Lod>,
}

/// One decoded vertex, in the space of its rigid part (or model space for skins)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vertex {
	pub pos:    [f32; 3],
	pub normal: [f32; 3],
	pub uv:     [f32; 2],
	/// Second texture coordinate set; zero when the vertex buffer has none
	pub uv1:    [f32; 2],
}

/// Dagor `IGenLoad::beginBlock`: u32 holding a 2-bit tag and a 30-bit size, then the block
pub(crate) fn read_block<'a>(r: &mut Cursor<'a>) -> Result<&'a [u8], DxpGrpError> {
	let head = r.u32()?;
	r.bytes((head & 0x3FFF_FFFF) as usize)
}

/// Compressed block body: a zstd frame, or u32 unpacked size followed by an Oodle stream
pub(crate) fn unpack_block(block: &[u8]) -> Result<Vec<u8>, DxpGrpError> {
	if block.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
		return zstd::stream::decode_all(block)
			.map_err(|e| DxpGrpError::Decompression(e.to_string()));
	}
	let mut r = Cursor::new(block);
	let size = r.u32()? as usize;
	unpack::oodle(r.rest(), size)
}

fn packed_index_size(w1: u32, w2: u32, flags: u32) -> usize {
	if flags & VDATA_PACKED_IB != 0 {
		((w1 >> 8) | ((w2 >> 28) << 24)) as usize
	} else {
		(w2 & 0x0FFF_FFFF) as usize
	}
}

/// ShaderMatVdata: the header size word also selects the compression (`ShaderMatVdata::loadMatVdata`)
fn read_mat_vdata(r: &mut Cursor, hdr_word: u32) -> Result<Vec<VData>, DxpGrpError> {
	if hdr_word & 0xE000_0000 == 0xC000_0000 {
		let vd = unpack_block(read_block(r)?)?;
		return read_vdata(&vd, (hdr_word & !0xC000_0000) as usize);
	}
	if hdr_word & 0xC000_0000 != 0 {
		return Err(model_err(
			"zlib or lzma packed vertex data is not supported",
		));
	}
	// Stored: header, then the buffers; the header gives their total size
	let hdr_size = hdr_word as usize;
	let header = r.bytes(hdr_size)?;
	let mut h = Cursor::at(header, 16)?;
	let (tab_ofs, count) = (h.u32()? as usize, h.u32()? as usize);
	let mut payload = 0_usize;
	for i in 0..count {
		let mut e = Cursor::at(header, tab_ofs + i * 32)?;
		let (vert_num, w1, w2, flags) = (e.u32()? as usize, e.u32()?, e.u32()?, e.u32()?);
		payload += vert_num * (w1 & 0xFF) as usize + packed_index_size(w1, w2, flags);
	}
	let mut all = header.to_vec();
	all.extend_from_slice(r.bytes(payload)?);
	read_vdata(&all, hdr_size)
}

fn read_vdata(vd: &[u8], payload_start: usize) -> Result<Vec<VData>, DxpGrpError> {
	let mut h = Cursor::new(vd);
	h.skip(16)?; // material table, empty when materials live in dynModelDesc.bin
	let (tab_ofs, count) = (h.u32()? as usize, h.u32()? as usize);
	let mut payload = Cursor::at(vd, payload_start)?;
	check_table(vd, tab_ofs, count, 32)?;
	let mut out = Vec::with_capacity(count);
	for i in 0..count {
		let mut e = Cursor::at(vd, tab_ofs + i * 32)?;
		let vert_num = e.u32()? as usize;
		let w1 = e.u32()?;
		let w2 = e.u32()?;
		let flags = e.u32()?;
		let (decl_ofs, decl_n) = (e.u32()? as usize, e.u32()? as usize);
		let stride = (w1 & 0xFF) as usize;
		let idx_size = (w2 & 0x0FFF_FFFF) as usize;
		// GlobalVertexData::initGvd: more than 64k vertices means 32-bit indices unless forced to 16
		let wide = flags & VDATA_I32 != 0 || (vert_num > 65536 && flags & VDATA_I16 == 0);
		let idx_count = idx_size / if wide { 4 } else { 2 };

		check_table(vd, decl_ofs, decl_n, 12)?;
		let mut channels = Vec::with_capacity(decl_n);
		let mut d = Cursor::at(vd, decl_ofs)?;
		let mut offset = 0;
		for _ in 0..decl_n {
			let ty = d.u32()? >> 16;
			let usage = d.u32()?;
			let index = d.u16()?;
			d.skip(2)?;
			channels.push(Channel {
				ty,
				usage,
				index,
				offset,
			});
			offset += match ty {
				E3DCOLOR | UBYTE4 | SHORT2 | SHORT2N | HALF2 => 4,
				FLOAT2 | SHORT4 | SHORT4N | HALF4 => 8,
				FLOAT3 => 12,
				FLOAT4 => 16,
				other => return Err(model_err(format!("vertex channel type {other}"))),
			};
		}
		if offset != stride {
			return Err(model_err(format!(
				"vertex declaration covers {offset} of {stride} bytes"
			)));
		}

		let vertices = payload.bytes(vert_num * stride)?.to_vec();
		let raw = payload.bytes(packed_index_size(w1, w2, flags))?;
		let indices = if flags & VDATA_PACKED_IB != 0 {
			decode_index_sequence(raw, idx_count, wide)?
		} else if wide {
			raw.chunks_exact(4)
				.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
				.collect()
		} else {
			raw.chunks_exact(2)
				.map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
				.collect()
		};
		out.push(VData {
			stride,
			vertices,
			indices,
			channels,
		});
	}
	if !payload.rest().is_empty() {
		return Err(model_err("vertex data has trailing bytes"));
	}
	Ok(out)
}

/// ShaderMesh at `at` inside `dump`, whose pointers are relative to the start of `dump`
fn read_mesh(dump: &[u8], at: usize, vdata: &[VData]) -> Result<Vec<Elem>, DxpGrpError> {
	let mut r = Cursor::at(dump, at)?;
	let (elems_ofs, count) = (r.u32()? as usize, r.u32()? as usize);
	check_table(dump, elems_ofs, count, 48)?;
	let mut elems = Vec::with_capacity(count);
	for i in 0..count {
		let mut e = Cursor::at(dump, elems_ofs + i * 48)?;
		e.skip(8)?; // shader element, set at runtime
		let material = e.u64_usize()?;
		let vd = e.u64_usize()?;
		e.skip(12)?; // vdata order index, vertex range (D3D9 hints)
		let elem = Elem {
			material,
			vdata: vd,
			first_index: e.u32()? as usize,
			triangles: e.u32()? as usize,
			base_vertex: e.u32()? as usize,
		};
		// Validate here so users can index without checks
		let buffer = vdata.get(elem.vdata).ok_or_else(|| {
			model_err(format!("element uses missing vertex buffer {}", elem.vdata))
		})?;
		let range = buffer
			.indices
			.get(elem.first_index..elem.first_index + elem.triangles * 3)
			.ok_or_else(|| model_err("element index range is out of bounds"))?;
		if range
			.iter()
			.any(|&i| i as usize + elem.base_vertex >= buffer.vertex_count())
		{
			return Err(model_err("element references a vertex out of bounds"));
		}
		elems.push(elem);
	}
	Ok(elems)
}

impl DynModel {
	pub fn parse(res: &[u8]) -> Result<Self, DxpGrpError> {
		let mut r = Cursor::new(res);
		let lods_dump_size = r.u32()? as usize;
		let (tex_count, _mat_count, _vdata_count, hdr_size) =
			(r.u32()?, r.u32()?, r.u32()?, r.u32()?);
		if tex_count != 0xFFFF_FFFF {
			return Err(model_err(
				"models with embedded materials are not supported",
			));
		}
		let vdata = read_mat_vdata(&mut r, hdr_size)?;

		let lods_dump = r.bytes(lods_dump_size)?;
		let vec3 = |at: usize| -> Result<[f32; 3], DxpGrpError> {
			let mut c = Cursor::at(lods_dump, at)?;
			Ok([c.f32()?, c.f32()?, c.f32()?])
		};
		let (pos_offset, pos_scale) = (vec3(0x28)?, vec3(0x38)?);

		// RoNameMapEx: names sorted for binary search, plus the node id of each sorted slot
		let names_size = r.u32()? as usize;
		let names_dump = r.bytes(names_size)?;
		let mut nr = Cursor::new(names_dump);
		let (names_ofs, names_count) = (nr.u32()? as usize, nr.u32()? as usize);
		nr.skip(8)?;
		let (ids_ofs, ids_count) = (nr.u32()? as usize, nr.u32()? as usize);
		if ids_count != names_count {
			return Err(model_err("node name map and id table differ in size"));
		}
		check_table(names_dump, names_ofs, names_count, 8)?;
		check_table(names_dump, ids_ofs, ids_count, 2)?;
		let budget = NameBudget::new(names_dump);
		let mut node_names = vec![String::new(); names_count];
		for i in 0..names_count {
			let name_ofs = Cursor::at(names_dump, names_ofs + i * 8)?.u64_usize()?;
			let id = Cursor::at(names_dump, ids_ofs + i * 2)?.u16()? as usize;
			*node_names
				.get_mut(id)
				.ok_or_else(|| model_err("node id out of range"))? =
				budget.take(Cursor::at(names_dump, name_ofs)?.cstr()?)?;
		}
		// Rigids and skins take copies of the names
		let copies = NameBudget::new(res);
		let node_name = |id: usize| {
			node_names
				.get(id)
				.cloned()
				.ok_or_else(|| model_err(format!("node id {id} out of range")))
				.and_then(|name| copies.take(name))
		};

		let mut lt = Cursor::new(lods_dump);
		let (lod_ofs, lod_count) = (lt.u32()? as usize, lt.u32()? as usize);
		check_table(lods_dump, lod_ofs, lod_count, 16)?;
		let mut lods = Vec::with_capacity(lod_count);
		let mut skin_counts = Vec::with_capacity(lod_count);
		for l in 0..lod_count {
			let mut rec = Cursor::at(lods_dump, lod_ofs + l * 16)?;
			let scene_size = rec.u64_usize()?;
			let range = rec.f32()?;
			let scene = r.bytes(scene_size)?;
			let mut sr = Cursor::new(scene);
			let (rig_ofs, rig_count) = (sr.u32()? as usize, sr.u32()? as usize);
			sr.skip(12)?;
			let skin_count = sr.u32()? as usize;
			check_table(scene, rig_ofs, rig_count, 32)?;
			let mut rigids = Vec::with_capacity(rig_count);
			for i in 0..rig_count {
				let mut ro = Cursor::at(scene, rig_ofs + i * 32)?;
				let mesh_size = ro.u64_usize()?;
				ro.skip(16)?; // bounding sphere
				let node = node_name(ro.u32()? as usize)?;
				rigids.push(Rigid {
					node,
					elems: read_mesh(r.bytes(mesh_size)?, 0, &vdata)?,
				});
			}
			lods.push(Lod {
				range,
				rigids,
				// Skins are read later, from a compressed section
				skins: Vec::new(),
			});
			skin_counts.push(skin_count);
		}

		let skin_vdata = if skin_counts.iter().any(|&n| n > 0) {
			read_skins(&mut r, &mut lods, &skin_counts, &node_name)?
		} else {
			Vec::new()
		};
		Ok(Self {
			vdata,
			skin_vdata,
			pos_offset,
			pos_scale,
			lods,
		})
	}

	/// Decodes vertex `index` of one of this model's vertex buffers (rigid or skinned)
	pub fn vertex(&self, vdata: &VData, index: usize) -> Result<Vertex, DxpGrpError> {
		let v = index
			.checked_mul(vdata.stride)
			.and_then(|start| vdata.vertices.get(start..start.checked_add(vdata.stride)?))
			.ok_or_else(|| model_err(format!("vertex {index} out of range")))?;
		let i16_at = |o: usize| f32::from(i16::from_le_bytes([v[o], v[o + 1]]));
		let f32_at = |o: usize| f32::from_le_bytes([v[o], v[o + 1], v[o + 2], v[o + 3]]);
		let mut out = Vertex::default();
		for c in &vdata.channels {
			let o = c.offset;
			match (c.usage, c.index, c.ty) {
				(POSITION, _, SHORT4N) => {
					for k in 0..3 {
						out.pos[k] =
							i16_at(o + k * 2) / 32767.0 * self.pos_scale[k] + self.pos_offset[k];
					}
				},
				(POSITION, _, FLOAT3 | FLOAT4) => {
					out.pos = [f32_at(o), f32_at(o + 4), f32_at(o + 8)]
				},
				// E3DCOLOR is stored as B, G, R, A
				(NORMAL, _, E3DCOLOR) => {
					out.normal = [v[o + 2], v[o + 1], v[o]].map(|x| f32::from(x) / 127.5 - 1.0)
				},
				(TEXCOORD, 0, SHORT2) => {
					out.uv = [i16_at(o), i16_at(o + 2)].map(|x| x * TEXCOORD_SCALE)
				},
				(TEXCOORD, 1, SHORT2) => {
					out.uv1 = [i16_at(o), i16_at(o + 2)].map(|x| x * TEXCOORD_SCALE)
				},
				(TEXCOORD, 0, FLOAT2) => out.uv = [f32_at(o), f32_at(o + 4)],
				(TEXCOORD, 1, FLOAT2) => out.uv1 = [f32_at(o), f32_at(o + 4)],
				_ => {},
			}
		}
		Ok(out)
	}
}

/// Skinned meshes follow the LODs: a table of variants per GPU constant budget, vertex data like the
/// rigid one, then a zlib stream holding, per LOD, the skin node ids, their sizes and the
/// ShaderSkinnedMesh dumps (bone tables, then a ShaderMesh at +0x38)
fn read_skins(
	r: &mut Cursor,
	lods: &mut [Lod],
	counts: &[usize],
	node_name: &dyn Fn(usize) -> Result<String, DxpGrpError>,
) -> Result<Vec<VData>, DxpGrpError> {
	/// Pick the variant a D3D11-class GPU would use
	const MAX_VS_CONSTS: u32 = 4096;
	let section = r.pos();
	let (variants, total) = (r.u32()?, r.u32()? as usize);
	let mut chosen = false;
	for _ in 0..variants {
		let (consts, skip) = (r.u32()?, r.u32()? as usize);
		if consts <= MAX_VS_CONSTS {
			r.skip(skip)?;
			chosen = true;
			break;
		}
	}
	if !chosen {
		return Err(model_err("no skinned mesh variant fits"));
	}

	let (tex_count, _mat_count, _vdata_count, hdr_size) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
	if tex_count != 0xFFFF_FFFF {
		return Err(model_err("skins with embedded materials are not supported"));
	}
	let vdata = read_mat_vdata(r, hdr_size)?;

	let end = section + 8 + total;
	let zipped = r.bytes(
		end.checked_sub(r.pos())
			.ok_or_else(|| model_err("skin section overrun"))?,
	)?;
	let mut unzipped = Vec::new();
	flate2::read::ZlibDecoder::new(zipped)
		.read_to_end(&mut unzipped)
		.map_err(|e| DxpGrpError::Decompression(e.to_string()))?;
	let mut z = Cursor::new(&unzipped);
	for (lod, &count) in lods.iter_mut().zip(counts) {
		let nodes = (0..count).map(|_| z.u32()).collect::<Result<Vec<_>, _>>()?;
		let sizes = (0..count)
			.map(|_| z.u64_usize())
			.collect::<Result<Vec<_>, _>>()?;
		for (node, size) in nodes.into_iter().zip(sizes) {
			if size & 0x4000_0000 == 0 {
				return Err(model_err("obsolete skinned mesh format"));
			}
			let dump = z.bytes(size & !0x4000_0000)?;
			lod.skins.push(Skin {
				node:  node_name(node as usize)?,
				elems: read_mesh(dump, 0x38, &vdata)?,
			});
		}
	}
	Ok(vdata)
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn vertex_index_out_of_range() {
		let model = DynModel {
			vdata:      Vec::new(),
			skin_vdata: Vec::new(),
			pos_offset: [0.0; 3],
			pos_scale:  [1.0; 3],
			lods:       Vec::new(),
		};
		let vdata = VData {
			stride:   8,
			vertices: vec![1; 16],
			indices:  Vec::new(),
			channels: vec![Channel {
				ty:     FLOAT2,
				usage:  TEXCOORD,
				index:  0,
				offset: 0,
			}],
		};
		assert!(model.vertex(&vdata, 1).is_ok());
		assert!(model.vertex(&vdata, 2).is_err());
		// `index * stride` overflowed: a panic in debug builds, and vertex 0 in release builds
		assert!(model.vertex(&vdata, 1 << 61).is_err());
		assert!(model.vertex(&vdata, usize::MAX).is_err());
	}
}
