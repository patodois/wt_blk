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
//! A skinned mesh (`ShaderSkinnedMesh`, `prog/engine/shaders/shSkinnedMesh.cpp`) is a dump whose
//! pointers are relative to its start:
//!
//! ```text
//! +0x00 bone table      {u32 offset, u32 count, 8 bytes}  48-byte TMatrix per bone (`boneOrgTm`), the
//!                       transform from model space to the bone's space in the bind pose
//! +0x10 material table  {u32 offset, u32 count, 8 bytes}  per skinned material: a palette
//!                       {u32 offset, u32 count, 8 bytes} of i16 bone-table indices; empty when the
//!                       palette is the bone table itself
//! +0x20 u32 offset      i16 skeleton node id of each bone (`boneNodeId`)
//! +0x28 u32 offset      u64 per skinned material: its index in the model's `matS` list
//! +0x30 u64             work data, set at runtime
//! +0x38 ShaderMesh
//! ```
//!
//! A vertex names its bones by palette slot. The tool that builds skins (`shSkinMeshData.cpp`) adds
//! them as `SCUSAGE_EXTRA` channels 0 (four slots, E3DCOLOR, or SHORT4 when the model has more than
//! 255 bones) and 1 (their four weights, E3DCOLOR), and 2 and 3 for a second group of four. In the
//! vertex declarations of the game's models they show up as texture coordinates 4 to 7. The renderer
//! works out each bone's matrix as `node_world_tm * boneOrgTm` (`recalcBones`) and blends them by
//! the weights.
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
	/// Extra channels `n` of skinned meshes (bone slots at 0, weights at 1, ...) are declared as texture
	/// coordinates numbered from this
	pub const EXTRA_TEXCOORD: u16 = 4;
}
use channel::*;

const EXTRA_TEXCOORD_WEIGHTS: u16 = EXTRA_TEXCOORD + 1;

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

	/// Number of groups of four bone influences each vertex carries (0 for rigid vertices)
	pub fn bone_groups(&self) -> usize {
		self.channels
			.iter()
			.filter(|c| c.usage == TEXCOORD && c.index >= EXTRA_TEXCOORD && c.index % 2 == 1)
			.count()
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

/// An entry of a skinned mesh's bone table
#[derive(Debug, Clone, PartialEq)]
pub struct Bone {
	/// Skeleton node (see [`crate::dxp_and_grp::geom_node_tree`]) that moves the bone
	pub node:         String,
	/// Maps model space to the bone's space in the bind pose (`boneOrgTm`), column-major. A vertex at
	/// `p` in bind pose lands at `node_wtm * inverse_bind * p` when the node has moved
	pub inverse_bind: [f32; 16],
}

/// One skinned material of a skinned mesh, with the bones its vertices can use
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkinMaterial {
	/// Index into the model's skinned material list (`matS`), as in [`Elem::material`]
	pub material: usize,
	/// The bone-table index of each palette slot that vertices of this material name. Empty when
	/// slots are bone-table indices already
	pub palette:  Vec<u16>,
}

/// A skinned mesh (tracks, cables), stored in bind pose in model space
#[derive(Debug, Clone)]
pub struct Skin {
	pub node:      String,
	pub elems:     Vec<Elem>,
	/// The bones the mesh is blended between, with the skeleton nodes that move them
	pub bones:     Vec<Bone>,
	pub materials: Vec<SkinMaterial>,
}

impl Skin {
	/// The bone-table entry a vertex of `material` (an [`Elem::material`]) means by palette `slot`
	pub fn bone(&self, material: usize, slot: usize) -> Option<&Bone> {
		let palette = &self
			.materials
			.iter()
			.find(|m| m.material == material)?
			.palette;
		let index = if palette.is_empty() {
			slot
		} else {
			usize::from(*palette.get(slot)?)
		};
		self.bones.get(index)
	}
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

/// The bone influences of one skinned vertex
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SkinWeights {
	/// Palette slots of the four bones that move the vertex (see [`Skin::bone`]); a slot with weight 0
	/// is unused
	pub joints:  [u16; 4],
	/// Their weights, which sum to 1 (up to rounding: they are stored as bytes). Only the first group
	/// of four influences is read; [`VData::bone_groups`] says when a buffer has more
	pub weights: [f32; 4],
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
	check_table(header, tab_ofs, count, 32)?;
	// Saturating: a total that does not fit in usize is past the end of the data anyway
	let mut payload = 0_usize;
	for i in 0..count {
		let mut e = Cursor::at(header, tab_ofs + i * 32)?;
		let (vert_num, w1, w2, flags) = (e.u32()? as usize, e.u32()?, e.u32()?, e.u32()?);
		payload = payload
			.saturating_add(vert_num.saturating_mul((w1 & 0xFF) as usize))
			.saturating_add(packed_index_size(w1, w2, flags));
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

		let vertices = payload.bytes(vert_num.saturating_mul(stride))?.to_vec();
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

/// ShaderMesh at `at` inside `dump`, whose pointers are relative to the start of `dump`: an element
/// table, then either the end index of each render stage's elements (bit 31 set in the word at +0x20)
/// or a second element table at +0x10, which holds the see-through parts (glass, sights, flares)
fn read_mesh(dump: &[u8], at: usize, vdata: &[VData]) -> Result<Vec<Elem>, DxpGrpError> {
	let mut r = Cursor::at(dump, at)?;
	let main = (r.u32()? as usize, r.u32()? as usize);
	r.skip(8)?;
	let second = (r.u32()? as usize, r.u32()? as usize);
	r.skip(8)?;
	let tables = if r.u32()? & 0x8000_0000 != 0 {
		&[main][..]
	} else {
		&[main, second][..]
	};
	let mut elems = Vec::new();
	for &(elems_ofs, count) in tables {
		check_table(dump, elems_ofs, count, 48)?;
		elems.reserve(count);
		read_elems(dump, elems_ofs, count, vdata, &mut elems)?;
	}
	Ok(elems)
}

fn read_elems(
	dump: &[u8],
	elems_ofs: usize,
	count: usize,
	vdata: &[VData],
	elems: &mut Vec<Elem>,
) -> Result<(), DxpGrpError> {
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
		let range = elem
			.triangles
			.checked_mul(3)
			.and_then(|n| n.checked_add(elem.first_index))
			.and_then(|end| buffer.indices.get(elem.first_index..end))
			.ok_or_else(|| model_err("element index range is out of bounds"))?;
		if range.iter().any(|&i| {
			(i as usize)
				.checked_add(elem.base_vertex)
				.is_none_or(|v| v >= buffer.vertex_count())
		}) {
			return Err(model_err("element references a vertex out of bounds"));
		}
		elems.push(elem);
	}
	Ok(())
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

	/// The bone slots and weights of vertex `index` of one of this model's skinned vertex buffers
	/// ([`DynModel::skin_vdata`]). Vertices of buffers without bones have no influences (all weights 0)
	pub fn skin_weights(&self, vdata: &VData, index: usize) -> Result<SkinWeights, DxpGrpError> {
		let v = index
			.checked_mul(vdata.stride)
			.and_then(|start| vdata.vertices.get(start..start.checked_add(vdata.stride)?))
			.ok_or_else(|| model_err(format!("vertex {index} out of range")))?;
		let u16_at = |o: usize| u16::from_le_bytes([v[o], v[o + 1]]);
		let f32_at = |o: usize| f32::from_le_bytes([v[o], v[o + 1], v[o + 2], v[o + 3]]);
		let mut out = SkinWeights::default();
		for c in vdata.channels.iter().filter(|c| c.usage == TEXCOORD) {
			let o = c.offset;
			match (c.index, c.ty) {
				// E3DCOLOR is stored as B, G, R, A, so x, y, z, w are bytes 2, 1, 0, 3
				(EXTRA_TEXCOORD, E3DCOLOR) => {
					out.joints = [v[o + 2], v[o + 1], v[o], v[o + 3]].map(u16::from)
				},
				(EXTRA_TEXCOORD, SHORT4) => out.joints = [0, 1, 2, 3].map(|k| u16_at(o + k * 2)),
				(EXTRA_TEXCOORD_WEIGHTS, E3DCOLOR) => {
					out.weights = [v[o + 2], v[o + 1], v[o], v[o + 3]].map(|x| f32::from(x) / 255.0)
				},
				(EXTRA_TEXCOORD_WEIGHTS, FLOAT4) => {
					out.weights = [f32_at(o), f32_at(o + 4), f32_at(o + 8), f32_at(o + 12)]
				},
				_ => {},
			}
		}
		Ok(out)
	}
}

/// The bone table, palettes and skinned materials at the start of a ShaderSkinnedMesh dump
fn read_bone_table(
	dump: &[u8],
	node_name: &dyn Fn(usize) -> Result<String, DxpGrpError>,
) -> Result<(Vec<Bone>, Vec<SkinMaterial>), DxpGrpError> {
	let mut r = Cursor::new(dump);
	let (tm_ofs, bone_count) = (r.u32()? as usize, r.u32()? as usize);
	r.skip(8)?;
	let (mat_ofs, mat_count) = (r.u32()? as usize, r.u32()? as usize);
	r.skip(8)?;
	let (node_ofs, shmat_ofs) = (r.u64_usize()?, r.u64_usize()?);
	check_table(dump, tm_ofs, bone_count, 48)?;
	check_table(dump, node_ofs, bone_count, 2)?;
	check_table(dump, mat_ofs, mat_count, 16)?;
	check_table(dump, shmat_ofs, mat_count, 8)?;

	let mut bones = Vec::with_capacity(bone_count);
	for i in 0..bone_count {
		let node = Cursor::at(dump, node_ofs + i * 2)?.u16()?;
		let mut tm = Cursor::at(dump, tm_ofs + i * 48)?;
		// TMatrix: three columns of axes, then the translation, as 12 floats
		let mut inverse_bind = [
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		];
		for col in 0..4 {
			for row in 0..3 {
				inverse_bind[col * 4 + row] = tm.f32()?;
			}
		}
		bones.push(Bone {
			node: node_name(node as usize)?,
			inverse_bind,
		});
	}

	let mut materials = Vec::with_capacity(mat_count);
	for i in 0..mat_count {
		let material = Cursor::at(dump, shmat_ofs + i * 8)?.u32()? as usize;
		let mut p = Cursor::at(dump, mat_ofs + i * 16)?;
		let (ofs, count) = (p.u32()? as usize, p.u32()? as usize);
		check_table(dump, ofs, count, 2)?;
		let mut palette = Vec::with_capacity(count);
		let mut slots = Cursor::at(dump, ofs)?;
		for _ in 0..count {
			let bone = slots.u16()?;
			if usize::from(bone) >= bone_count {
				return Err(model_err("palette names a bone outside the bone table"));
			}
			palette.push(bone);
		}
		materials.push(SkinMaterial { material, palette });
	}
	Ok((bones, materials))
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

	let end = section.saturating_add(8).saturating_add(total);
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
			let (bones, materials) = read_bone_table(dump, node_name)?;
			lod.skins.push(Skin {
				node: node_name(node as usize)?,
				elems: read_mesh(dump, 0x38, &vdata)?,
				bones,
				materials,
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

	fn push_u32(out: &mut Vec<u8>, v: u32) {
		out.extend_from_slice(&v.to_le_bytes());
	}

	fn push_u64(out: &mut Vec<u8>, v: u64) {
		out.extend_from_slice(&v.to_le_bytes());
	}

	/// A ShaderSkinnedMesh dump with `bones` (node id, translation of its inverse bind matrix),
	/// skinned materials given as (matS index, palette) and an empty ShaderMesh
	fn skinned_mesh_dump(bones: &[(i16, [f32; 3])], materials: &[(u32, Vec<i16>)]) -> Vec<u8> {
		skinned_mesh_dump_with(bones, materials, &[])
	}

	/// Like `skinned_mesh_dump`, with elements `(material, vdata, first_index, triangles)`
	fn skinned_mesh_dump_with(
		bones: &[(i16, [f32; 3])],
		materials: &[(u32, Vec<i16>)],
		elems: &[(u64, u64, u32, u32)],
	) -> Vec<u8> {
		// Header (0x38 bytes) and ShaderMesh (0x28 bytes), then the arrays they point at
		let mut tail = Vec::new();
		let base = 0x38 + 0x28;
		let node_ofs = base + tail.len();
		for b in bones {
			tail.extend_from_slice(&b.0.to_le_bytes());
		}
		while tail.len() % 8 != 0 {
			tail.push(0);
		}
		let shmat_ofs = base + tail.len();
		for m in materials {
			push_u64(&mut tail, u64::from(m.0));
		}
		let tm_ofs = base + tail.len();
		for b in bones {
			// Rotation columns of the identity, then the translation
			for v in [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] {
				tail.extend_from_slice(&v.to_le_bytes());
			}
			for v in b.1 {
				tail.extend_from_slice(&v.to_le_bytes());
			}
		}
		let mut palette_ofs = Vec::new();
		for m in materials {
			palette_ofs.push(base + tail.len());
			for slot in &m.1 {
				tail.extend_from_slice(&slot.to_le_bytes());
			}
		}
		while tail.len() % 8 != 0 {
			tail.push(0);
		}
		let mat_ofs = base + tail.len();
		for (m, ofs) in materials.iter().zip(&palette_ofs) {
			push_u32(&mut tail, if m.1.is_empty() { 0 } else { *ofs as u32 });
			push_u32(&mut tail, m.1.len() as u32);
			push_u64(&mut tail, 0);
		}
		let elems_ofs = base + tail.len();
		for e in elems {
			push_u64(&mut tail, 0);
			push_u64(&mut tail, e.0);
			push_u64(&mut tail, e.1);
			tail.extend_from_slice(&[0; 12]);
			push_u32(&mut tail, e.2);
			push_u32(&mut tail, e.3);
			push_u32(&mut tail, 0);
		}

		let mut out = Vec::new();
		for (ofs, count) in [(tm_ofs, bones.len()), (mat_ofs, materials.len())] {
			push_u32(&mut out, ofs as u32);
			push_u32(&mut out, count as u32);
			push_u64(&mut out, 0);
		}
		push_u64(&mut out, node_ofs as u64);
		push_u64(&mut out, shmat_ofs as u64);
		push_u64(&mut out, 0);
		// ShaderMesh: element table, an unused second one, and the flag that says the first holds all
		push_u32(&mut out, elems_ofs as u32);
		push_u32(&mut out, elems.len() as u32);
		push_u64(&mut out, 0);
		push_u64(&mut out, 0);
		push_u64(&mut out, 0);
		push_u32(&mut out, 0x8000_0000);
		push_u32(&mut out, 0);
		assert_eq!(out.len(), base);
		out.extend_from_slice(&tail);
		out
	}

	fn names(id: i16) -> Result<String, DxpGrpError> {
		Ok(format!("node{id}"))
	}

	#[test]
	fn bone_table_names_nodes_and_palettes() {
		let dump = skinned_mesh_dump(
			&[
				(3, [0.0, 1.0, 2.0]),
				(1, [4.0, 5.0, 6.0]),
				(2, [7.0, 8.0, 9.0]),
			],
			&[(0, vec![]), (2, vec![2, 0])],
		);
		let (bones, materials) = read_bone_table(&dump, &|id| names(id as i16)).unwrap();
		assert_eq!(
			bones.iter().map(|b| b.node.as_str()).collect::<Vec<_>>(),
			["node3", "node1", "node2"]
		);
		// Column-major, translation in the last column
		assert_eq!(bones[1].inverse_bind[12..], [4.0, 5.0, 6.0, 1.0]);
		assert_eq!(bones[1].inverse_bind[..3], [1.0, 0.0, 0.0]);
		assert_eq!(bones[1].inverse_bind[5], 1.0);
		assert_eq!(
			materials,
			[
				SkinMaterial {
					material: 0,
					palette:  vec![],
				},
				SkinMaterial {
					material: 2,
					palette:  vec![2, 0],
				}
			]
		);

		let skin = Skin {
			node: "track".into(),
			elems: Vec::new(),
			bones,
			materials,
		};
		// A material without a palette uses slots as bone-table indices; others map through theirs
		assert_eq!(skin.bone(0, 1).unwrap().node, "node1");
		assert_eq!(skin.bone(2, 0).unwrap().node, "node2");
		assert_eq!(skin.bone(2, 1).unwrap().node, "node3");
		assert!(
			skin.bone(2, 2).is_none() && skin.bone(0, 3).is_none() && skin.bone(5, 0).is_none()
		);
	}

	#[test]
	fn bad_bone_tables_are_errors() {
		let good = skinned_mesh_dump(&[(0, [0.0; 3]); 2], &[(0, vec![1, 0])]);
		assert!(read_bone_table(&good, &|id| names(id as i16)).is_ok());
		// A palette entry past the bone table
		let bad = skinned_mesh_dump(&[(0, [0.0; 3]); 2], &[(0, vec![1, 2])]);
		assert!(read_bone_table(&bad, &|id| names(id as i16)).is_err());
		// Counts that do not fit the dump are rejected before anything is allocated
		for at in [4, 20] {
			let mut huge = good.clone();
			put_u32(&mut huge, at, u32::MAX);
			assert!(read_bone_table(&huge, &|id| names(id as i16)).is_err());
		}
		// A bone whose node does not exist
		let missing =
			|id: usize| -> Result<String, DxpGrpError> { Err(model_err(format!("node id {id}"))) };
		assert!(read_bone_table(&good, &missing).is_err());
		assert!(read_bone_table(&good[..0x20], &|id| names(id as i16)).is_err());
	}

	fn put_u32(buf: &mut [u8], at: usize, v: u32) {
		buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
	}

	/// A vertex buffer of 8-byte vertices holding only the bone channels: slots and weights as E3DCOLORs
	fn bone_vdata(vertices: &[([u8; 4], [u8; 4])], indices: &[u16]) -> VData {
		let mut bytes = Vec::new();
		for (slots, weights) in vertices {
			bytes.extend_from_slice(slots);
			bytes.extend_from_slice(weights);
		}
		let channel = |index, offset| Channel {
			ty: E3DCOLOR,
			usage: TEXCOORD,
			index,
			offset,
		};
		VData {
			stride:   8,
			vertices: bytes,
			indices:  indices.iter().map(|&i| u32::from(i)).collect(),
			channels: vec![channel(4, 0), channel(5, 4)],
		}
	}

	fn empty_model() -> DynModel {
		DynModel {
			vdata:      Vec::new(),
			skin_vdata: Vec::new(),
			pos_offset: [0.0; 3],
			pos_scale:  [1.0; 3],
			lods:       Vec::new(),
		}
	}

	#[test]
	fn skin_weights_follow_dagors_color_byte_order() {
		// E3DCOLOR(x, y, z, w) is stored B, G, R, A: slot 0 is the third byte
		let vd = bone_vdata(
			&[
				([2, 1, 0, 3], [0, 64, 191, 0]),
				([7, 0, 0, 0], [0, 0, 0, 255]),
			],
			&[0, 1],
		);
		assert_eq!(vd.bone_groups(), 1);
		let model = empty_model();
		let a = model.skin_weights(&vd, 0).unwrap();
		assert_eq!(a.joints, [0, 1, 2, 3]);
		assert_eq!(a.weights, [191.0 / 255.0, 64.0 / 255.0, 0.0, 0.0]);
		assert!((a.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
		// The fourth weight is the alpha byte
		let b = model.skin_weights(&vd, 1).unwrap();
		assert_eq!((b.joints, b.weights), ([0, 0, 7, 0], [0.0, 0.0, 0.0, 1.0]));
		assert!(model.skin_weights(&vd, 2).is_err());

		// Rigid buffers have no influences
		let rigid = VData {
			stride:   8,
			vertices: vec![9; 8],
			indices:  Vec::new(),
			channels: vec![Channel {
				ty:     FLOAT2,
				usage:  TEXCOORD,
				index:  0,
				offset: 0,
			}],
		};
		assert_eq!(rigid.bone_groups(), 0);
		assert_eq!(
			model.skin_weights(&rigid, 0).unwrap(),
			SkinWeights::default()
		);
	}

	/// A whole model with no rigid parts and one skin, built the way the game stores it: this goes
	/// through the LOD table, the name map, the skin variant table and the zlib section
	#[test]
	fn skins_come_with_their_bones_and_weights() {
		use std::io::Write;

		// Vertex buffer for the skin: three vertices, one triangle; stored (uncompressed)
		let vertices: [([u8; 4], [u8; 4]); 3] = [
			([0, 0, 0, 0], [0, 0, 0, 255]),
			([1, 0, 0, 0], [0, 0, 0, 255]),
			([1, 0, 0, 0], [0, 0, 0, 128]),
		];
		let mut vdata = Vec::new();
		// Header: material table (empty), vertex buffer table, then the buffer's channels
		vdata.extend_from_slice(&[0; 16]);
		push_u32(&mut vdata, 24);
		push_u32(&mut vdata, 1);
		for v in [3, 8, 6, 0, 56, 2] {
			push_u32(&mut vdata, v);
		}
		push_u64(&mut vdata, 0);
		for index in [4_u16, 5] {
			push_u32(&mut vdata, E3DCOLOR << 16);
			push_u32(&mut vdata, TEXCOORD);
			vdata.extend_from_slice(&index.to_le_bytes());
			vdata.extend_from_slice(&[0, 0]);
		}
		let header = vdata.len();
		for (slots, weights) in &vertices {
			vdata.extend_from_slice(slots);
			vdata.extend_from_slice(weights);
		}
		for i in [0_u16, 1, 2] {
			vdata.extend_from_slice(&i.to_le_bytes());
		}

		// Node names, sorted for binary search ("hull" is node 0, "track" node 1): a table of name
		// pointers at 32, the node id of each at 48, then the strings
		let mut names_dump = vec![0_u8; 32];
		put_u32(&mut names_dump, 0, 32);
		put_u32(&mut names_dump, 4, 2);
		put_u32(&mut names_dump, 16, 48);
		put_u32(&mut names_dump, 20, 2);
		push_u64(&mut names_dump, 52);
		push_u64(&mut names_dump, 57);
		names_dump.extend_from_slice(&[0, 0, 1, 0]);
		names_dump.extend_from_slice(b"hull\0track\0");

		// The skinned mesh: bones are the hull (node 0) and the track (node 1)
		let mesh = skinned_mesh_dump_with(
			&[(0, [1.0, 2.0, 3.0]), (1, [4.0, 5.0, 6.0])],
			&[(0, vec![])],
			&[(0, 0, 0, 1)],
		);
		let mut inner = Vec::new();
		push_u32(&mut inner, 1);
		push_u64(&mut inner, mesh.len() as u64 | 0x4000_0000);
		inner.extend_from_slice(&mesh);
		let mut zlib = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
		zlib.write_all(&inner).unwrap();
		let zipped = zlib.finish().unwrap();

		// LOD table: one LOD whose scene has no rigid parts and one skin
		let mut lods_dump = vec![0_u8; 0x44];
		put_u32(&mut lods_dump, 0, 0x44);
		put_u32(&mut lods_dump, 4, 1);
		let mut rec = Vec::new();
		push_u64(&mut rec, 24);
		rec.extend_from_slice(&100.0_f32.to_le_bytes());
		push_u32(&mut rec, 0);
		lods_dump.extend_from_slice(&rec);
		let mut scene = vec![0_u8; 24];
		put_u32(&mut scene, 20, 1);

		let mut res = Vec::new();
		push_u32(&mut res, lods_dump.len() as u32);
		for v in [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 24] {
			push_u32(&mut res, v);
		}
		// Rigid vertex data: a header with an empty buffer table
		let mut rigid = vec![0_u8; 24];
		put_u32(&mut rigid, 16, 24);
		res.extend_from_slice(&rigid);
		res.extend_from_slice(&lods_dump);
		push_u32(&mut res, names_dump.len() as u32);
		res.extend_from_slice(&names_dump);
		res.extend_from_slice(&scene);
		// Skin section: one variant, then vertex data like the rigid one, then the zlib stream
		let section_body = {
			let mut b = Vec::new();
			push_u32(&mut b, 0);
			push_u32(&mut b, 0);
			for v in [0xFFFF_FFFF, 0xFFFF_FFFF, 1, header as u32] {
				push_u32(&mut b, v);
			}
			b.extend_from_slice(&vdata);
			b.extend_from_slice(&zipped);
			b
		};
		push_u32(&mut res, 1);
		push_u32(&mut res, section_body.len() as u32);
		res.extend_from_slice(&section_body);

		let model = DynModel::parse(&res).unwrap();
		let lod = &model.lods[0];
		assert_eq!(lod.skins.len(), 1);
		let skin = &lod.skins[0];
		assert_eq!(skin.node, "track");
		assert_eq!(
			skin.bones
				.iter()
				.map(|b| b.node.as_str())
				.collect::<Vec<_>>(),
			["hull", "track"]
		);
		assert_eq!(skin.bones[1].inverse_bind[12..15], [4.0, 5.0, 6.0]);
		assert_eq!(
			skin.materials,
			[SkinMaterial {
				material: 0,
				palette:  vec![],
			}]
		);
		let e = skin.elems[0];
		assert_eq!((e.triangles, e.vdata), (1, 0));
		let vd = &model.skin_vdata[e.vdata];
		let last = model.skin_weights(vd, 2).unwrap();
		// The slot is the first byte of the E3DCOLOR, which is its third component
		assert_eq!(last.joints, [0, 0, 1, 0]);
		assert_eq!(last.weights, [0.0, 0.0, 0.0, 128.0 / 255.0]);
		assert_eq!(
			skin.bone(e.material, last.joints[2] as usize).unwrap().node,
			"track"
		);
		assert_eq!(
			model.skin_weights(vd, 0).unwrap().weights,
			[0.0, 0.0, 0.0, 1.0]
		);
	}
}
