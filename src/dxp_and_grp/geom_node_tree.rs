//! GeomNodeTree resources: the node hierarchy (skeleton) that places a DynModel's rigid parts
//!
//! Stream (`GeomNodeTree::load` in Dagor's `prog/engine/math/geomTree.cpp`): u32 dump size (low 20
//! bits), u32 node count (top bit: the dump is in a compressed block), then the dump. Each node takes
//! 160 bytes: local and world 4x4 matrices (column-major), a child table, a parent pointer and a name
//! pointer, all relative to the start of the dump.

use crate::dxp_and_grp::{
	cursor::Cursor,
	dynmodel::{read_block, unpack_block},
	error::DxpGrpError,
};

const NODE_SIZE: usize = 160;

#[derive(Debug, Clone, PartialEq)]
pub struct GeomNode {
	pub name:   String,
	pub parent: Option<usize>,
	/// Transform relative to the parent, column-major
	pub tm:     [f32; 16],
	/// Transform in model space, column-major
	pub wtm:    [f32; 16],
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomNodeTree {
	pub nodes: Vec<GeomNode>,
}

impl GeomNodeTree {
	pub fn parse(res: &[u8]) -> Result<Self, DxpGrpError> {
		let mut r = Cursor::new(res);
		let dump_size = (r.u32()? & 0xF_FFFF) as usize;
		let count_raw = r.u32()?;
		let count = (count_raw & 0x7FFF_FFFF) as usize;
		let data = if count_raw & 0x8000_0000 != 0 {
			unpack_block(read_block(&mut r)?)?
		} else {
			r.rest().to_vec()
		};
		if data.len() < dump_size || dump_size < count * NODE_SIZE {
			return Err(DxpGrpError::Model("node tree data is truncated".to_owned()));
		}

		let mut nodes = Vec::with_capacity(count);
		for i in 0..count {
			let mut n = Cursor::at(&data, i * NODE_SIZE)?;
			let mut matrix = || -> Result<[f32; 16], DxpGrpError> {
				let mut m = [0_f32; 16];
				for v in m.iter_mut() {
					*v = n.f32()?;
				}
				Ok(m)
			};
			let tm = matrix()?;
			let wtm = matrix()?;
			n.skip(16)?; // children: offset and count
			let parent_ofs = n.u64_usize()?;
			let name_ofs = n.u64_usize()?;
			// Node 0 is the root; its parent pointer is null, which reads like an offset of 0
			let parent = if i == 0 {
				None
			} else if parent_ofs % NODE_SIZE == 0 && parent_ofs / NODE_SIZE < count {
				Some(parent_ofs / NODE_SIZE)
			} else {
				return Err(DxpGrpError::Model(format!(
					"node {i} has an invalid parent"
				)));
			};
			nodes.push(GeomNode {
				name: Cursor::at(&data, name_ofs)?.cstr()?,
				parent,
				tm,
				wtm,
			});
		}
		Ok(Self { nodes })
	}

	pub fn find(&self, name: &str) -> Option<&GeomNode> {
		self.nodes.iter().find(|n| n.name == name)
	}
}
