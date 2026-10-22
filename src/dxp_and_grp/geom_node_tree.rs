//! GeomNodeTree resources: the node hierarchy (skeleton) that places a DynModel's rigid parts
//!
//! Stream (`GeomNodeTree::load` in Dagor's `prog/engine/math/geomTree.cpp`): u32 dump size (low 20
//! bits), u32 node count (top bit: the dump is in a compressed block), then the dump. Each node takes
//! 160 bytes: local and world 4x4 matrices (column-major), a child table, a parent pointer and a name
//! pointer, all relative to the start of the dump.

use crate::dxp_and_grp::{
	cursor::{Cursor, NameBudget},
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

/// Nodes are stored parents first: a node's parent has a lower index
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
		if data.len() < dump_size || dump_size / NODE_SIZE < count {
			return Err(DxpGrpError::Model("node tree data is truncated".to_owned()));
		}

		let budget = NameBudget::new(&data);
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
			} else if parent_ofs % NODE_SIZE == 0 && parent_ofs / NODE_SIZE < i {
				// Parents come first, as world matrices are computed in node order; no cycles
				Some(parent_ofs / NODE_SIZE)
			} else {
				return Err(DxpGrpError::Model(format!(
					"node {i} has an invalid parent"
				)));
			};
			nodes.push(GeomNode {
				name: budget.take(Cursor::at(&data, name_ofs)?.cstr()?)?,
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

#[cfg(test)]
mod test {
	use super::*;

	/// Uncompressed tree whose node `i` has the parent pointer `parents[i]`
	fn tree(parents: &[usize]) -> Vec<u8> {
		let names = parents.len() * NODE_SIZE;
		let mut dump = vec![0_u8; names];
		for (i, &parent) in parents.iter().enumerate() {
			let node = &mut dump[i * NODE_SIZE..];
			node[144..152].copy_from_slice(&(parent as u64).to_le_bytes());
			node[152..160].copy_from_slice(&((names + i * 2) as u64).to_le_bytes());
			dump.extend_from_slice(&[b'a' + i as u8, 0]);
		}
		let mut res = (dump.len() as u32).to_le_bytes().to_vec();
		res.extend_from_slice(&(parents.len() as u32).to_le_bytes());
		res.extend_from_slice(&dump);
		res
	}

	#[test]
	fn parents_come_first() {
		let nodes = GeomNodeTree::parse(&tree(&[0, 0, NODE_SIZE]))
			.unwrap()
			.nodes;
		let parents: Vec<_> = nodes.iter().map(|n| n.parent).collect();
		assert_eq!(parents, [None, Some(0), Some(1)]);
		assert_eq!(nodes[2].name, "c");

		// A node that is its own parent, and two nodes that are each other's: walking up to the root
		// from them never ends
		assert!(GeomNodeTree::parse(&tree(&[0, NODE_SIZE])).is_err());
		assert!(GeomNodeTree::parse(&tree(&[0, 2 * NODE_SIZE, NODE_SIZE])).is_err());
	}

	#[test]
	fn names_are_not_copied_over_and_over() {
		// Four nodes named by one 1000-byte string: every node used to copy it, so memory grew with
		// the square of the input size
		let mut res = tree(&[0, 0, 0, 0]);
		let name = res.len() - 8;
		res.extend_from_slice(&[b'a'; 1000]);
		res.push(0);
		for i in 0..4 {
			let at = 8 + i * NODE_SIZE + 152;
			res[at..at + 8].copy_from_slice(&(name as u64).to_le_bytes());
		}
		assert!(GeomNodeTree::parse(&res).is_err());
	}
}
