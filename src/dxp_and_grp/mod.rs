use std::fs::File;

use crate::dxp_and_grp::{
	dxp::parse_dxp,
	error::{DxpGrpError, DxpGrpError::Utf8Error},
	grp::parse_grp,
};

mod cursor;
pub mod ddsx;
pub mod dxp;
pub mod dynmodel;
pub mod error;
pub mod geom_node_tree;
pub mod grp;
pub mod grp_pack;
pub mod meshopt;
mod unpack;

pub fn parse_either(file: &[u8]) -> Result<Vec<String>, DxpGrpError> {
	if file.len() < 4 {
		return Err(DxpGrpError::FileTooShort { len: file.len() });
	}

	match <&[u8; 4]>::try_from(&file[0..4]).expect("Infallible") {
		b"GRP2" => parse_grp(file),
		b"DxP2" => parse_dxp(file),
		_ => Err(DxpGrpError::InvalidHeader {
			found: String::from_utf8(file[0..4].to_owned())
				.map_err(|e| Utf8Error(e.utf8_error()))?,
		}),
	}
}

/// This function yields the names from a DXP file, using a relative OS buffer
/// # Safety
/// Do not modify or change the file during this call, as it will result in UB
/// It is recommended to open the file in Read-Write mode, to avoid  invalid external accesses
pub fn parse_buffered(file: &File) -> Result<Vec<String>, DxpGrpError> {
	let file = unsafe { memmap2::Mmap::map(file)? };

	parse_either(&file)
}

#[cfg(test)]
mod test {
	use std::{fs, fs::File};

	use crate::dxp_and_grp::{dxp::parse_dxp, parse_buffered};

	#[test]
	fn fat_hq_tex() {
		let f = fs::read("./samples/dxp/hq_tex_water_garbage_piles.dxp.bin").unwrap();
		let _out = parse_dxp(&f).unwrap();
	}

	#[test]
	fn fat_hq_tex_buffered() {
		let _out = parse_buffered(
			&File::open("./samples/dxp/hq_tex_water_garbage_piles.dxp.bin").unwrap(),
		)
		.unwrap();
	}

	#[test]
	fn grp_buffered() {
		let out = parse_buffered(&File::open("./samples/dxp/bf_109a_1.grp").unwrap()).unwrap();
		assert_eq!(
			[
				"bf_109a_1_cockpit_char",
				"bf_109a_1_cockpit_animtree",
				"bf_109a_1_cockpit_skeleton",
				"bf_109a_1_cockpit",
				"bf_109a_1_cockpit_anim",
				"bf_109a_1_anim",
			]
			.to_vec(),
			out.iter().map(|e| e.as_str()).collect::<Vec<_>>()
		)
	}

	#[test]
	fn dxp_pack_decodes_oodle_textures() {
		use crate::dxp_and_grp::{dxp::DxpPack, parse_dxp};

		let file = std::fs::read("./samples/dxp/hq_tex_water_garbage_piles.dxp.bin").unwrap();
		let pack = DxpPack::parse(&file).unwrap();
		let names: Vec<_> = pack.entries.iter().map(|e| e.name.clone()).collect();
		assert_eq!(names, parse_dxp(&file).unwrap());

		for (i, entry) in pack.entries.iter().enumerate() {
			let tex = pack.texture(i).unwrap();
			assert_eq!(tex.data.len(), entry.header.mem_size as usize);
			let dds = tex.to_dds().unwrap();
			assert_eq!(&dds[..4], b"DDS ");
			assert_eq!(&dds[84..88], &entry.header.d3d_format.to_le_bytes());
			assert_eq!(dds.len(), 128 + tex.data.len());
		}
	}

	fn sample_pack() -> Vec<u8> {
		std::fs::read("./samples/dxp/bf_109a_1.grp").unwrap()
	}

	#[test]
	fn grp_pack_resources_come_from_the_name_table() {
		use crate::dxp_and_grp::grp_pack::GrpPack;

		let file = sample_pack();
		let pack = GrpPack::parse(&file).unwrap();
		// The name table also holds names of resources referenced from other packs
		let table = crate::dxp_and_grp::grp::parse_grp(&file).unwrap();
		assert_eq!(pack.resources.len(), 5);
		assert!(pack.resources.iter().all(|r| table.contains(&r.name)));
	}

	#[test]
	fn dynmodel_sample() {
		use crate::dxp_and_grp::{
			dynmodel::DynModel,
			geom_node_tree::GeomNodeTree,
			grp_pack::{GrpPack, class_id},
		};

		let file = sample_pack();
		let pack = GrpPack::parse(&file).unwrap();
		let model = pack.find("bf_109a_1_cockpit", class_id::DYN_MODEL).unwrap();
		let model = DynModel::parse(pack.data(model)).unwrap();
		let tree = pack
			.find("bf_109a_1_cockpit_skeleton", class_id::GEOM_NODE_TREE)
			.unwrap();
		let tree = GeomNodeTree::parse(pack.data(tree)).unwrap();

		assert_eq!(model.lods.len(), 1);
		let lod = &model.lods[0];
		assert_eq!(lod.rigids.len(), 104);
		let triangles: usize = lod
			.rigids
			.iter()
			.flat_map(|r| &r.elems)
			.map(|e| e.triangles)
			.sum();
		assert_eq!(triangles, 135_665);
		assert!(lod.skins.is_empty() && model.skin_vdata.is_empty());

		// Every part is placed by a skeleton node ("@root" is the model origin)
		for rigid in &lod.rigids {
			assert!(
				rigid.node == "@root" || tree.find(&rigid.node).is_some(),
				"{}",
				rigid.node
			);
		}

		// Decoded vertices are finite, inside a cockpit-sized box, with unit-ish normals
		for e in lod.rigids.iter().flat_map(|r| &r.elems) {
			let vdata = &model.vdata[e.vdata];
			for &i in &vdata.indices[e.first_index..e.first_index + e.triangles * 3] {
				let v = model.vertex(vdata, i as usize + e.base_vertex).unwrap();
				assert!(
					v.pos.iter().all(|x| x.is_finite() && x.abs() < 10.0),
					"{v:?}"
				);
				let len = v.normal.iter().map(|x| x * x).sum::<f32>().sqrt();
				assert!((0.9..1.1).contains(&len), "{v:?}");
			}
		}
	}

	#[test]
	fn geom_node_tree_world_matrices_follow_parents() {
		use crate::dxp_and_grp::{
			geom_node_tree::GeomNodeTree,
			grp_pack::{GrpPack, class_id},
		};

		let file = sample_pack();
		let pack = GrpPack::parse(&file).unwrap();
		let tree = pack
			.find("bf_109a_1_cockpit_skeleton", class_id::GEOM_NODE_TREE)
			.unwrap();
		let tree = GeomNodeTree::parse(pack.data(tree)).unwrap();
		assert_eq!(tree.nodes.len(), 156);
		assert_eq!(tree.nodes[0].parent, None);

		// Column-major 4x4 product
		let mul = |a: &[f32; 16], b: &[f32; 16]| {
			let mut m = [0_f32; 16];
			for c in 0..4 {
				for r in 0..4 {
					m[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
				}
			}
			m
		};
		for (i, node) in tree.nodes.iter().enumerate().skip(1) {
			let parent = node.parent.unwrap();
			assert!(parent < i, "nodes are stored parents first");
			let expected = mul(&tree.nodes[parent].wtm, &node.tm);
			for (a, b) in expected.iter().zip(&node.wtm) {
				assert!(
					(a - b).abs() < 1e-3,
					"node {} wtm {:?} vs {:?}",
					node.name,
					node.wtm,
					expected
				);
			}
		}
	}

	#[test]
	fn packed_block_sizes_are_not_trusted_for_allocation() {
		use crate::dxp_and_grp::geom_node_tree::GeomNodeTree;

		// A compressed node tree whose Oodle block claims 4 GiB, which used to be allocated up front
		let mut res = Vec::new();
		for v in [0, 0x8000_0001, 10, u32::MAX] {
			res.extend_from_slice(&u32::to_le_bytes(v));
		}
		res.extend_from_slice(&[0x8C, 0x06, 0, 0, 0, 0]);
		assert!(GeomNodeTree::parse(&res).is_err());
	}

	fn put_u32(buf: &mut [u8], at: usize, v: u32) {
		buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
	}

	#[test]
	fn huge_table_counts_are_rejected_before_allocating() {
		use crate::dxp_and_grp::{
			dxp::DxpPack,
			dynmodel::DynModel,
			grp_pack::{GrpPack, class_id},
		};

		// Counts were used for Vec::with_capacity (and vec![String::new(); n], which writes every
		// element) before checking that their tables fit in the file: 4G entries meant 100+ GiB
		let mut grp = vec![0_u8; 0x40];
		grp[..4].copy_from_slice(b"GRP2");
		put_u32(&mut grp, 0x10, 0x30);
		put_u32(&mut grp, 0x14, u32::MAX);
		assert!(GrpPack::parse(&grp).is_err());
		put_u32(&mut grp, 0x14, 0);
		put_u32(&mut grp, 0x20, 0x30);
		put_u32(&mut grp, 0x24, u32::MAX);
		assert!(GrpPack::parse(&grp).is_err());

		let mut dxp = vec![0_u8; 0x40];
		dxp[..4].copy_from_slice(b"DxP2");
		put_u32(&mut dxp, 0x4, 2);
		put_u32(&mut dxp, 0xC, 0x30);
		for table in [0x10, 0x20, 0x30] {
			put_u32(&mut dxp, table + 4, u32::MAX);
		}
		assert!(DxpPack::parse(&dxp).is_err());

		// Vertex data with 4G vertex buffers, then one buffer with 4G channels, zstd packed
		let model = |vdata: &[u8]| {
			let block = zstd::bulk::compress(vdata, 0).unwrap();
			let mut res = Vec::new();
			for v in [0, u32::MAX, u32::MAX, 1, 0xC000_0000 | 24] {
				res.extend_from_slice(&v.to_le_bytes());
			}
			res.extend_from_slice(&(block.len() as u32 | 1 << 30).to_le_bytes());
			res.extend_from_slice(&block);
			res
		};
		let mut vdata = vec![0_u8; 24];
		put_u32(&mut vdata, 16, 24);
		put_u32(&mut vdata, 20, u32::MAX);
		assert!(DynModel::parse(&model(&vdata)).is_err());
		put_u32(&mut vdata, 20, 1);
		vdata.extend_from_slice(&[0; 32]);
		put_u32(&mut vdata, 24 + 16, 56);
		put_u32(&mut vdata, 24 + 20, u32::MAX);
		assert!(DynModel::parse(&model(&vdata)).is_err());

		// The sample model with each of its other counts raised
		let file = sample_pack();
		let pack = GrpPack::parse(&file).unwrap();
		let data = pack.data(pack.find("bf_109a_1_cockpit", class_id::DYN_MODEL).unwrap());
		let u32_at = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
		let lods = 24 + (u32_at(20) & 0x3FFF_FFFF);
		let names = lods + u32_at(0);
		let scene = names + 4 + u32_at(names);
		let mesh = scene + u32_at(lods + u32_at(lods));
		for (at, what) in [
			(lods + 4, "LODs"),
			(names + 4 + 4, "node names"),
			(scene + 4, "rigids"),
			(scene + 20, "skins"),
			(mesh + 4, "mesh elements"),
		] {
			let mut patched = data.to_vec();
			put_u32(&mut patched, at, u32::MAX);
			if what == "node names" {
				put_u32(&mut patched, at + 16, u32::MAX);
			}
			assert!(DynModel::parse(&patched).is_err(), "{what}");
		}
	}

	#[test]
	fn names_are_not_copied_over_and_over() {
		use crate::dxp_and_grp::{dxp::DxpPack, dynmodel::DynModel, grp_pack::GrpPack};

		// Tables of names whose entries all point at one long string: every entry used to copy it,
		// so memory grew with the square of the input size
		let long = [b'a'; 1000];

		let mut grp = vec![0_u8; 0x40];
		grp[..4].copy_from_slice(b"GRP2");
		put_u32(&mut grp, 0x10, 0x40);
		put_u32(&mut grp, 0x14, 8);
		for _ in 0..8 {
			grp.extend_from_slice(&0x60_u32.to_le_bytes());
		}
		grp.extend_from_slice(&long);
		grp.push(0);
		assert!(GrpPack::parse(&grp).is_err());
		// One name, given to many resources
		put_u32(&mut grp, 0x14, 1);
		let res_ofs = grp.len() as u32;
		put_u32(&mut grp, 0x20, res_ofs);
		put_u32(&mut grp, 0x24, 8);
		let end = grp.len() as u32 + 8 * 12;
		for _ in 0..8 {
			for v in [0, end, 0] {
				grp.extend_from_slice(&v.to_le_bytes());
			}
		}
		assert!(GrpPack::parse(&grp).is_err());
		put_u32(&mut grp, 0x24, 1);
		assert_eq!(GrpPack::parse(&grp).unwrap().resources.len(), 1);

		// Four textures with the same 1000-byte name
		let mut dump = vec![0_u8; 0x30];
		for (table, ofs) in [(0x00, 0x30), (0x10, 0x50), (0x20, 0xD0)] {
			put_u32(&mut dump, table, ofs);
			put_u32(&mut dump, table + 4, 4);
		}
		for _ in 0..4 {
			dump.extend_from_slice(&0x130_u64.to_le_bytes());
		}
		for _ in 0..4 {
			dump.extend_from_slice(b"DDSx");
			dump.extend_from_slice(&[0; 28]);
		}
		dump.extend_from_slice(&[0; 4 * 24]);
		dump.extend_from_slice(&long);
		dump.push(0);
		let mut dxp = b"DxP2".to_vec();
		for v in [2, 4, dump.len() as u32] {
			dxp.extend_from_slice(&u32::to_le_bytes(v));
		}
		dxp.extend_from_slice(&dump);
		assert!(DxpPack::parse(&dxp).is_err());

		// A model with one long node name and 40 rigid parts placed by that node
		let model = |rigids: usize| {
			let mut res = Vec::new();
			let put = |res: &mut Vec<u8>, values: &[u32]| {
				for v in values {
					res.extend_from_slice(&v.to_le_bytes());
				}
			};
			// No vertex buffers, stored
			put(&mut res, &[0x58, u32::MAX, u32::MAX, 0, 24]);
			put(&mut res, &[0, 0, 0, 0, 24, 0]);
			// LOD table at 0x48 in the LODs dump, then the one LOD's scene size
			let scene_size = 24 + 32 * rigids as u32;
			let mut lods = [0_u32; 0x58 / 4];
			lods[..2].copy_from_slice(&[0x48, 1]);
			lods[0x48 / 4] = scene_size;
			put(&mut res, &lods);
			// Node names: one name at 34, node id 0
			put(&mut res, &[34 + 1001, 24, 1, 0, 0, 32, 1, 34, 0, 0]);
			res.truncate(res.len() - 2);
			res.extend_from_slice(&long);
			res.push(0);
			// Scene: rigids at 24, each with an 8-byte empty mesh
			put(&mut res, &[24, rigids as u32, 0, 0, 0, 0]);
			for _ in 0..rigids {
				put(&mut res, &[8, 0, 0, 0, 0, 0, 0, 0]);
			}
			for _ in 0..rigids {
				put(&mut res, &[0, 0]);
			}
			res
		};
		let parsed = DynModel::parse(&model(1)).unwrap();
		assert_eq!(parsed.lods[0].rigids[0].node.len(), 1000);
		assert!(DynModel::parse(&model(40)).is_err());
	}

	#[test]
	fn sizes_do_not_overflow_on_32_bit_targets() {
		use crate::dxp_and_grp::{
			dxp::DxpPack,
			dynmodel::DynModel,
			geom_node_tree::GeomNodeTree,
			grp_pack::{GrpPack, class_id},
		};

		// Where usize is 32 bits (wasm32), these sizes overflowed: a panic in debug builds, and in
		// release builds checks that passed on the wrapped values

		// 0x7FFFFFFF nodes of 160 bytes
		let mut tree = vec![0_u8; 8 + 0x100];
		put_u32(&mut tree, 0, 0x100);
		put_u32(&mut tree, 4, 0x7FFF_FFFF);
		assert!(GeomNodeTree::parse(&tree).is_err());

		// A table dump of 0xFFFFFFFF bytes after the 16-byte header
		let mut dxp = vec![0_u8; 0x40];
		dxp[..4].copy_from_slice(b"DxP2");
		put_u32(&mut dxp, 0x4, 2);
		put_u32(&mut dxp, 0xC, u32::MAX);
		assert!(DxpPack::parse(&dxp).is_err());

		// Mesh elements of 0x55555556 triangles (0x100000002 indices) and with a base vertex of
		// 0xFFFFFFFF
		let file = sample_pack();
		let pack = GrpPack::parse(&file).unwrap();
		let data = pack.data(pack.find("bf_109a_1_cockpit", class_id::DYN_MODEL).unwrap());
		let u32_at = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
		let lods = 24 + (u32_at(20) & 0x3FFF_FFFF);
		let names = lods + u32_at(0);
		let scene = names + 4 + u32_at(names);
		// The first rigid's mesh lists no elements in its main table, the second's lists one
		let mesh = scene + u32_at(lods + u32_at(lods));
		let mesh = mesh + u32_at(scene + u32_at(scene));
		assert_eq!(u32_at(mesh + 4), 1);
		let elem = mesh + u32_at(mesh);
		for (field, value) in [(40, 0x5555_5556), (44, u32::MAX)] {
			let mut patched = data.to_vec();
			put_u32(&mut patched, elem + 36, 0);
			put_u32(&mut patched, elem + field, value);
			assert!(DynModel::parse(&patched).is_err());
		}
	}

	#[test]
	fn truncated_models_fail_without_panicking() {
		use crate::dxp_and_grp::{
			dynmodel::DynModel,
			grp_pack::{GrpPack, class_id},
		};

		let file = sample_pack();
		let pack = GrpPack::parse(&file).unwrap();
		let data = pack.data(pack.find("bf_109a_1_cockpit", class_id::DYN_MODEL).unwrap());
		// Resources end with alignment padding, so only cut well inside the data
		for cut in [
			0,
			3,
			20,
			64,
			data.len() / 3,
			data.len() / 2,
			data.len() * 3 / 4,
		] {
			assert!(DynModel::parse(&data[..cut]).is_err(), "cut at {cut}");
		}
		assert!(GrpPack::parse(&file[..40]).is_err());
	}
}
