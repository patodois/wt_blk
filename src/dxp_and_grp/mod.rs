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
