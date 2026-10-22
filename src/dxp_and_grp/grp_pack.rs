//! GRP game resource packs, with access to each resource's data
//!
//! Layout (`gamerespackbin` in Dagor's `prog/engine/gameRes/grpData.h`), offsets from the file start:
//!
//! |Offset|Content|
//! |---|---|
//! |0x00|`GRP2` or `GRP3`, u32 description size, u32 dump size, u32 file size minus 16|
//! |0x10|Name table: u32 offset, u32 count, 8 bytes padding. Points at i32 name offsets (-1: none)|
//! |0x20|Resource table: u32 offset, u32 count, padding. 12 bytes per resource: u32 class id, u32 file offset, u16 name id, u16|
//! |0x30|Resource dependency table (not read here)|
//!
//! A resource's data runs from its offset to the next resource's offset, or to the end of the file.

use crate::dxp_and_grp::{
	cursor::{Cursor, NameBudget, check_table},
	error::DxpGrpError,
};

/// Class ids of the resource kinds this crate can read (`dag_stdGameResId.h`)
pub mod class_id {
	/// [`crate::dxp_and_grp::dynmodel::DynModel`]
	pub const DYN_MODEL: u32 = 0xB4B7_D9C4;
	/// [`crate::dxp_and_grp::geom_node_tree::GeomNodeTree`]
	pub const GEOM_NODE_TREE: u32 = 0x56F8_1B6D;
	pub const REND_INST: u32 = 0x77F8_232F;
	pub const CHARACTER: u32 = 0xA6F8_7A9B;
	pub const COLLISION: u32 = 0xACE5_0000;
	pub const FAST_PHYS: u32 = 0x855A_1BE6;
	pub const PHYS_OBJ: u32 = 0xD543_E771;
	pub const EFFECT: u32 = 0x88B7_A117;
	pub const ANIM2_DATA: u32 = 0x40C5_86F9;
	pub const ANIM_GRAPH: u32 = 0x8F2A_701A;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrpResource {
	pub name:     String,
	pub class_id: u32,
	offset:       usize,
	size:         usize,
}

pub struct GrpPack<'a> {
	file:          &'a [u8],
	pub resources: Vec<GrpResource>,
}

impl<'a> GrpPack<'a> {
	pub fn parse(file: &'a [u8]) -> Result<Self, DxpGrpError> {
		let mut r = Cursor::new(file);
		let label = r.bytes(4)?;
		if label != b"GRP2" && label != b"GRP3" {
			return Err(DxpGrpError::InvalidHeader {
				found: String::from_utf8_lossy(label).into_owned(),
			});
		}
		let tab = |at: usize| -> Result<(usize, usize), DxpGrpError> {
			let mut t = Cursor::at(file, at)?;
			Ok((t.u32()? as usize, t.u32()? as usize))
		};
		let (names_ofs, names_count) = tab(0x10)?;
		let (res_ofs, res_count) = tab(0x20)?;
		check_table(file, names_ofs, names_count, 4)?;
		check_table(file, res_ofs, res_count, 12)?;

		let budget = NameBudget::new(file);
		let mut names = Vec::with_capacity(names_count);
		let mut nr = Cursor::at(file, names_ofs)?;
		for _ in 0..names_count {
			let ofs = nr.u32()? as i32;
			names.push(if ofs < 0 {
				String::new()
			} else {
				budget.take(Cursor::at(file, ofs as usize)?.cstr()?)?
			});
		}

		let copies = NameBudget::new(file);
		let mut raw = Vec::with_capacity(res_count);
		let mut rr = Cursor::at(file, res_ofs)?;
		for _ in 0..res_count {
			let class_id = rr.u32()?;
			let offset = rr.u32()? as usize;
			let name_id = rr.u16()? as usize;
			rr.skip(2)?;
			let name = names.get(name_id).cloned().ok_or_else(|| {
				DxpGrpError::Model(format!("resource name id {name_id} out of range"))
			})?;
			let name = copies.take(name)?;
			raw.push((name, class_id, offset));
		}

		let mut resources = Vec::with_capacity(raw.len());
		for (i, (name, class_id, offset)) in raw.iter().enumerate() {
			let end = raw.get(i + 1).map_or(file.len(), |next| next.2);
			if *offset > end || end > file.len() {
				return Err(DxpGrpError::Model(format!(
					"resource {name} has a bad offset"
				)));
			}
			resources.push(GrpResource {
				name:     name.clone(),
				class_id: *class_id,
				offset:   *offset,
				size:     end - offset,
			});
		}
		Ok(Self { file, resources })
	}

	/// Raw data of a resource
	pub fn data(&self, resource: &GrpResource) -> &'a [u8] {
		&self.file[resource.offset..resource.offset + resource.size]
	}

	/// Finds a resource by name and class
	pub fn find(&self, name: &str, class_id: u32) -> Option<&GrpResource> {
		self.resources
			.iter()
			.find(|r| r.name == name && r.class_id == class_id)
	}
}
