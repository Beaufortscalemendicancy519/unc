pub mod serialize;
pub mod deserialize;

pub use serialize::write_unc;
pub use deserialize::read_unc;

/// Magic bytes at the start of every .unc file.
pub const MAGIC: &[u8; 4] = b"UNC\x01";

/// Current format version.
pub const FORMAT_VERSION: u32 = 1;
