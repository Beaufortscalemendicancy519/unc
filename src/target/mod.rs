//! Hardware target definitions.
//!
//! Each target represents a specific GPU family with known capabilities.
//! The compiler selects kernels based on the target.

pub mod detect;

/// Top-level compilation target.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Target {
    Metal(MetalTarget),
}

impl Target {
    pub fn name(&self) -> &str {
        match self {
            Target::Metal(_) => "metal",
        }
    }

    pub fn max_shared_memory(&self) -> usize {
        match self {
            Target::Metal(m) => m.gpu_family.max_threadgroup_memory(),
        }
    }

    pub fn max_threads_per_group(&self) -> u32 {
        match self {
            Target::Metal(m) => m.gpu_family.max_threads_per_threadgroup(),
        }
    }
}

// ---------------------------------------------------------------------------
// Metal targets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MetalTarget {
    pub gpu_family: AppleGPUFamily,
    pub unified_memory_gb: usize,
    pub memory_bandwidth_gbps: u32, // stored as u32 to allow Hash/Eq
}

/// Apple GPU families that affect kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum AppleGPUFamily {
    Apple7,  // M1
    Apple8,  // M2
    Apple9,  // M3
    Apple10, // M4
}

impl AppleGPUFamily {
    pub fn max_threadgroup_memory(&self) -> usize {
        32768 // 32KB on all M-series
    }

    pub fn max_threads_per_threadgroup(&self) -> u32 {
        1024
    }

    pub fn simd_width(&self) -> u32 {
        32
    }

    pub fn supports_nax_attention(&self) -> bool {
        *self >= AppleGPUFamily::Apple9
    }

    /// Preferred GEMM tile size [BM, BN, BK] for f16.
    pub fn preferred_gemm_tile_f16(&self) -> (u32, u32, u32) {
        match self {
            AppleGPUFamily::Apple7 | AppleGPUFamily::Apple8 => (32, 32, 16),
            AppleGPUFamily::Apple9 | AppleGPUFamily::Apple10 => (64, 64, 32),
        }
    }
}
