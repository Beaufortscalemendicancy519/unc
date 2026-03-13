//! Runtime target detection.

use super::{AppleGPUFamily, MetalTarget};

/// Detect Metal target by querying the default MTLDevice.
#[cfg(target_os = "macos")]
pub fn detect_metal_target() -> anyhow::Result<MetalTarget> {
    use metal::Device;

    let device = Device::system_default()
        .ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;

    let name = device.name().to_lowercase();

    let gpu_family = if name.contains("m4") || name.contains("apple10") {
        AppleGPUFamily::Apple10
    } else if name.contains("m3") || name.contains("apple9") {
        AppleGPUFamily::Apple9
    } else if name.contains("m2") || name.contains("apple8") {
        AppleGPUFamily::Apple8
    } else {
        // M1 or unknown Apple Silicon — default to Apple7
        AppleGPUFamily::Apple7
    };

    // Get unified memory size via sysctl
    let unified_memory_gb = get_unified_memory_gb().unwrap_or(8);

    // Rough bandwidth estimates per family (GB/s)
    let memory_bandwidth_gbps = match gpu_family {
        AppleGPUFamily::Apple7 => 68,
        AppleGPUFamily::Apple8 => 100,
        AppleGPUFamily::Apple9 => 150,
        AppleGPUFamily::Apple10 => 120,
    };

    Ok(MetalTarget {
        gpu_family,
        unified_memory_gb,
        memory_bandwidth_gbps,
    })
}

#[cfg(target_os = "macos")]
fn get_unified_memory_gb() -> Option<usize> {
    use std::process::Command;
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let s = String::from_utf8(output.stdout).ok()?;
    let bytes: u64 = s.trim().parse().ok()?;
    Some((bytes / (1024 * 1024 * 1024)) as usize)
}
