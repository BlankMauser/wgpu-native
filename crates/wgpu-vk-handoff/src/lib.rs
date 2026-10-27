//! The owner side of wgpu-native's Vulkan interop: builds a
//! `WGPUVulkanHandoffExt` (ffi/wgpu_vk_interop.h) from the owner's own
//! `wgpu::Device` and `wgpu::Texture` (wgpu 30, Vulkan backend), so the
//! adopting side never types a handle, size, format, usage or extension list
//! by hand.
//!
//! The owner's half of the protocol:
//! - Write the texture once (a clear pass is enough) before the first handoff,
//!   then bring it to the handoff state with
//!   [`Handoff::prepare`] (`CommandEncoder::transition_resources`) and submit.
//! - After every use of the texture, transition it back to that state.
//! - Take the [`QueueLock`] around every submission to the shared queue, and
//!   around dropping the device/queue. Never call into the adopting side while
//!   holding it (it takes the same lock: deadlock).
//! - The lock's `context` must stay valid until the adopting side's teardown
//!   returns (it copies the lock and calls it until then); the `Handoff`
//!   itself is read only during adoption and import.
//! - `VulkanHandoff` is append-only (see the C header): fields are never
//!   removed, reordered or reinterpreted; `struct_size` is the version.
//! - Keep the `Handoff` alive, and destroy the device or texture only after
//!   the adopting side released what it adopted/imported (LunarFenrir:
//!   `Gpu::close`, `Gpu::release_shared` or a `Reloader::resize`).

use ash::vk::Handle;
use std::ffi::{c_char, c_void};
use wgpu::hal::api::Vulkan;

/// `WGPUVulkanDeviceHandlesExt`.
#[repr(C)]
pub struct VulkanDeviceHandles {
    pub get_instance_proc_addr: *const c_void,
    pub instance: u64,
    pub instance_api_version: u32,
    pub instance_extension_count: usize,
    pub instance_extensions: *const *const c_char,
    pub physical_device: u64,
    pub device: u64,
    pub device_extension_count: usize,
    pub device_extensions: *const *const c_char,
    pub queue_family_index: u32,
    pub queue_index: u32,
}

/// `WGPUVulkanHandoffLockExt`.
pub type LockFn = unsafe extern "C" fn(context: *mut c_void);

/// `WGPUVulkanHandoffExt`.
#[repr(C)]
pub struct VulkanHandoff {
    /// `size_of::<VulkanHandoff>()`, so readers can check the layout.
    pub struct_size: u32,
    pub device: VulkanDeviceHandles,
    /// `VkImage`.
    pub image: u64,
    pub width: u32,
    pub height: u32,
    /// `WGPUTextureFormat`.
    pub format: u32,
    /// `WGPUTextureUsage`: the texture's usage.
    pub usage: u64,
    /// `WGPUTextureUsage`: the agreed handoff state, exactly one flag.
    pub state: u64,
    pub lock: Option<LockFn>,
    pub unlock: Option<LockFn>,
    pub lock_context: *mut c_void,
    /// Appended: the device's features as `WGPUFeatureName` values.
    pub feature_count: usize,
    pub features: *const u32,
}

/// The owner's lock for the shared queue, exported through the handoff.
#[derive(Clone, Copy)]
pub struct QueueLock {
    pub lock: LockFn,
    pub unlock: LockFn,
    pub context: *mut c_void,
}

/// A handoff for one texture. Its pointer stays valid while it lives.
pub struct Handoff {
    raw: Box<VulkanHandoff>,
    _instance_extensions: Vec<*const c_char>,
    _device_extensions: Vec<*const c_char>,
    _features: Vec<u32>,
    state: wgpu::TextureUses,
}

/// The `WGPUFeatureName` values of the features a handoff can carry: the
/// WebGPU features wgpu-native maps, and the timestamp features. Others are
/// left out (the adopter then sees a subset, which is safe).
pub fn webgpu_features(features: wgpu::Features) -> Vec<u32> {
    use wgpu::Features as F;
    let table: &[(F, u32)] = &[
        (F::DEPTH_CLIP_CONTROL, 0x02),
        (F::DEPTH32FLOAT_STENCIL8, 0x03),
        (F::TEXTURE_COMPRESSION_BC, 0x04),
        (F::TEXTURE_COMPRESSION_BC_SLICED_3D, 0x05),
        (F::TEXTURE_COMPRESSION_ETC2, 0x06),
        (F::TEXTURE_COMPRESSION_ASTC, 0x07),
        (F::TEXTURE_COMPRESSION_ASTC_SLICED_3D, 0x08),
        (F::TIMESTAMP_QUERY, 0x09),
        (F::INDIRECT_FIRST_INSTANCE, 0x0A),
        (F::SHADER_F16, 0x0B),
        (F::RG11B10UFLOAT_RENDERABLE, 0x0C),
        (F::BGRA8UNORM_STORAGE, 0x0D),
        (F::FLOAT32_FILTERABLE, 0x0E),
        (F::FLOAT32_BLENDABLE, 0x0F),
        (F::CLIP_DISTANCES, 0x10),
        (F::DUAL_SOURCE_BLENDING, 0x11),
        // wgpu-native (wgpu.h WGPUNativeFeature).
        (F::TIMESTAMP_QUERY_INSIDE_ENCODERS, 0x0003_0024),
        (F::TIMESTAMP_QUERY_INSIDE_PASSES, 0x0003_0025),
    ];
    table.iter().filter(|(f, _)| features.contains(*f)).map(|(_, v)| *v).collect()
}

/// The `WGPUTextureFormat` value of a wgpu format (the formats a handoff
/// supports: colour and depth formats without compression).
pub fn webgpu_format(format: wgpu::TextureFormat) -> Option<u32> {
    use wgpu::TextureFormat as F;
    Some(match format {
        F::R8Unorm => 0x01,
        F::R8Snorm => 0x02,
        F::R8Uint => 0x03,
        F::R8Sint => 0x04,
        F::R16Uint => 0x07,
        F::R16Sint => 0x08,
        F::R16Float => 0x09,
        F::Rg8Unorm => 0x0A,
        F::Rg8Snorm => 0x0B,
        F::Rg8Uint => 0x0C,
        F::Rg8Sint => 0x0D,
        F::R32Float => 0x0E,
        F::R32Uint => 0x0F,
        F::R32Sint => 0x10,
        F::Rg16Uint => 0x13,
        F::Rg16Sint => 0x14,
        F::Rg16Float => 0x15,
        F::Rgba8Unorm => 0x16,
        F::Rgba8UnormSrgb => 0x17,
        F::Rgba8Snorm => 0x18,
        F::Rgba8Uint => 0x19,
        F::Rgba8Sint => 0x1A,
        F::Bgra8Unorm => 0x1B,
        F::Bgra8UnormSrgb => 0x1C,
        F::Rgb10a2Uint => 0x1D,
        F::Rgb10a2Unorm => 0x1E,
        F::Rg11b10Ufloat => 0x1F,
        F::Rgb9e5Ufloat => 0x20,
        F::Rg32Float => 0x21,
        F::Rg32Uint => 0x22,
        F::Rg32Sint => 0x23,
        F::Rgba16Uint => 0x26,
        F::Rgba16Sint => 0x27,
        F::Rgba16Float => 0x28,
        F::Rgba32Float => 0x29,
        F::Rgba32Uint => 0x2A,
        F::Rgba32Sint => 0x2B,
        F::Stencil8 => 0x2C,
        F::Depth16Unorm => 0x2D,
        F::Depth24Plus => 0x2E,
        F::Depth24PlusStencil8 => 0x2F,
        F::Depth32Float => 0x30,
        F::Depth32FloatStencil8 => 0x31,
        _ => return None,
    })
}

/// The one `WGPUTextureUsage` flag a single tracker state corresponds to.
pub fn webgpu_state(state: wgpu::TextureUses) -> Option<u64> {
    use wgpu::TextureUses as U;
    Some(match state {
        s if s == U::UNINITIALIZED => 0,
        s if s == U::COPY_SRC => wgpu::TextureUsages::COPY_SRC.bits() as u64,
        s if s == U::COPY_DST => wgpu::TextureUsages::COPY_DST.bits() as u64,
        s if s == U::RESOURCE => wgpu::TextureUsages::TEXTURE_BINDING.bits() as u64,
        s if s == U::STORAGE_READ_WRITE => wgpu::TextureUsages::STORAGE_BINDING.bits() as u64,
        s if s == U::COLOR_TARGET || s == U::DEPTH_STENCIL_WRITE => {
            wgpu::TextureUsages::RENDER_ATTACHMENT.bits() as u64
        }
        _ => return None,
    })
}

impl Handoff {
    /// Describes `texture` on `device` (2D, 1 mip, 1 sample, Vulkan) in
    /// the handoff `state`: `RESOURCE`, `COLOR_TARGET` (colour formats) or
    /// `DEPTH_STENCIL_WRITE` (depth formats), `COPY_SRC`, `COPY_DST`,
    /// `STORAGE_READ_WRITE`, or `UNINITIALIZED` (contents do not matter).
    pub fn new(
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        state: wgpu::TextureUses,
        lock: Option<QueueLock>,
    ) -> Result<Handoff, String> {
        let format = webgpu_format(texture.format())
            .ok_or_else(|| format!("{:?} is not a format a handoff supports", texture.format()))?;
        let depth = texture.format().is_depth_stencil_format();
        if (state == wgpu::TextureUses::COLOR_TARGET && depth)
            || (state == wgpu::TextureUses::DEPTH_STENCIL_WRITE && !depth)
        {
            return Err(format!("state {state:?} does not fit a {:?} texture", texture.format()));
        }
        let state_bits = webgpu_state(state).ok_or_else(|| format!("{state:?} is not a single handoff state"))?;
        let usage = texture.usage().bits() as u64;
        if state_bits != 0 && usage & state_bits == 0 {
            return Err(format!("state {state:?} is not part of the texture's usage {:?}", texture.usage()));
        }
        if texture.dimension() != wgpu::TextureDimension::D2
            || texture.depth_or_array_layers() != 1
            || texture.mip_level_count() != 1
            || texture.sample_count() != 1
        {
            return Err("only 2D textures with one layer, mip level and sample".into());
        }
        let image = unsafe { texture.as_hal::<Vulkan>() }
            .map(|t| unsafe { t.raw_handle() }.as_raw())
            .ok_or("not a Vulkan texture")?;
        let hal = unsafe { device.as_hal::<Vulkan>() }.ok_or("not a Vulkan device")?;
        let instance = hal.shared_instance();
        let instance_extensions: Vec<*const c_char> = instance.extensions().iter().map(|e| e.as_ptr()).collect();
        let device_extensions: Vec<*const c_char> =
            hal.enabled_device_extensions().iter().map(|e| e.as_ptr()).collect();
        let features = webgpu_features(device.features());
        let raw = Box::new(VulkanHandoff {
            struct_size: std::mem::size_of::<VulkanHandoff>() as u32,
            device: VulkanDeviceHandles {
                get_instance_proc_addr: instance.entry().static_fn().get_instance_proc_addr as *const c_void,
                instance: instance.raw_instance().handle().as_raw(),
                instance_api_version: instance.instance_api_version(),
                instance_extension_count: instance_extensions.len(),
                instance_extensions: instance_extensions.as_ptr(),
                physical_device: hal.raw_physical_device().as_raw(),
                device: hal.raw_device().handle().as_raw(),
                device_extension_count: device_extensions.len(),
                device_extensions: device_extensions.as_ptr(),
                queue_family_index: hal.queue_family_index(),
                queue_index: hal.queue_index(),
            },
            image,
            width: texture.width(),
            height: texture.height(),
            format,
            usage,
            state: state_bits,
            lock: lock.map(|l| l.lock),
            unlock: lock.map(|l| l.unlock),
            lock_context: lock.map_or(std::ptr::null_mut(), |l| l.context),
            feature_count: features.len(),
            features: features.as_ptr(),
        });
        drop(hal);
        Ok(Handoff {
            raw,
            _instance_extensions: instance_extensions,
            _device_extensions: device_extensions,
            _features: features,
            state,
        })
    }

    /// The `const WGPUVulkanHandoffExt *` to hand over.
    pub fn as_ptr(&self) -> *const VulkanHandoff {
        &*self.raw
    }

    /// Records the transition of `texture` to the handoff state (the owner's
    /// tracker then agrees with the importer's initial state).
    pub fn prepare(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        encoder.transition_resources(
            std::iter::empty(),
            std::iter::once(wgpu::TextureTransition { texture, selector: None, state: self.state }),
        );
    }
}
