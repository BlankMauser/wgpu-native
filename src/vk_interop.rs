//! Vulkan interop extension (BlankMauser fork, spike).
//!
//! Lets wgpu-native drive a `VkDevice` that another wgpu (for example the Rust
//! `wgpu` 30 crate) created, and import that wgpu's `VkImage`s as textures.
//! Each side keeps its own wgpu-core device (own resource tracker), on one
//! `VkDevice` and one `VkQueue`. Correctness rests on a handoff protocol:
//!
//! - Submissions from both sides are externally serialised (one thread, or a
//!   shared lock): `VkQueue` requires external synchronisation.
//! - At every handoff the shared texture is in one agreed state on both
//!   trackers. Import with that state (`initialState`), and before handing
//!   back call `wgpuCommandEncoderTransitionTextureExt` to that state; the Rust
//!   side uses `CommandEncoder::transition_resources` the same way.
//! - The importer must not outlive the exporter's `VkDevice` / `VkImage`.
//!
//! Neither object is destroyed by wgpu-native (drop callbacks are no-ops).

use crate::conv::map_device_descriptor;
use crate::native;
use crate::{
    handle_error, Context, ErrorSinkRaw, QueueId, TextureData, WGPUCommandEncoderImpl,
    WGPUDeviceImpl, WGPUTextureImpl, DEFAULT_DEVICE_LOST_HANDLER,
};
use ash::vk::{self, Handle};
use parking_lot::Mutex;
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

#[repr(C)]
pub struct WGPUVulkanDeviceHandlesExt {
    /// `PFN_vkGetInstanceProcAddr` the exporter loaded.
    pub get_instance_proc_addr: *const c_void,
    /// `VkInstance`.
    pub instance: u64,
    pub instance_api_version: u32,
    pub instance_extension_count: usize,
    pub instance_extensions: *const *const c_char,
    /// `VkPhysicalDevice`.
    pub physical_device: u64,
    /// `VkDevice`.
    pub device: u64,
    pub device_extension_count: usize,
    pub device_extensions: *const *const c_char,
    pub queue_family_index: u32,
    pub queue_index: u32,
}

unsafe fn leak_names(ptr: *const *const c_char, count: usize) -> Vec<&'static CStr> {
    if count == 0 {
        return Vec::new();
    }
    std::slice::from_raw_parts(ptr, count)
        .iter()
        .map(|&p| &*Box::leak(CStr::from_ptr(p).to_owned().into_boxed_c_str()))
        .collect()
}

/// One wgpu `TextureUsage` flag to the tracker state wgpu-core keeps.
fn map_state(usage: native::WGPUTextureUsage) -> wgt::TextureUses {
    match usage {
        native::WGPUTextureUsage_None => wgt::TextureUses::UNINITIALIZED,
        native::WGPUTextureUsage_CopySrc => wgt::TextureUses::COPY_SRC,
        native::WGPUTextureUsage_CopyDst => wgt::TextureUses::COPY_DST,
        native::WGPUTextureUsage_TextureBinding => wgt::TextureUses::RESOURCE,
        native::WGPUTextureUsage_StorageBinding => wgt::TextureUses::STORAGE_READ_WRITE,
        native::WGPUTextureUsage_RenderAttachment => wgt::TextureUses::COLOR_TARGET,
        _ => panic!("handoff state must be exactly one WGPUTextureUsage flag"),
    }
}

/// Wrap an existing Vulkan device. `descriptor` lists the features and limits
/// the `VkDevice` was created with (NULL: none, WebGPU default limits).
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateDeviceFromVulkanExt(
    handles: Option<&WGPUVulkanDeviceHandlesExt>,
    descriptor: Option<&native::WGPUDeviceDescriptor>,
) -> native::WGPUDevice {
    let h = handles.expect("invalid handles");
    let static_fn = ash::StaticFn {
        get_instance_proc_addr: std::mem::transmute::<*const c_void, vk::PFN_vkGetInstanceProcAddr>(
            h.get_instance_proc_addr,
        ),
    };
    let entry = ash::Entry::from_static_fn(static_fn);
    let raw_instance = ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(h.instance));
    let hal_instance = match hal::vulkan::Instance::from_raw(
        entry,
        raw_instance.clone(),
        h.instance_api_version,
        0,
        None,
        leak_names(h.instance_extensions, h.instance_extension_count),
        wgt::InstanceFlags::empty(),
        wgt::MemoryBudgetThresholds::default(),
        false,
        Some(Box::new(|| {})),
    ) {
        Ok(i) => i,
        Err(e) => {
            log::error!("wgpuCreateDeviceFromVulkanExt: {e}");
            return std::ptr::null();
        }
    };
    let Some(exposed) =
        hal_instance.expose_adapter(vk::PhysicalDevice::from_raw(h.physical_device))
    else {
        log::error!("wgpuCreateDeviceFromVulkanExt: physical device not usable");
        return std::ptr::null();
    };
    let (desc, _) = match descriptor {
        Some(d) => map_device_descriptor(d, wgt::Limits::default(), None),
        None => (wgt::DeviceDescriptor::default(), None),
    };
    let raw_device = ash::Device::load(raw_instance.fp_v1_0(), vk::Device::from_raw(h.device));
    let open = match exposed.adapter.device_from_raw(
        raw_device,
        Some(Box::new(|| {})),
        &leak_names(h.device_extensions, h.device_extension_count),
        desc.required_features,
        &desc.required_limits,
        &desc.memory_hints,
        h.queue_family_index,
        h.queue_index,
    ) {
        Ok(o) => o,
        Err(e) => {
            log::error!("wgpuCreateDeviceFromVulkanExt: {e}");
            return std::ptr::null();
        }
    };
    let context = Arc::new(Context::from_hal_instance::<hal::api::Vulkan>(
        "wgpu-native (shared VkDevice)",
        hal_instance,
    ));
    let adapter_id = context.create_adapter_from_hal(exposed.into(), None);
    let (device_id, queue_id) =
        match context.create_device_from_hal(adapter_id, open.into(), &desc, None, None) {
            Ok(ids) => ids,
            Err(e) => {
                log::error!("wgpuCreateDeviceFromVulkanExt: {e}");
                return std::ptr::null();
            }
        };
    context.adapter_drop(adapter_id);
    Arc::into_raw(Arc::new_cyclic(|device| WGPUDeviceImpl {
        context: context.clone(),
        id: device_id,
        queue: Arc::new(QueueId {
            context: context.clone(),
            id: queue_id,
        }),
        error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new(
            DEFAULT_DEVICE_LOST_HANDLER,
            device.clone(),
        ))),
    }))
}

/// Import a `VkImage` owned by the exporter. `initialState` is the single
/// `WGPUTextureUsage` flag the image is in now (the agreed handoff state;
/// `None` means undefined contents).
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceImportVulkanImageExt(
    device: native::WGPUDevice,
    vk_image: u64,
    descriptor: Option<&native::WGPUTextureDescriptor>,
    initial_state: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let device = device.as_ref().expect("invalid device");
    let (device_id, context, error_sink) = (device.id, &device.context, &device.error_sink);
    let descriptor = descriptor.expect("invalid descriptor");
    let desc = wgt::TextureDescriptor {
        label: crate::utils::string_view_into_label(descriptor.label),
        size: crate::conv::map_extent3d(&descriptor.size),
        mip_level_count: descriptor.mipLevelCount,
        sample_count: descriptor.sampleCount,
        dimension: crate::conv::map_texture_dimension(descriptor.dimension)
            .unwrap_or(wgt::TextureDimension::D2),
        format: crate::conv::map_texture_format(descriptor.format)
            .expect("invalid texture format"),
        usage: crate::conv::from_u64_bits(descriptor.usage).expect("invalid texture usage"),
        view_formats: vec![],
    };
    let hal_texture = {
        let hal_device = context
            .device_as_hal::<hal::api::Vulkan>(device_id)
            .expect("not a Vulkan device");
        hal_device.texture_from_raw(
            vk::Image::from_raw(vk_image),
            &hal::TextureDescriptor {
                label: None,
                size: desc.size,
                mip_level_count: desc.mip_level_count,
                sample_count: desc.sample_count,
                dimension: desc.dimension,
                format: desc.format,
                usage: wgt::TextureUses::empty(),
                memory_flags: hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            Some(Box::new(|| {})),
            hal::vulkan::TextureMemory::External,
        )
    };
    let (texture_id, error) = context.create_texture_from_hal(
        Box::new(hal_texture),
        device_id,
        &desc,
        map_state(initial_state),
        None,
    );
    if let Some(cause) = error {
        handle_error(error_sink, cause, desc.label, "wgpuDeviceImportVulkanImageExt");
    }
    Arc::into_raw(Arc::new(WGPUTextureImpl {
        context: context.clone(),
        id: texture_id,
        error_sink: error_sink.clone(),
        surface_id: None,
        has_surface_presented: Arc::default(),
        data: TextureData {
            usage: descriptor.usage,
            dimension: descriptor.dimension,
            size: descriptor.size,
            format: descriptor.format,
            mip_level_count: descriptor.mipLevelCount,
            sample_count: descriptor.sampleCount,
        },
    }))
}

/// Record a transition of the whole texture to one `WGPUTextureUsage` state
/// (wgpu's `CommandEncoder::transition_resources`), for handing it back.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderTransitionTextureExt(
    command_encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    state: native::WGPUTextureUsage,
) {
    let encoder: &WGPUCommandEncoderImpl = command_encoder.as_ref().expect("invalid encoder");
    let texture = texture.as_ref().expect("invalid texture");
    let transition = wgt::TextureTransition {
        texture: texture.id,
        selector: None,
        state: map_state(state),
    };
    if let Err(cause) = encoder.context.command_encoder_transition_resources(
        encoder.id,
        std::iter::empty(),
        std::iter::once(transition),
    ) {
        handle_error(
            &encoder.error_sink,
            cause,
            None,
            "wgpuCommandEncoderTransitionTextureExt",
        );
    }
}

