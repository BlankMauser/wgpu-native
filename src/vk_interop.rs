//! Vulkan interop extension (BlankMauser fork).
//!
//! Lets wgpu-native drive a `VkDevice` that another wgpu (for example the Rust
//! `wgpu` 30 crate) created, and import that wgpu's `VkImage`s as textures.
//! Each side keeps its own wgpu-core device (own resource tracker), on one
//! `VkDevice` and one `VkQueue`. Correctness rests on a handoff protocol:
//!
//! - Submissions from both sides are externally serialised (one thread, or a
//!   shared lock): `VkQueue` requires external synchronisation. This includes
//!   releasing the adopted queue/device, which waits for the queue to be idle.
//! - At every handoff the shared texture is in one agreed state on both
//!   trackers. Import with that state (`initialState`), and before handing
//!   back call `wgpuCommandEncoderTransitionTextureExt` to that state; the Rust
//!   side uses `CommandEncoder::transition_resources` the same way.
//! - The exporter writes a texture once before its first handoff (its own
//!   lazy initialisation would otherwise clear it).
//! - Everything adopted or imported here is released before the exporter
//!   destroys its `VkDevice` / `VkImage`: the adopted device owns Vulkan
//!   objects (fence, command and descriptor pools, memory) on that `VkDevice`.
//! - The extension lists, queue indices and image descriptors must be exactly
//!   the exporter's. Vulkan cannot report an image's create info and hal
//!   trusts the extension list, so these are not verifiable here; take them
//!   from the exporter's own objects, never type them by hand.
//!
//! Neither object is destroyed by wgpu-native (drop callbacks are no-ops).
//!
//! Errors: no entry point unwinds into C. Bad input returns NULL (or 0 for
//! the transition) and leaves a message for `wgpuVulkanInteropLastErrorExt`
//! (per thread).

use crate::conv::map_device_descriptor;
use crate::native;
use crate::utils::texture_format_has_depth;
use crate::{
    handle_error, Context, ErrorSinkRaw, QueueId, TextureData, WGPUAdapterImpl,
    WGPUCommandEncoderImpl, WGPUDeviceImpl, WGPUInstanceImpl, WGPUTextureImpl,
    DEFAULT_DEVICE_LOST_HANDLER,
};
use ash::vk::{self, Handle};
use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
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

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: String) {
    log::error!("vulkan interop: {message}");
    LAST_ERROR.with(|e| *e.borrow_mut() = message);
}

/// Copies this thread's last interop error (UTF-8, not NUL-terminated) into
/// `out` when `capacity` is large enough; returns its length (0: none).
#[no_mangle]
pub unsafe extern "C" fn wgpuVulkanInteropLastErrorExt(out: *mut c_char, capacity: usize) -> usize {
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        if !out.is_null() && capacity >= e.len() {
            std::ptr::copy_nonoverlapping(e.as_ptr(), out as *mut u8, e.len());
        }
        e.len()
    })
}

/// Runs an entry point: an `Err` or a panic becomes `fallback` plus a message;
/// nothing unwinds across the C boundary.
fn guarded<T>(what: &str, fallback: T, f: impl FnOnce() -> Result<T, String>) -> T {
    LAST_ERROR.with(|e| e.borrow_mut().clear());
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(message)) => {
            set_error(format!("{what}: {message}"));
            fallback
        }
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            set_error(format!("{what}: {message}"));
            fallback
        }
    }
}

/// Extension names live for the program (hal wants `&'static CStr`);
/// interned so re-adoption does not leak them again.
unsafe fn names(ptr: *const *const c_char, count: usize, what: &str) -> Result<Vec<&'static CStr>, String> {
    static INTERNED: std::sync::Mutex<Option<HashSet<&'static CStr>>> = std::sync::Mutex::new(None);
    if count == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(format!("{what}: {count} names but a null array"));
    }
    let mut guard = INTERNED.lock().map_err(|_| "extension name table poisoned".to_string())?;
    let set = guard.get_or_insert_with(HashSet::new);
    let mut out = Vec::with_capacity(count);
    for &p in std::slice::from_raw_parts(ptr, count) {
        if p.is_null() {
            return Err(format!("{what}: a null name"));
        }
        let name = CStr::from_ptr(p);
        let interned = match set.get(name) {
            Some(n) => *n,
            None => {
                let leaked: &'static CStr = Box::leak(name.to_owned().into_boxed_c_str());
                set.insert(leaked);
                leaked
            }
        };
        out.push(interned);
    }
    Ok(out)
}

fn handles<'a>(h: Option<&'a WGPUVulkanDeviceHandlesExt>) -> Result<&'a WGPUVulkanDeviceHandlesExt, String> {
    h.ok_or_else(|| "null handles".to_string())
}

/// One `WGPUTextureUsage` flag to the tracker state wgpu-core keeps, by
/// format (a depth/stencil attachment is not a colour target).
fn map_state(
    usage: native::WGPUTextureUsage,
    format: wgt::TextureFormat,
) -> Result<wgt::TextureUses, String> {
    let depth = texture_format_has_depth(format) || format == wgt::TextureFormat::Stencil8;
    Ok(match usage {
        native::WGPUTextureUsage_None => wgt::TextureUses::UNINITIALIZED,
        native::WGPUTextureUsage_CopySrc => wgt::TextureUses::COPY_SRC,
        native::WGPUTextureUsage_CopyDst => wgt::TextureUses::COPY_DST,
        native::WGPUTextureUsage_TextureBinding => wgt::TextureUses::RESOURCE,
        native::WGPUTextureUsage_StorageBinding if !depth => wgt::TextureUses::STORAGE_READ_WRITE,
        native::WGPUTextureUsage_RenderAttachment if depth => wgt::TextureUses::DEPTH_STENCIL_WRITE,
        native::WGPUTextureUsage_RenderAttachment => wgt::TextureUses::COLOR_TARGET,
        native::WGPUTextureUsage_StorageBinding => {
            return Err(format!("StorageBinding is not a state a {format:?} texture can be in"))
        }
        other => {
            return Err(format!(
                "state 0x{other:x} is not exactly one WGPUTextureUsage flag (CopySrc, CopyDst, TextureBinding, StorageBinding, RenderAttachment, or None)"
            ))
        }
    })
}

/// Wrap the exporter's `VkInstance` (never destroyed here). The instance has
/// only a Vulkan backend; it serves `wgpuInstanceProcessEvents`, waits and
/// the adapter/device calls below.
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstanceFromVulkanExt(
    handles_ptr: Option<&WGPUVulkanDeviceHandlesExt>,
) -> native::WGPUInstance {
    guarded("wgpuCreateInstanceFromVulkanExt", std::ptr::null(), || {
        let h = handles(handles_ptr)?;
        if h.get_instance_proc_addr.is_null() {
            return Err("null getInstanceProcAddr".into());
        }
        if h.instance == 0 {
            return Err("null VkInstance".into());
        }
        let extensions = names(h.instance_extensions, h.instance_extension_count, "instance extensions")?;
        let static_fn = ash::StaticFn {
            get_instance_proc_addr: std::mem::transmute::<*const c_void, vk::PFN_vkGetInstanceProcAddr>(
                h.get_instance_proc_addr,
            ),
        };
        let entry = ash::Entry::from_static_fn(static_fn);
        let raw_instance = ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(h.instance));
        let hal_instance = hal::vulkan::Instance::from_raw(
            entry,
            raw_instance,
            h.instance_api_version,
            0,
            None,
            extensions,
            wgt::InstanceFlags::empty(),
            wgt::MemoryBudgetThresholds::default(),
            false,
            Some(Box::new(|| {})),
        )
        .map_err(|e| e.to_string())?;
        Ok(Arc::into_raw(Arc::new(WGPUInstanceImpl {
            context: Arc::new(Context::from_hal_instance::<hal::api::Vulkan>(
                "wgpu-native (shared VkInstance)",
                hal_instance,
            )),
        })))
    })
}

/// The exporter's `VkPhysicalDevice` as an adapter of `instance` (which must
/// come from `wgpuCreateInstanceFromVulkanExt`).
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceCreateAdapterFromVulkanExt(
    instance: native::WGPUInstance,
    handles_ptr: Option<&WGPUVulkanDeviceHandlesExt>,
) -> native::WGPUAdapter {
    guarded("wgpuInstanceCreateAdapterFromVulkanExt", std::ptr::null(), || {
        let instance = instance.as_ref().ok_or("null instance")?;
        let h = handles(handles_ptr)?;
        if h.physical_device == 0 {
            return Err("null VkPhysicalDevice".into());
        }
        let context = &instance.context;
        let hal_instance = context
            .instance_as_hal::<hal::api::Vulkan>()
            .ok_or("not a Vulkan instance (use wgpuCreateInstanceFromVulkanExt)")?;
        let exposed = hal_instance
            .expose_adapter(vk::PhysicalDevice::from_raw(h.physical_device))
            .ok_or("the physical device is not usable by wgpu")?;
        let id = context.create_adapter_from_hal(exposed.into(), None);
        Ok(Arc::into_raw(Arc::new(WGPUAdapterImpl {
            context: context.clone(),
            id,
        })))
    })
}

/// Wrap the exporter's `VkDevice` and queue (never destroyed here).
/// `descriptor` lists the features and limits the `VkDevice` was created with
/// (NULL: none, WebGPU default limits).
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterCreateDeviceFromVulkanExt(
    adapter: native::WGPUAdapter,
    handles_ptr: Option<&WGPUVulkanDeviceHandlesExt>,
    descriptor: Option<&native::WGPUDeviceDescriptor>,
) -> native::WGPUDevice {
    guarded("wgpuAdapterCreateDeviceFromVulkanExt", std::ptr::null(), || {
        let adapter = adapter.as_ref().ok_or("null adapter")?;
        let h = handles(handles_ptr)?;
        if h.device == 0 {
            return Err("null VkDevice".into());
        }
        let extensions = names(h.device_extensions, h.device_extension_count, "device extensions")?;
        let context = &adapter.context;
        let (desc, _) = match descriptor {
            Some(d) => map_device_descriptor(d, wgt::Limits::default(), None),
            None => (wgt::DeviceDescriptor::default(), None),
        };
        let open = {
            let hal_instance = context
                .instance_as_hal::<hal::api::Vulkan>()
                .ok_or("not a Vulkan instance")?;
            let raw_instance = hal_instance.shared_instance().raw_instance();
            let hal_adapter = context
                .adapter_as_hal::<hal::api::Vulkan>(adapter.id)
                .ok_or("not a Vulkan adapter")?;
            // The queue the exporter names must exist on its physical device.
            let families =
                raw_instance.get_physical_device_queue_family_properties(hal_adapter.raw_physical_device());
            let family = families.get(h.queue_family_index as usize).ok_or_else(|| {
                format!(
                    "queue family {} does not exist (the physical device has {})",
                    h.queue_family_index,
                    families.len()
                )
            })?;
            if h.queue_index >= family.queue_count {
                return Err(format!(
                    "queue {} does not exist in family {} ({} queues)",
                    h.queue_index, h.queue_family_index, family.queue_count
                ));
            }
            let raw_device = ash::Device::load(raw_instance.fp_v1_0(), vk::Device::from_raw(h.device));
            hal_adapter
                .device_from_raw(
                    raw_device,
                    Some(Box::new(|| {})),
                    &extensions,
                    desc.required_features,
                    &desc.required_limits,
                    &desc.memory_hints,
                    h.queue_family_index,
                    h.queue_index,
                )
                .map_err(|e| e.to_string())?
        };
        let (device_id, queue_id) = context
            .create_device_from_hal(adapter.id, open.into(), &desc, None, None)
            .map_err(|e| e.to_string())?;
        Ok(Arc::into_raw(Arc::new_cyclic(|device| WGPUDeviceImpl {
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
        })))
    })
}

/// Convenience: the three calls above, keeping only the device (the
/// instance and adapter stay alive through the device's shared context).
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateDeviceFromVulkanExt(
    handles: Option<&WGPUVulkanDeviceHandlesExt>,
    descriptor: Option<&native::WGPUDeviceDescriptor>,
) -> native::WGPUDevice {
    let instance = wgpuCreateInstanceFromVulkanExt(handles);
    if instance.is_null() {
        return std::ptr::null();
    }
    let adapter = wgpuInstanceCreateAdapterFromVulkanExt(instance, handles);
    let device = if adapter.is_null() {
        std::ptr::null()
    } else {
        let d = wgpuAdapterCreateDeviceFromVulkanExt(adapter, handles, descriptor);
        Arc::decrement_strong_count(adapter);
        d
    };
    Arc::decrement_strong_count(instance);
    device
}

/// Import a `VkImage` owned by the exporter. `initialState` is the single
/// `WGPUTextureUsage` flag the image is in now (the agreed handoff state;
/// `None` means undefined contents); it must be part of the descriptor's
/// usage. The descriptor must be the image's own (2D, 1 mip, 1 sample).
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceImportVulkanImageExt(
    device: native::WGPUDevice,
    vk_image: u64,
    descriptor: Option<&native::WGPUTextureDescriptor>,
    initial_state: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    guarded("wgpuDeviceImportVulkanImageExt", std::ptr::null(), || {
        let device = device.as_ref().ok_or("null device")?;
        if vk_image == 0 {
            return Err("null VkImage".into());
        }
        let descriptor = descriptor.ok_or("null descriptor")?;
        let (device_id, context, error_sink) = (device.id, &device.context, &device.error_sink);
        let format = crate::conv::map_texture_format(descriptor.format)
            .ok_or_else(|| format!("unknown texture format 0x{:x}", descriptor.format))?;
        let usage: wgt::TextureUsages = crate::conv::from_u64_bits(descriptor.usage)
            .ok_or_else(|| format!("unknown texture usage bits 0x{:x}", descriptor.usage))?;
        let size = crate::conv::map_extent3d(&descriptor.size);
        if size.width == 0 || size.height == 0 || size.depth_or_array_layers != 1 {
            return Err(format!("size {}x{}x{}: only one 2D layer is supported", size.width, size.height, size.depth_or_array_layers));
        }
        if descriptor.mipLevelCount != 1 || descriptor.sampleCount != 1 {
            return Err("only 1 mip level and 1 sample are supported".into());
        }
        let dimension = crate::conv::map_texture_dimension(descriptor.dimension).unwrap_or(wgt::TextureDimension::D2);
        if dimension != wgt::TextureDimension::D2 {
            return Err("only 2D images are supported".into());
        }
        let state = map_state(initial_state, format)?;
        if initial_state != native::WGPUTextureUsage_None && (descriptor.usage & initial_state) == 0 {
            return Err(format!(
                "initial state 0x{initial_state:x} is not part of the texture's usage 0x{:x}",
                descriptor.usage
            ));
        }
        let desc = wgt::TextureDescriptor {
            label: crate::utils::string_view_into_label(descriptor.label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: vec![],
        };
        let hal_texture = {
            let hal_device = context
                .device_as_hal::<hal::api::Vulkan>(device_id)
                .ok_or("not a Vulkan device")?;
            hal_device.texture_from_raw(
                vk::Image::from_raw(vk_image),
                &hal::TextureDescriptor {
                    label: None,
                    size: desc.size,
                    mip_level_count: 1,
                    sample_count: 1,
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
        let (texture_id, error) =
            context.create_texture_from_hal(Box::new(hal_texture), device_id, &desc, state, None);
        if let Some(cause) = error {
            let message = cause.to_string();
            handle_error(error_sink, cause, desc.label, "wgpuDeviceImportVulkanImageExt");
            context.texture_drop(texture_id);
            return Err(message);
        }
        Ok(Arc::into_raw(Arc::new(WGPUTextureImpl {
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
                mip_level_count: 1,
                sample_count: 1,
            },
        })))
    })
}

/// Record a transition of the whole texture to one `WGPUTextureUsage` state
/// (wgpu's `CommandEncoder::transition_resources`), for handing it back.
/// Returns 1, or 0 with an error message when nothing was recorded.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderTransitionTextureExt(
    command_encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    state: native::WGPUTextureUsage,
) -> native::WGPUBool {
    guarded("wgpuCommandEncoderTransitionTextureExt", 0, || {
        let encoder: &WGPUCommandEncoderImpl = command_encoder.as_ref().ok_or("null command encoder")?;
        let texture = texture.as_ref().ok_or("null texture")?;
        let format = crate::conv::map_texture_format(texture.data.format)
            .ok_or("the texture has an unknown format")?;
        let uses = map_state(state, format)?;
        if state != native::WGPUTextureUsage_None && (texture.data.usage & state) == 0 {
            return Err(format!(
                "state 0x{state:x} is not part of the texture's usage 0x{:x}",
                texture.data.usage
            ));
        }
        let transition = wgt::TextureTransition {
            texture: texture.id,
            selector: None,
            state: uses,
        };
        if let Err(cause) = encoder.context.command_encoder_transition_resources(
            encoder.id,
            std::iter::empty(),
            std::iter::once(transition),
        ) {
            let message = cause.to_string();
            handle_error(&encoder.error_sink, cause, None, "wgpuCommandEncoderTransitionTextureExt");
            return Err(message);
        }
        Ok(1)
    })
}
