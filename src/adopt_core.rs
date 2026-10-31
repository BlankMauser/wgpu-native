//! Adopt a device, queue and textures a host already owns in the same wgpu-core
//! `Global` (feature `adopt-core`; BlankMauser fork).
//!
//! The host is a Rust program on the `wgpu` 30 crate whose Cargo graph also
//! holds this crate (as an rlib) and whose `[patch.crates-io]` points
//! `wgpu`, `wgpu-core` and `wgpu-hal` at one copy (ShibaPan's `vendor/`). There is
//! then exactly one wgpu-core in the process, one `Global`, one device, one
//! queue and one resource tracker: a texture the host created is the same
//! object here, with no handoff protocol, no queue lock and no state to agree
//! on. The host reads its ids with ShibaPan's `Instance::as_core_global`,
//! `Adapter::as_core_id`, `Device::as_core_id`, `Queue::as_core_id` and
//! `Texture::as_core_id`, and turns them into `u64` with [`id_to_u64`].
//!
//! Ownership: the host keeps every id. Adopted handles never drop, poll or
//! destroy what they wrap (`adopted: true`); releasing them only drops this
//! side's `Arc<Global>` and `Arc`s. Everything created through the adopted
//! device afterwards (buffers, pipelines, views, textures) is this side's and
//! is dropped normally. The host must keep its own objects alive until every
//! handle adopted here is released (`wgpuInstanceLiveObjectsExt`-style checks
//! are the host's job).
//!
//! Misuse returns NULL with a per-thread message for
//! [`wgpuCoreAdoptLastErrorExt`]; nothing unwinds into C. Refused: a null or
//! zero argument, an id that is not live in the `Global`, a device that is lost
//! or not from the adapter given, a texture that is destroyed. What cannot be
//! checked, and is the caller's: that `global` really is a pointer from
//! `Arc::as_ptr` of a live `Arc<wgpu_core::global::Global>`, and that a
//! texture belongs to the device (wgpu-core rejects another device's texture
//! with a validation error when it is used).

use crate::native;
use crate::{
    Context, ErrorSinkRaw, QueueId, TextureData, WGPUAdapterImpl, WGPUDeviceImpl,
    WGPUInstanceImpl, WGPUTextureImpl, DEFAULT_DEVICE_LOST_HANDLER,
};
use parking_lot::Mutex;
use std::cell::RefCell;
use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use wgc::id::{self, markers};

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: String) {
    log::error!("adopt core: {message}");
    LAST_ERROR.with(|e| *e.borrow_mut() = message);
}

/// Copies this thread's last adoption error (UTF-8, not NUL-terminated) into
/// `out`; returns its full length (0: no error). Cleared by the next adoption call.
#[no_mangle]
pub unsafe extern "C" fn wgpuCoreAdoptLastErrorExt(out: *mut c_char, capacity: usize) -> usize {
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        if !out.is_null() {
            let n = e.len().min(capacity);
            std::ptr::copy_nonoverlapping(e.as_ptr(), out.cast::<u8>(), n);
        }
        e.len()
    })
}

fn guarded<T>(what: &str, fallback: T, f: impl FnOnce() -> Result<T, String>) -> T {
    LAST_ERROR.with(|e| e.borrow_mut().clear());
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(message)) => {
            set_error(format!("{what}: {message}"));
            fallback
        }
        Err(panic) => {
            // wgpu-core panics on an id that is not live; that is a refusal here.
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "panic".into());
            set_error(format!("{what}: {message}"));
            fallback
        }
    }
}

/// A wgpu-core id as the `u64` the adoption calls take: its index in the low
/// 32 bits, its epoch in the high 32 (`RawId::zip`).
pub fn id_to_u64<T: wgc::id::Marker>(id: id::Id<T>) -> u64 {
    let (index, epoch) = id.into_raw().unzip();
    u64::from(index) | (u64::from(epoch) << 32)
}

fn id_from_u64<T: wgc::id::Marker>(raw: u64, what: &str) -> Result<id::Id<T>, String> {
    if raw == 0 {
        return Err(format!("zero {what} id"));
    }
    // Non-zero, so `zip` does not panic. Whether it names a live object is checked by the caller.
    let raw_id = wgc::id::RawId::zip(raw as u32, (raw >> 32) as u32);
    // SAFETY: a RawId of the right kind is the caller's claim; liveness is checked by the caller.
    Ok(unsafe { id::Id::from_raw(raw_id) })
}

/// Share the host's `Global`. `global` is `Arc::as_ptr` of a live
/// `Arc<wgpu_core::global::Global>` (`Instance::as_core_global`); this takes
/// its own strong count, so the host may drop its `Arc` afterwards.
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstanceFromCoreExt(global: *const c_void) -> native::WGPUInstance {
    guarded("wgpuCreateInstanceFromCoreExt", std::ptr::null(), || {
        if global.is_null() {
            return Err("null Global".into());
        }
        let global = global.cast::<Context>();
        Arc::increment_strong_count(global);
        let context = Arc::from_raw(global);
        Ok(Arc::into_raw(Arc::new(WGPUInstanceImpl { context })))
    })
}

/// The host's adapter (`Adapter::as_core_id`) as a handle of `instance`
/// (which must come from `wgpuCreateInstanceFromCoreExt`).
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceAdoptAdapterExt(
    instance: native::WGPUInstance,
    adapter_id: u64,
) -> native::WGPUAdapter {
    guarded("wgpuInstanceAdoptAdapterExt", std::ptr::null(), || {
        let instance = instance.as_ref().ok_or("null instance")?;
        let id = id_from_u64::<markers::Adapter>(adapter_id, "adapter")?;
        let context = &instance.context;
        // Panics (caught) when the id is not live in this Global.
        let _ = context.adapter_get_info(id);
        Ok(Arc::into_raw(Arc::new(WGPUAdapterImpl {
            context: context.clone(),
            id,
            adopted: true,
        })))
    })
}

/// The host's device and queue (`Device::as_core_id`, `Queue::as_core_id`).
/// The device must be live (not lost) and open on `adapter`. Features and limits
/// are the host's own: read them with `wgpuDeviceHasFeature` / `wgpuDeviceGetLimits`.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterAdoptDeviceExt(
    adapter: native::WGPUAdapter,
    device_id: u64,
    queue_id: u64,
) -> native::WGPUDevice {
    guarded("wgpuAdapterAdoptDeviceExt", std::ptr::null(), || {
        let adapter = adapter.as_ref().ok_or("null adapter")?;
        let context = &adapter.context;
        let device = id_from_u64::<markers::Device>(device_id, "device")?;
        let queue = id_from_u64::<markers::Queue>(queue_id, "queue")?;
        // Panics (caught) when either id is not live in this Global.
        let device_info = context.device_adapter_info(device);
        let adapter_info = context.adapter_get_info(adapter.id);
        if device_info != adapter_info {
            return Err(format!(
                "the device is on adapter {:?}, not {:?}",
                device_info.name, adapter_info.name
            ));
        }
        context
            .device_poll(device, wgt::PollType::Poll)
            .map_err(|e| format!("the device is not usable: {e}"))?;
        // Panics (caught) when the queue id is not live. wgpu-core 30 cannot say
        // which device a queue belongs to: pass the queue of the same wgpu::Device.
        let _ = context.queue_get_timestamp_period(queue);
        Ok(Arc::into_raw(Arc::new_cyclic(|weak| WGPUDeviceImpl {
            context: context.clone(),
            id: device,
            queue: Arc::new(QueueId {
                context: context.clone(),
                id: queue,
                adopted: true,
            }),
            error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new(
                DEFAULT_DEVICE_LOST_HANDLER,
                weak.clone(),
            ))),
            adopted: true,
        })))
    })
}

/// The host's texture (`Texture::as_core_id`) as a handle of `device`.
/// `descriptor` must be the texture's own (format, size, usage, dimension,
/// mips, samples): wgpu-core cannot report them back, so the handle's getters
/// answer from it. A texture that is gone or destroyed is refused.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceAdoptTextureExt(
    device: native::WGPUDevice,
    texture_id: u64,
    descriptor: Option<&native::WGPUTextureDescriptor>,
) -> native::WGPUTexture {
    guarded("wgpuDeviceAdoptTextureExt", std::ptr::null(), || {
        let device = device.as_ref().ok_or("null device")?;
        let descriptor = descriptor.ok_or("null descriptor")?;
        let id = id_from_u64::<markers::Texture>(texture_id, "texture")?;
        crate::conv::map_texture_format(descriptor.format)
            .ok_or_else(|| format!("unknown texture format 0x{:x}", descriptor.format))?;
        let context = &device.context;
        // A default view proves the texture is live and not destroyed (the
        // first panics, the second errors), then goes away.
        let (view, error) = context.texture_create_view(id, &wgc::resource::TextureViewDescriptor::default(), None);
        if let Some(cause) = error {
            // Not through the error sink: its uncaptured-error default aborts.
            context.texture_view_drop(view);
            return Err(format!("the texture is not usable: {cause}"));
        }
        context.texture_view_drop(view);
        Ok(Arc::into_raw(Arc::new(WGPUTextureImpl {
            context: context.clone(),
            id,
            error_sink: device.error_sink.clone(),
            surface_id: None,
            has_surface_presented: Arc::default(),
            adopted: true,
            data: TextureData {
                usage: descriptor.usage,
                dimension: descriptor.dimension,
                size: descriptor.size,
                format: descriptor.format,
                mip_level_count: descriptor.mipLevelCount,
                sample_count: descriptor.sampleCount,
            },
        })))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn last() -> String {
        let mut b = vec![0u8; 512];
        let n = wgpuCoreAdoptLastErrorExt(b.as_mut_ptr().cast(), b.len());
        String::from_utf8_lossy(&b[..n.min(512)]).into()
    }

    /// Misuse of the adoption calls returns NULL with a message and never
    /// unwinds into C. Needs an adapter: skipped without one.
    #[test]
    fn misuse_is_refused() {
        let global = Arc::new(Context::new("adopt-core test", wgt::InstanceDescriptor::new_without_display_handle(), None));
        let adapters = global.enumerate_adapters(wgt::Backends::all(), false);
        let Some(&adapter_id) = adapters.first() else {
            eprintln!("no adapter: skipped");
            return;
        };
        let (device_id, queue_id) = global
            .adapter_request_device(adapter_id, &wgt::DeviceDescriptor::default(), None, None)
            .expect("request_device");
        let texture_desc = wgt::TextureDescriptor {
            label: None,
            size: wgt::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgt::TextureDimension::D2,
            format: wgt::TextureFormat::Rgba8Unorm,
            usage: wgt::TextureUsages::RENDER_ATTACHMENT | wgt::TextureUsages::COPY_SRC,
            view_formats: vec![],
        };
        let (texture_id, error) = global.device_create_texture(device_id, &texture_desc, None);
        assert!(error.is_none());
        let mut native_desc: native::WGPUTextureDescriptor = unsafe { std::mem::zeroed() };
        native_desc.usage = (native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc) as _;
        native_desc.dimension = native::WGPUTextureDimension_2D;
        native_desc.size = native::WGPUExtent3D { width: 4, height: 4, depthOrArrayLayers: 1 };
        native_desc.format = native::WGPUTextureFormat_RGBA8Unorm;
        native_desc.mipLevelCount = 1;
        native_desc.sampleCount = 1;
        unsafe {
            // The good path.
            let instance = wgpuCreateInstanceFromCoreExt(Arc::as_ptr(&global).cast());
            assert!(!instance.is_null(), "{}", last());
            let adapter = wgpuInstanceAdoptAdapterExt(instance, id_to_u64(adapter_id));
            assert!(!adapter.is_null(), "{}", last());
            let device = wgpuAdapterAdoptDeviceExt(adapter, id_to_u64(device_id), id_to_u64(queue_id));
            assert!(!device.is_null(), "{}", last());
            let texture = wgpuDeviceAdoptTextureExt(device, id_to_u64(texture_id), Some(&native_desc));
            assert!(!texture.is_null(), "{}", last());
            // Refused: each returns NULL with a message.
            assert!(wgpuCreateInstanceFromCoreExt(std::ptr::null()).is_null());
            assert!(last().contains("null Global"));
            assert!(wgpuInstanceAdoptAdapterExt(instance, 0).is_null());
            assert!(last().contains("zero"));
            let dead = 0xdead_0000_0007_u64;
            assert!(wgpuAdapterAdoptDeviceExt(adapter, dead, id_to_u64(queue_id)).is_null(), "dead device");
            assert!(!last().is_empty());
            assert!(wgpuAdapterAdoptDeviceExt(adapter, id_to_u64(device_id), dead).is_null(), "dead queue");
            assert!(!last().is_empty());
            assert!(wgpuDeviceAdoptTextureExt(device, dead, Some(&native_desc)).is_null(), "dead texture");
            assert!(!last().is_empty());
            // A destroyed texture, then a dropped one.
            let (gone, _) = global.device_create_texture(device_id, &texture_desc, None);
            global.texture_destroy(gone);
            assert!(wgpuDeviceAdoptTextureExt(device, id_to_u64(gone), Some(&native_desc)).is_null());
            assert!(last().contains("not usable"), "{}", last());
            global.texture_drop(gone);
            assert!(wgpuDeviceAdoptTextureExt(device, id_to_u64(gone), Some(&native_desc)).is_null());
            // A device that is not on the adapter given.
            if let Some(&other) = adapters.get(1) {
                let other_adapter = wgpuInstanceAdoptAdapterExt(instance, id_to_u64(other));
                assert!(!other_adapter.is_null(), "{}", last());
                assert!(wgpuAdapterAdoptDeviceExt(other_adapter, id_to_u64(device_id), id_to_u64(queue_id)).is_null());
                assert!(last().contains("not"), "{}", last());
                crate::wgpuAdapterRelease(other_adapter);
            }
            // Releasing adopted handles leaves the host's objects alone.
            crate::wgpuTextureRelease(texture);
            crate::wgpuDeviceRelease(device);
            crate::wgpuAdapterRelease(adapter);
            crate::wgpuInstanceRelease(instance);
        }
        assert!(global.device_poll(device_id, wgt::PollType::Poll).is_ok(), "the host's device must survive");
    }
}
