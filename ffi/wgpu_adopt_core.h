/**
 * @file wgpu_adopt_core.h
 * @brief Adopt a device, queue and textures a Rust host owns in the same
 * wgpu-core `Global` (cargo feature `adopt-core`; BlankMauser fork). See
 * src/adopt_core.rs for the ownership and misuse rules.
 *
 * Ids are wgpu-core's, as `uint64_t` (`wgpu_native::adopt_core::id_to_u64`).
 * Every call returns NULL on misuse and leaves a message for
 * wgpuCoreAdoptLastErrorExt. Adopted handles never drop or destroy the host's
 * objects (wgpuDeviceDestroy and wgpuTextureDestroy on them are no-ops); adoption
 * makes one non-blocking poll and one empty submission as liveness checks. Every id
 * must come from the Global passed to wgpuCreateInstanceFromCoreExt: an id means
 * something only in its own Global, and one from another is not detected.
 */
#ifndef WGPU_ADOPT_CORE_H_
#define WGPU_ADOPT_CORE_H_

#include "wgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/** `global` is Arc::as_ptr of the host's live Arc<wgpu_core::global::Global>. */
WGPUInstance wgpuCreateInstanceFromCoreExt(const void* global);
WGPUAdapter wgpuInstanceAdoptAdapterExt(WGPUInstance instance, uint64_t adapterId);
/** `descriptor` may be NULL; only its uncapturedErrorCallbackInfo and deviceLostCallbackInfo are used
 *  (the defaults treat both as fatal). The queue must be the device's own (checked with an empty submission). */
WGPUDevice wgpuAdapterAdoptDeviceExt(WGPUAdapter adapter, uint64_t deviceId, uint64_t queueId, const WGPUDeviceDescriptor* descriptor);
/** `descriptor` must be the texture's own; the handle's getters answer from it. */
WGPUTexture wgpuDeviceAdoptTextureExt(WGPUDevice device, uint64_t textureId, const WGPUTextureDescriptor* descriptor);
/** Copies this thread's last adoption error (UTF-8, not NUL-terminated); returns its length. */
size_t wgpuCoreAdoptLastErrorExt(char* out, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
