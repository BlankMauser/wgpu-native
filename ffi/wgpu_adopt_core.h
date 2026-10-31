/**
 * @file wgpu_adopt_core.h
 * @brief Adopt a device, queue and textures a Rust host owns in the same
 * wgpu-core `Global` (cargo feature `adopt-core`; BlankMauser fork). See
 * src/adopt_core.rs for the ownership and misuse rules.
 *
 * Ids are wgpu-core's, as `uint64_t` (`wgpu_native::adopt_core::id_to_u64`).
 * Every call returns NULL on misuse and leaves a message for
 * wgpuCoreAdoptLastErrorExt. Adopted handles never drop, poll or destroy the
 * host's objects.
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
WGPUDevice wgpuAdapterAdoptDeviceExt(WGPUAdapter adapter, uint64_t deviceId, uint64_t queueId);
/** `descriptor` must be the texture's own; the handle's getters answer from it. */
WGPUTexture wgpuDeviceAdoptTextureExt(WGPUDevice device, uint64_t textureId, const WGPUTextureDescriptor* descriptor);
/** Copies this thread's last adoption error (UTF-8, not NUL-terminated); returns its length. */
size_t wgpuCoreAdoptLastErrorExt(char* out, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
