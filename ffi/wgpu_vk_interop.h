/* Vulkan interop extension (BlankMauser/wgpu-native fork, spike).
 * Drive a VkDevice created by another wgpu (e.g. the Rust wgpu 30 crate) and
 * import its VkImages. See src/vk_interop.rs for the handoff protocol:
 * serialise queue submissions, and agree on one texture state at every
 * handoff (import with it; transition back to it before handing back). */
#ifndef WGPU_VK_INTEROP_H_
#define WGPU_VK_INTEROP_H_

#include "webgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WGPUVulkanDeviceHandlesExt {
    void const * getInstanceProcAddr; /* PFN_vkGetInstanceProcAddr */
    uint64_t instance;                /* VkInstance */
    uint32_t instanceApiVersion;
    size_t instanceExtensionCount;
    char const * const * instanceExtensions;
    uint64_t physicalDevice;          /* VkPhysicalDevice */
    uint64_t device;                  /* VkDevice */
    size_t deviceExtensionCount;
    char const * const * deviceExtensions;
    uint32_t queueFamilyIndex;
    uint32_t queueIndex;
} WGPUVulkanDeviceHandlesExt;

typedef void (*WGPUVulkanHandoffLockExt)(void * context);

/* Everything the adopting side needs for one shared texture, built by the
 * owner from its own objects (Rust: crates/wgpu-vk-handoff), so nothing is
 * typed by hand.
 *
 * Lifetimes: the struct (and the extension name arrays it points to) must
 * stay valid while the adopting side reads it, i.e. during adoption and
 * import; it copies what it keeps. The lock functions and lockContext are
 * copied and called until the adopting side's teardown returns (LunarFenrir:
 * Gpu::close), so lockContext must stay valid until then.
 *
 * Versioning: append-only. Fields are never removed, reordered, resized or
 * reinterpreted; new fields are only appended. structSize is the version: a
 * reader requires structSize >= the size of the layout it knows, and a newer
 * writer's extra fields are ignored. (No separate version field: with one
 * layout rule there is nothing for it to say that structSize does not.) */
typedef struct WGPUVulkanHandoffExt {
    uint32_t structSize;               /* sizeof(WGPUVulkanHandoffExt) */
    WGPUVulkanDeviceHandlesExt device;
    uint64_t image;                    /* VkImage: 2D, 1 layer, 1 mip, 1 sample */
    uint32_t width;
    uint32_t height;
    WGPUTextureFormat format;
    WGPUTextureUsage usage;            /* the texture's usage */
    WGPUTextureUsage state;            /* the agreed handoff state: one flag */
    /* The owner's lock for the shared queue (both NULL: one thread only).
     * The owner takes it around its own submissions; the adopting side around
     * its submissions and teardown. Calling into the adopting side while
     * holding it deadlocks. */
    WGPUVulkanHandoffLockExt lock;
    WGPUVulkanHandoffLockExt unlock;
    void * lockContext;
} WGPUVulkanHandoffExt;

/* Instance -> adapter -> device over the exporter's Vulkan objects (none of
 * them is ever destroyed by wgpu-native). Each returns NULL on failure. */
WGPUInstance wgpuCreateInstanceFromVulkanExt(WGPUVulkanDeviceHandlesExt const * handles);
WGPUAdapter wgpuInstanceCreateAdapterFromVulkanExt(WGPUInstance instance,
                                                   WGPUVulkanDeviceHandlesExt const * handles);
WGPUDevice wgpuAdapterCreateDeviceFromVulkanExt(WGPUAdapter adapter,
                                                WGPUVulkanDeviceHandlesExt const * handles,
                                                WGPUDeviceDescriptor const * descriptor);
/* Convenience: all three, returning only the device.
 * descriptor: features/limits the VkDevice was created with (NULL: none,
 * WebGPU default limits). Returns NULL on failure. Never destroys the VkDevice. */
WGPUDevice wgpuCreateDeviceFromVulkanExt(WGPUVulkanDeviceHandlesExt const * handles,
                                         WGPUDeviceDescriptor const * descriptor);
/* initialState: exactly one WGPUTextureUsage flag (None = undefined contents),
 * part of the descriptor's usage. RenderAttachment on a depth/stencil format
 * is the depth-stencil attachment state. The descriptor must be the image's
 * own (2D, 1 mip, 1 sample); Vulkan cannot check it, so build it from the
 * exporter's texture, never by hand. */
WGPUTexture wgpuDeviceImportVulkanImageExt(WGPUDevice device, uint64_t vkImage,
                                           WGPUTextureDescriptor const * descriptor,
                                           WGPUTextureUsage initialState);
/* Returns 1, or 0 (nothing recorded) with a message for
 * wgpuVulkanInteropLastErrorExt. state: exactly one WGPUTextureUsage flag. */
WGPUBool wgpuCommandEncoderTransitionTextureExt(WGPUCommandEncoder encoder, WGPUTexture texture,
                                                WGPUTextureUsage state);
/* This thread's last interop error: copies it (UTF-8, not NUL-terminated)
 * into out when capacity suffices; returns its length (0: none). Every entry
 * point above clears it on entry and sets it when it returns NULL / 0; none
 * of them aborts on bad input. */
size_t wgpuVulkanInteropLastErrorExt(char * out, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
