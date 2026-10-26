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

/* descriptor: features/limits the VkDevice was created with (NULL: none,
 * WebGPU default limits). Returns NULL on failure. Never destroys the VkDevice. */
WGPUDevice wgpuCreateDeviceFromVulkanExt(WGPUVulkanDeviceHandlesExt const * handles,
                                         WGPUDeviceDescriptor const * descriptor);
/* initialState: exactly one WGPUTextureUsage flag (None = undefined contents). */
WGPUTexture wgpuDeviceImportVulkanImageExt(WGPUDevice device, uint64_t vkImage,
                                           WGPUTextureDescriptor const * descriptor,
                                           WGPUTextureUsage initialState);
void wgpuCommandEncoderTransitionTextureExt(WGPUCommandEncoder encoder, WGPUTexture texture,
                                            WGPUTextureUsage state);

#ifdef __cplusplus
}
#endif

#endif
