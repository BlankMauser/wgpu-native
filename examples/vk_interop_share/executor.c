/* Stand-in for LunarFenrir's executor: talks only to wgpu-native's C API
 * (webgpu.h v29 + the fork's Vulkan interop extension), as wgpu_mbt would.
 * Handoff state for the shared texture: TextureBinding (shader-read). */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "webgpu.h"
#include "wgpu.h"
#include "wgpu_vk_interop.h"

static int g_errors;
static void on_log(WGPULogLevel l, WGPUStringView m, void *u) {
  (void)u;
  if (l <= WGPULogLevel_Warn) {
    if (l == WGPULogLevel_Error) g_errors++;
    fprintf(stderr, "[wgpu-native %d] %.*s\n", (int)l, (int)m.length, m.data);
  }
}
int exec_errors(void) { return g_errors; }
/* HANDOFF_RT: agree on RenderAttachment instead of TextureBinding. */
static WGPUTextureUsage handoff_state(void) {
  return getenv("HANDOFF_RT") ? WGPUTextureUsage_RenderAttachment : WGPUTextureUsage_TextureBinding;
}

static WGPUDevice g_device;
static WGPUQueue g_queue;
static WGPUTexture g_target;
static WGPURenderPipeline g_pipeline;

static WGPUStringView sv(const char *s) { WGPUStringView v = {s, WGPU_STRLEN}; return v; }

static const char *WGSL =
    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
    "  var p = array(vec2f(-1.0, 1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0));\n"
    "  return vec4f(p[i], 0.0, 1.0);\n"
    "}\n"
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.5, 0.0, 1.0); }\n";

int exec_init(const WGPUVulkanDeviceHandlesExt *h) {
  wgpuSetLogCallback(on_log, NULL);
  wgpuSetLogLevel(WGPULogLevel_Warn);
  g_device = wgpuCreateDeviceFromVulkanExt(h, NULL);
  if (!g_device) return 1;
  g_queue = wgpuDeviceGetQueue(g_device);

  WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
  src.code = sv(WGSL);
  WGPUShaderModuleDescriptor smd = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
  smd.nextInChain = &src.chain;
  WGPUShaderModule sm = wgpuDeviceCreateShaderModule(g_device, &smd);

  WGPUColorTargetState ct = WGPU_COLOR_TARGET_STATE_INIT;
  ct.format = WGPUTextureFormat_RGBA8Unorm;
  WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
  fs.module = sm;
  fs.entryPoint = sv("fs");
  fs.targetCount = 1;
  fs.targets = &ct;
  WGPURenderPipelineDescriptor rpd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
  rpd.label = sv("standin-triangle");
  rpd.vertex.module = sm;
  rpd.vertex.entryPoint = sv("vs");
  rpd.fragment = &fs;
  g_pipeline = wgpuDeviceCreateRenderPipeline(g_device, &rpd);
  wgpuShaderModuleRelease(sm);
  return g_pipeline ? 0 : 2;
}

int exec_import(uint64_t image, uint32_t w, uint32_t hgt) {
  WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
  td.label = sv("shared-from-rust");
  td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding |
             WGPUTextureUsage_CopySrc;
  td.size.width = w;
  td.size.height = hgt;
  td.size.depthOrArrayLayers = 1;
  td.format = WGPUTextureFormat_RGBA8Unorm;
  /* BAD_HANDOFF: positive control for the validation layer (lie about the state). */
  WGPUTextureUsage state = getenv("BAD_HANDOFF") ? WGPUTextureUsage_CopyDst : handoff_state();
  g_target = wgpuDeviceImportVulkanImageExt(g_device, image, &td, state);
  return g_target ? 0 : 1;
}

/* frame 0: clear blue, orange triangle over the top-left half.
 * frame 1: clear green. Hands back in TextureBinding state. */
void exec_frame(int frame) {
  WGPUTextureView view = wgpuTextureCreateView(g_target, NULL);
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, NULL);
  WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
  ca.view = view;
  ca.loadOp = WGPULoadOp_Clear;
  ca.storeOp = WGPUStoreOp_Store;
  WGPUColor blue = {0, 0, 1, 1}, green = {0, 1, 0, 1};
  ca.clearValue = frame == 0 ? blue : green;
  WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
  rp.colorAttachmentCount = 1;
  rp.colorAttachments = &ca;
  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
  if (frame == 0) {
    wgpuRenderPassEncoderSetPipeline(pass, g_pipeline);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
  }
  wgpuRenderPassEncoderEnd(pass);
  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandEncoderTransitionTextureExt(enc, g_target, handoff_state());
  WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(g_queue, 1, &cb);
  wgpuCommandBufferRelease(cb);
  wgpuCommandEncoderRelease(enc);
  wgpuTextureViewRelease(view);
}

void exec_shutdown(void) {
  wgpuTextureRelease(g_target);
  wgpuRenderPipelineRelease(g_pipeline);
  wgpuQueueRelease(g_queue);
  wgpuDeviceRelease(g_device); /* waits for the GPU; never destroys the VkDevice */
}
