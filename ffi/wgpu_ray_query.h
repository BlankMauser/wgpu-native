/**
 * @file wgpu_ray_query.h
 * @brief Acceleration structures for ray queries (BlankMauser fork). See
 * src/ray_query.rs.
 *
 * webgpu.h and wgpu.h name the feature (WGPUNativeFeature_RayQuery) but have no
 * call to make, build or bind an acceleration structure. Request the feature
 * on the device (that also turns wgpu's experimental features on for it),
 * create BLASes and a TLAS, record their builds, and bind the TLAS to a WGSL
 * `acceleration_structure` global (`enable wgpu_ray_query;`).
 *
 * Errors go to the device's error sink, like every other call's.
 */
#ifndef WGPU_RAY_QUERY_H_
#define WGPU_RAY_QUERY_H_

#include "wgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WGPUBlasImpl* WGPUBlasExt;
typedef struct WGPUTlasImpl* WGPUTlasExt;

/** Buffer usages of BLAS geometry (vertex and index) and TLAS instance buffers. */
#define WGPUBufferUsage_BlasInputExt ((WGPUBufferUsage)0x0000000000000400)
#define WGPUBufferUsage_TlasInputExt ((WGPUBufferUsage)0x0000000000000800)

/** wgpu's AccelerationStructureFlags. */
#define WGPUAccelerationStructureFlags_AllowUpdateExt 1u
#define WGPUAccelerationStructureFlags_AllowCompactionExt 2u
#define WGPUAccelerationStructureFlags_PreferFastTraceExt 4u
#define WGPUAccelerationStructureFlags_PreferFastBuildExt 8u
#define WGPUAccelerationStructureFlags_LowMemoryExt 16u

/** One triangle geometry of a BLAS: the most its builds hold.
 *  indexFormat WGPUIndexFormat_Undefined: no index buffer. */
typedef struct WGPUBlasTriangleGeometryExt {
    WGPUVertexFormat vertexFormat;
    uint32_t vertexCount;
    WGPUIndexFormat indexFormat;
    uint32_t indexCount;
    WGPUBool opaque;
} WGPUBlasTriangleGeometryExt;

/** The buffers of one geometry in a build, in the BLAS's geometry order. */
typedef struct WGPUBlasBuildGeometryExt {
    WGPUBuffer vertexBuffer;
    uint32_t firstVertex;
    uint64_t vertexStride;
    WGPUBuffer indexBuffer; /* NULL without an index format */
    uint32_t firstIndex;
} WGPUBlasBuildGeometryExt;

typedef struct WGPUBlasBuildExt {
    WGPUBlasExt blas;
    const WGPUBlasBuildGeometryExt* geometries;
    size_t geometryCount;
} WGPUBlasBuildExt;

/** A row-major 3x4 transform, 24 bits the shader reads back, an 8-bit mask.
 *  A NULL blas leaves the slot empty. */
typedef struct WGPUTlasInstanceExt {
    WGPUBlasExt blas;
    float transform[12];
    uint32_t custom;
    uint32_t mask;
} WGPUTlasInstanceExt;

typedef struct WGPUTlasBuildExt {
    WGPUTlasExt tlas;
    const WGPUTlasInstanceExt* instances;
    size_t instanceCount;
} WGPUTlasBuildExt;

typedef struct WGPUTlasLayoutEntryExt {
    uint32_t binding;
    WGPUShaderStage visibility;
    WGPUBool vertexReturn;
} WGPUTlasLayoutEntryExt;

typedef struct WGPUTlasBindingExt {
    uint32_t binding;
    WGPUTlasExt tlas;
} WGPUTlasBindingExt;

WGPUBlasExt wgpuDeviceCreateBlasExt(WGPUDevice device, const WGPUBlasTriangleGeometryExt* geometries, size_t geometryCount, uint32_t flagBits, WGPUStringView label);
WGPUTlasExt wgpuDeviceCreateTlasExt(WGPUDevice device, uint32_t maxInstances, uint32_t flagBits, WGPUStringView label);
void wgpuBlasReleaseExt(WGPUBlasExt blas);
void wgpuTlasReleaseExt(WGPUTlasExt tlas);
/** A TLAS is built after the BLASes its instances name (the same command or an earlier one). */
void wgpuCommandEncoderBuildAccelerationStructuresExt(WGPUCommandEncoder encoder, const WGPUBlasBuildExt* blasBuilds, size_t blasCount, const WGPUTlasBuildExt* tlasBuilds, size_t tlasCount);
/** wgpuDeviceCreateBindGroupLayout and wgpuDeviceCreateBindGroup with TLAS bindings beside the descriptor's entries. */
WGPUBindGroupLayout wgpuDeviceCreateBindGroupLayoutTlasExt(WGPUDevice device, const WGPUBindGroupLayoutDescriptor* descriptor, const WGPUTlasLayoutEntryExt* tlasEntries, size_t tlasCount);
WGPUBindGroup wgpuDeviceCreateBindGroupTlasExt(WGPUDevice device, const WGPUBindGroupDescriptor* descriptor, const WGPUTlasBindingExt* tlasEntries, size_t tlasCount);

#ifdef __cplusplus
}
#endif

#endif
