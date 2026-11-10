//! Acceleration structures for ray queries (BlankMauser fork; `ffi/wgpu_ray_query.h`).
//!
//! `webgpu.h` and `wgpu.h` have the `RayQuery` feature name but no calls to make,
//! build or bind an acceleration structure. These are the calls: bottom-level
//! structures (BLAS) of triangle geometries, a top-level structure (TLAS) of
//! instances, one build command for both, and bind group and layout creation
//! that take TLAS bindings beside the ordinary entries. They go to wgpu-core's
//! own ray-tracing API (`device_create_blas`, `device_create_tlas`,
//! `command_encoder_build_acceleration_structures`), so validation and errors
//! are wgpu's: an error reaches the device's error sink like any other call's.
//!
//! The device needs `WGPUNativeFeature_RayQuery` (wgpu's
//! `EXPERIMENTAL_RAY_QUERY`); requesting it turns wgpu's experimental features
//! on for that device (`conv::map_device_descriptor`).

#![allow(non_snake_case)] // parameter and field names follow the C header

use crate::conv::{self, map_bind_group_entry, map_bind_group_layout_entry};
use crate::native;
use crate::utils::{make_slice, string_view_into_label};
use crate::{follow_chain, handle_error, Context, ErrorSink, WGPUBindGroupImpl, WGPUBindGroupLayoutImpl};
use std::borrow::Cow;
use std::sync::Arc;
use std::thread;
use wgc::id;

pub struct WGPUBlasImpl {
    context: Arc<Context>,
    id: id::BlasId,
    /// The geometry sizes it was created with: a build names only buffers.
    sizes: Vec<wgt::BlasTriangleGeometrySizeDescriptor>,
}
impl Drop for WGPUBlasImpl {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.blas_drop(self.id);
        }
    }
}

pub struct WGPUTlasImpl {
    context: Arc<Context>,
    id: id::TlasId,
}
impl Drop for WGPUTlasImpl {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.tlas_drop(self.id);
        }
    }
}

pub type WGPUBlasExt = *const WGPUBlasImpl;
pub type WGPUTlasExt = *const WGPUTlasImpl;

/// One triangle geometry of a BLAS: what its builds will hold at most.
/// `indexFormat` `WGPUIndexFormat_Undefined` means no index buffer.
#[repr(C)]
pub struct WGPUBlasTriangleGeometryExt {
    pub vertexFormat: native::WGPUVertexFormat,
    pub vertexCount: u32,
    pub indexFormat: native::WGPUIndexFormat,
    pub indexCount: u32,
    pub opaque: native::WGPUBool,
}

/// The buffers of one geometry in a build, in the BLAS's geometry order.
/// `indexBuffer` is NULL for a geometry created without an index format.
#[repr(C)]
pub struct WGPUBlasBuildGeometryExt {
    pub vertexBuffer: native::WGPUBuffer,
    pub firstVertex: u32,
    pub vertexStride: u64,
    pub indexBuffer: native::WGPUBuffer,
    pub firstIndex: u32,
}

#[repr(C)]
pub struct WGPUBlasBuildExt {
    pub blas: WGPUBlasExt,
    pub geometries: *const WGPUBlasBuildGeometryExt,
    pub geometryCount: usize,
}

/// One TLAS instance: a BLAS, a row-major 3x4 transform, 24 bits the shader
/// reads back and an 8-bit visibility mask. A NULL `blas` leaves the slot empty.
#[repr(C)]
pub struct WGPUTlasInstanceExt {
    pub blas: WGPUBlasExt,
    pub transform: [f32; 12],
    pub custom: u32,
    pub mask: u32,
}

#[repr(C)]
pub struct WGPUTlasBuildExt {
    pub tlas: WGPUTlasExt,
    pub instances: *const WGPUTlasInstanceExt,
    pub instanceCount: usize,
}

/// A TLAS binding of a bind group layout: `var name: acceleration_structure`.
#[repr(C)]
pub struct WGPUTlasLayoutEntryExt {
    pub binding: u32,
    pub visibility: native::WGPUShaderStage,
    pub vertexReturn: native::WGPUBool,
}

#[repr(C)]
pub struct WGPUTlasBindingExt {
    pub binding: u32,
    pub tlas: WGPUTlasExt,
}

fn flags(bits: u32) -> wgt::AccelerationStructureFlags {
    wgt::AccelerationStructureFlags::from_bits_truncate(bits as u8)
}

/// A BLAS for `geometryCount` triangle geometries. `flags` are wgpu's
/// `AccelerationStructureFlags` bits (1 allow update, 2 allow compaction, 4
/// prefer fast trace, 8 prefer fast build, 16 low memory). Never NULL: a
/// refused creation reports through the device's error sink and gives an
/// invalid BLAS, as wgpu's other objects do.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBlasExt(
    device: native::WGPUDevice,
    geometries: *const WGPUBlasTriangleGeometryExt,
    geometryCount: usize,
    flagBits: u32,
    label: native::WGPUStringView,
) -> WGPUBlasExt {
    let (device_id, context, error_sink) = {
        let device = device.as_ref().expect("invalid device");
        (device.id, &device.context, &device.error_sink)
    };
    let sizes: Vec<wgt::BlasTriangleGeometrySizeDescriptor> = make_slice(geometries, geometryCount)
        .iter()
        .map(|g| {
            let indexed = g.indexFormat != native::WGPUIndexFormat_Undefined;
            wgt::BlasTriangleGeometrySizeDescriptor {
                vertex_format: conv::map_vertex_format(g.vertexFormat).expect("invalid vertex format"),
                vertex_count: g.vertexCount,
                index_format: conv::map_index_format(g.indexFormat).ok(),
                index_count: indexed.then_some(g.indexCount),
                flags: if g.opaque != 0 {
                    wgt::AccelerationStructureGeometryFlags::OPAQUE
                } else {
                    wgt::AccelerationStructureGeometryFlags::empty()
                },
            }
        })
        .collect();
    let desc = wgc::resource::BlasDescriptor {
        label: string_view_into_label(label),
        flags: flags(flagBits),
        update_mode: wgt::AccelerationStructureUpdateMode::Build,
    };
    let (id, _handle, error) = context.device_create_blas(
        device_id,
        &desc,
        wgt::BlasGeometrySizeDescriptors::Triangles { descriptors: sizes.clone() },
        None,
    );
    if let Some(cause) = error {
        handle_error(error_sink, cause, desc.label, "wgpuDeviceCreateBlasExt");
    }
    Arc::into_raw(Arc::new(WGPUBlasImpl { context: context.clone(), id, sizes }))
}

/// A TLAS of at most `maxInstances` instances.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateTlasExt(
    device: native::WGPUDevice,
    maxInstances: u32,
    flagBits: u32,
    label: native::WGPUStringView,
) -> WGPUTlasExt {
    let (device_id, context, error_sink) = {
        let device = device.as_ref().expect("invalid device");
        (device.id, &device.context, &device.error_sink)
    };
    let desc = wgc::resource::TlasDescriptor {
        label: string_view_into_label(label),
        max_instances: maxInstances,
        flags: flags(flagBits),
        update_mode: wgt::AccelerationStructureUpdateMode::Build,
    };
    let (id, error) = context.device_create_tlas(device_id, &desc, None);
    if let Some(cause) = error {
        handle_error(error_sink, cause, desc.label, "wgpuDeviceCreateTlasExt");
    }
    Arc::into_raw(Arc::new(WGPUTlasImpl { context: context.clone(), id }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBlasReleaseExt(blas: WGPUBlasExt) {
    if !blas.is_null() {
        Arc::decrement_strong_count(blas);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTlasReleaseExt(tlas: WGPUTlasExt) {
    if !tlas.is_null() {
        Arc::decrement_strong_count(tlas);
    }
}

/// Records the builds of `blasCount` BLASes and `tlasCount` TLASes (a TLAS is
/// built after the BLASes its instances name, in the same command or an
/// earlier one). Geometry buffers need wgpu's `BLAS_INPUT` usage
/// (`WGPUBufferUsage_BlasInputExt`).
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderBuildAccelerationStructuresExt(
    encoder: native::WGPUCommandEncoder,
    blasBuilds: *const WGPUBlasBuildExt,
    blasCount: usize,
    tlasBuilds: *const WGPUTlasBuildExt,
    tlasCount: usize,
) {
    let (encoder_id, context, error_sink): (id::CommandEncoderId, &Arc<Context>, &ErrorSink) = {
        let encoder = encoder.as_ref().expect("invalid command encoder");
        (encoder.id, &encoder.context, &encoder.error_sink)
    };
    let blas_iter = make_slice(blasBuilds, blasCount).iter().map(|build| {
        let blas = build.blas.as_ref().expect("invalid BLAS");
        let geometries = make_slice(build.geometries, build.geometryCount);
        assert_eq!(geometries.len(), blas.sizes.len(), "a BLAS build names every geometry it was created with");
        let triangles = geometries.iter().zip(blas.sizes.iter()).map(|(g, size)| {
            let index = g.indexBuffer.as_ref().map(|b| b.id);
            wgc::ray_tracing::BlasTriangleGeometry {
                size,
                vertex_buffer: g.vertexBuffer.as_ref().expect("invalid vertex buffer").id,
                index_buffer: index,
                transform_buffer: None,
                first_vertex: g.firstVertex,
                vertex_stride: g.vertexStride,
                first_index: index.map(|_| g.firstIndex),
                transform_buffer_offset: None,
            }
        });
        wgc::ray_tracing::BlasBuildEntry {
            blas_id: blas.id,
            geometries: wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(triangles)),
        }
    });
    let tlas_iter = make_slice(tlasBuilds, tlasCount).iter().map(|build| {
        let tlas = build.tlas.as_ref().expect("invalid TLAS");
        let instances = make_slice(build.instances, build.instanceCount).iter().map(|i| {
            i.blas.as_ref().map(|blas| wgc::ray_tracing::TlasInstance {
                blas_id: blas.id,
                transform: &i.transform,
                custom_data: i.custom,
                mask: i.mask as u8,
            })
        });
        wgc::ray_tracing::TlasPackage { tlas_id: tlas.id, instances: Box::new(instances), lowest_unmodified: 0 }
    });
    if let Err(cause) = context.command_encoder_build_acceleration_structures(encoder_id, blas_iter, tlas_iter) {
        handle_error(error_sink, cause, None, "wgpuCommandEncoderBuildAccelerationStructuresExt");
    }
}

/// `wgpuDeviceCreateBindGroupLayout` with `tlasCount` acceleration-structure
/// bindings beside the descriptor's entries.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroupLayoutTlasExt(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUBindGroupLayoutDescriptor>,
    tlasEntries: *const WGPUTlasLayoutEntryExt,
    tlasCount: usize,
) -> native::WGPUBindGroupLayout {
    let (device_id, context, error_sink) = {
        let device = device.as_ref().expect("invalid device");
        (device.id, &device.context, &device.error_sink)
    };
    let descriptor = descriptor.expect("invalid descriptor");
    let mut entries = make_slice(descriptor.entries, descriptor.entryCount)
        .iter()
        .map(|entry| {
            follow_chain!(map_bind_group_layout_entry((entry),
                WGPUSType_BindGroupLayoutEntryExtras => native::WGPUBindGroupLayoutEntryExtras)
            )
        })
        .collect::<Vec<_>>();
    for e in make_slice(tlasEntries, tlasCount) {
        entries.push(wgt::BindGroupLayoutEntry {
            binding: e.binding,
            visibility: wgt::ShaderStages::from_bits(e.visibility as u32).expect("invalid visibility for bind group layout entry"),
            ty: wgt::BindingType::AccelerationStructure { vertex_return: e.vertexReturn != 0 },
            count: None,
        });
    }
    let desc = wgc::binding_model::BindGroupLayoutDescriptor {
        label: string_view_into_label(descriptor.label),
        entries: Cow::Borrowed(&entries),
    };
    let (id, error) = context.device_create_bind_group_layout(device_id, &desc, None);
    if let Some(cause) = error {
        handle_error(error_sink, cause, desc.label, "wgpuDeviceCreateBindGroupLayoutTlasExt");
    }
    Arc::into_raw(Arc::new(WGPUBindGroupLayoutImpl { context: context.clone(), id }))
}

/// `wgpuDeviceCreateBindGroup` with `tlasCount` TLAS bindings beside the
/// descriptor's entries.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroupTlasExt(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUBindGroupDescriptor>,
    tlasEntries: *const WGPUTlasBindingExt,
    tlasCount: usize,
) -> native::WGPUBindGroup {
    let (device_id, context, error_sink) = {
        let device = device.as_ref().expect("invalid device");
        (device.id, &device.context, &device.error_sink)
    };
    let descriptor = descriptor.expect("invalid descriptor");
    let layout = descriptor.layout.as_ref().expect("invalid bind group layout for bind group descriptor").id;
    let mut entries = make_slice(descriptor.entries, descriptor.entryCount)
        .iter()
        .map(|entry| {
            follow_chain!(map_bind_group_entry((entry),
                WGPUSType_BindGroupEntryExtras => native::WGPUBindGroupEntryExtras)
            )
        })
        .collect::<Vec<_>>();
    for e in make_slice(tlasEntries, tlasCount) {
        entries.push(wgc::binding_model::BindGroupEntry {
            binding: e.binding,
            resource: wgc::binding_model::BindingResource::AccelerationStructure(e.tlas.as_ref().expect("invalid TLAS").id),
        });
    }
    let desc = wgc::binding_model::BindGroupDescriptor {
        label: string_view_into_label(descriptor.label),
        layout,
        entries: Cow::Borrowed(&entries),
    };
    let (id, error) = context.device_create_bind_group(device_id, &desc, None);
    if let Some(cause) = error {
        handle_error(error_sink, cause, desc.label, "wgpuDeviceCreateBindGroupTlasExt");
    }
    Arc::into_raw(Arc::new(WGPUBindGroupImpl { context: context.clone(), id }))
}
