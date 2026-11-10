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

#[cfg(all(test, feature = "adopt-core"))]
mod tests {
    use super::*;
    use crate::adopt_core::{id_to_u64, wgpuAdapterAdoptDeviceExt, wgpuCreateInstanceFromCoreExt, wgpuDeviceAdoptBufferExt, wgpuInstanceAdoptAdapterExt};
    use std::ffi::c_void;
    use std::sync::Mutex;

    /// What the device's error sink saw (the tests run one device at a time).
    static ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

    unsafe extern "C" fn on_error(
        _device: *const native::WGPUDevice,
        _kind: native::WGPUErrorType,
        message: native::WGPUStringView,
        _u1: *mut c_void,
        _u2: *mut c_void,
    ) {
        let text = if message.data.is_null() {
            String::new()
        } else {
            String::from_utf8_lossy(std::slice::from_raw_parts(message.data.cast::<u8>(), message.length)).into_owned()
        };
        ERRORS.lock().unwrap().push(text);
    }

    unsafe extern "C" fn on_map(status: native::WGPUMapAsyncStatus, _message: native::WGPUStringView, u1: *mut c_void, _u2: *mut c_void) {
        *u1.cast::<u32>() = status as u32;
    }

    fn view(s: &str) -> native::WGPUStringView {
        native::WGPUStringView { data: s.as_ptr().cast(), length: s.len() }
    }

    fn stage(what: &str) -> Result<(), String> {
        let errors = std::mem::take(&mut *ERRORS.lock().unwrap());
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("{what}: {}", errors.join(" | ")))
        }
    }

    /// Two rays down +z: one over the triangle at the origin, one 10 units to
    /// the side. `out[i]` is the committed intersection's kind (1: a triangle),
    /// `out[2 + i]` the instance's custom data. The `ray_query` is declared
    /// outside the loop: inside one, naga 30.0.1 writes a zero value of the
    /// ray query on every iteration. That panics its HLSL writer (DX12,
    /// `back/hlsl/conv.rs` `hlsl_type_id`), and on llvmpipe the pipeline
    /// is refused and wgpu reports the device lost (both seen 2026-11-10;
    /// the NVIDIA Vulkan driver accepted it).
    const RAYS: &str = "enable wgpu_ray_query;
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn cs_rays() {
  var rq: ray_query;
  for (var i = 0u; i < 2u; i++) {
    let origin = vec3<f32>(f32(i) * 10.0, 0.0, -1.0);
    rayQueryInitialize(&rq, scene, RayDesc(0u, 0xffu, 0.01, 100.0, origin, vec3<f32>(0.0, 0.0, 1.0)));
    while (rayQueryProceed(&rq)) {}
    let hit = rayQueryGetCommittedIntersection(&rq);
    out[i] = hit.kind;
    out[2u + i] = hit.instance_custom_data;
  }
}
";

    /// The whole path on one adapter, through the C calls a binding makes: a
    /// BLAS of one triangle, a TLAS of one instance, their build, a compute
    /// pass that traces two rays into a buffer the host made (adopted with
    /// `wgpuDeviceAdoptBufferExt`), and the buffer read back by the host.
    unsafe fn trace(global: &Arc<Context>, adapter_id: id::AdapterId) -> Result<[u32; 4], String> {
        let desc = wgt::DeviceDescriptor {
            required_features: wgt::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: wgt::Limits::default().using_acceleration_structure_values(global.adapter_limits(adapter_id)),
            experimental_features: wgt::ExperimentalFeatures::enabled(),
            ..Default::default()
        };
        let (device_id, queue_id) =
            global.adapter_request_device(adapter_id, &desc, None, None).map_err(|e| format!("request_device: {e}"))?;
        ERRORS.lock().unwrap().clear();
        let instance = wgpuCreateInstanceFromCoreExt(Arc::as_ptr(global).cast());
        let adapter = wgpuInstanceAdoptAdapterExt(instance, id_to_u64(adapter_id));
        let mut dd: native::WGPUDeviceDescriptor = std::mem::zeroed();
        dd.uncapturedErrorCallbackInfo.callback = Some(on_error);
        let device = wgpuAdapterAdoptDeviceExt(adapter, id_to_u64(device_id), id_to_u64(queue_id), Some(&dd));
        if device.is_null() {
            return Err("the device was not adopted".into());
        }
        let queue = crate::wgpuDeviceGetQueue(device);

        // One triangle around the origin in the z = 0 plane.
        let vertices: [f32; 9] = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0];
        let mut bd: native::WGPUBufferDescriptor = std::mem::zeroed();
        bd.usage = 0x400 | native::WGPUBufferUsage_CopyDst as native::WGPUBufferUsage; // WGPUBufferUsage_BlasInputExt
        bd.size = 36;
        let vertex_buffer = crate::wgpuDeviceCreateBuffer(device, Some(&bd));
        crate::wgpuQueueWriteBuffer(queue, vertex_buffer, 0, vertices.as_ptr().cast(), 36);
        stage("the vertex buffer")?;

        let geometry = WGPUBlasTriangleGeometryExt {
            vertexFormat: native::WGPUVertexFormat_Float32x3,
            vertexCount: 3,
            indexFormat: native::WGPUIndexFormat_Undefined,
            indexCount: 0,
            opaque: 1,
        };
        let blas = wgpuDeviceCreateBlasExt(device, &geometry, 1, 0, view("triangle"));
        let tlas = wgpuDeviceCreateTlasExt(device, 1, 0, view("world"));
        stage("creating the structures")?;

        let build_geometry = WGPUBlasBuildGeometryExt {
            vertexBuffer: vertex_buffer,
            firstVertex: 0,
            vertexStride: 12,
            indexBuffer: std::ptr::null(),
            firstIndex: 0,
        };
        let blas_build = WGPUBlasBuildExt { blas, geometries: &build_geometry, geometryCount: 1 };
        let instance_entry = WGPUTlasInstanceExt {
            blas,
            transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            custom: 7,
            mask: 0xff,
        };
        let tlas_build = WGPUTlasBuildExt { tlas, instances: &instance_entry, instanceCount: 1 };
        let encoder = crate::wgpuDeviceCreateCommandEncoder(device, None);
        wgpuCommandEncoderBuildAccelerationStructuresExt(encoder, &blas_build, 1, &tlas_build, 1);
        let commands = crate::wgpuCommandEncoderFinish(encoder, None);
        crate::wgpuQueueSubmit(queue, 1, &commands);
        crate::wgpuCommandBufferRelease(commands);
        crate::wgpuCommandEncoderRelease(encoder);
        stage("building the structures")?;

        // The output is the host's buffer.
        let (out_id, error) = global.device_create_buffer(
            device_id,
            &wgc::resource::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgt::BufferUsages::STORAGE | wgt::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
            None,
        );
        if let Some(e) = error {
            return Err(format!("the host's buffer: {e}"));
        }
        let mut od: native::WGPUBufferDescriptor = std::mem::zeroed();
        od.usage = (native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc) as _;
        od.size = 16;
        let out = wgpuDeviceAdoptBufferExt(device, id_to_u64(out_id), Some(&od));
        if out.is_null() {
            return Err("the host's buffer was not adopted".into());
        }

        let mut wgsl: native::WGPUShaderSourceWGSL = std::mem::zeroed();
        wgsl.chain.sType = native::WGPUSType_ShaderSourceWGSL;
        wgsl.code = view(RAYS);
        let mut sd: native::WGPUShaderModuleDescriptor = std::mem::zeroed();
        sd.nextInChain = std::ptr::addr_of!(wgsl.chain).cast_mut();
        let module = crate::wgpuDeviceCreateShaderModule(device, Some(&sd));
        stage("the shader module")?;

        let mut entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
        entry.binding = 1;
        entry.visibility = native::WGPUShaderStage_Compute as _;
        entry.buffer.type_ = native::WGPUBufferBindingType_Storage;
        let mut ld: native::WGPUBindGroupLayoutDescriptor = std::mem::zeroed();
        ld.entryCount = 1;
        ld.entries = &entry;
        let tlas_layout = WGPUTlasLayoutEntryExt { binding: 0, visibility: native::WGPUShaderStage_Compute as _, vertexReturn: 0 };
        let layout = wgpuDeviceCreateBindGroupLayoutTlasExt(device, Some(&ld), &tlas_layout, 1);
        stage("the bind group layout")?;

        let mut pd: native::WGPUPipelineLayoutDescriptor = std::mem::zeroed();
        pd.bindGroupLayoutCount = 1;
        pd.bindGroupLayouts = &layout;
        let pipeline_layout = crate::wgpuDeviceCreatePipelineLayout(device, Some(&pd));
        let mut cd: native::WGPUComputePipelineDescriptor = std::mem::zeroed();
        cd.layout = pipeline_layout;
        cd.compute.module = module;
        cd.compute.entryPoint = view("cs_rays");
        let pipeline = crate::wgpuDeviceCreateComputePipeline(device, Some(&cd));
        stage("the ray-query pipeline")?;

        let mut be: native::WGPUBindGroupEntry = std::mem::zeroed();
        be.binding = 1;
        be.buffer = out;
        be.size = 16;
        let mut gd: native::WGPUBindGroupDescriptor = std::mem::zeroed();
        gd.layout = layout;
        gd.entryCount = 1;
        gd.entries = &be;
        let tlas_binding = WGPUTlasBindingExt { binding: 0, tlas };
        let group = wgpuDeviceCreateBindGroupTlasExt(device, Some(&gd), &tlas_binding, 1);
        stage("the bind group")?;

        let mut rd: native::WGPUBufferDescriptor = std::mem::zeroed();
        rd.usage = (native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst) as _;
        rd.size = 16;
        let staging = crate::wgpuDeviceCreateBuffer(device, Some(&rd));
        let encoder = crate::wgpuDeviceCreateCommandEncoder(device, None);
        let pass = crate::wgpuCommandEncoderBeginComputePass(encoder, None);
        crate::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        crate::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
        crate::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        crate::wgpuComputePassEncoderEnd(pass);
        crate::wgpuComputePassEncoderRelease(pass);
        crate::wgpuCommandEncoderCopyBufferToBuffer(encoder, out, 0, staging, 0, 16);
        let commands = crate::wgpuCommandEncoderFinish(encoder, None);
        crate::wgpuQueueSubmit(queue, 1, &commands);
        crate::wgpuCommandBufferRelease(commands);
        crate::wgpuCommandEncoderRelease(encoder);
        stage("tracing the rays")?;

        let mut status: u32 = 0;
        let mut info: native::WGPUBufferMapCallbackInfo = std::mem::zeroed();
        info.callback = Some(on_map);
        info.userdata1 = std::ptr::addr_of_mut!(status).cast();
        crate::wgpuBufferMapAsync(staging, native::WGPUMapMode_Read as _, 0, 16, info);
        crate::wgpuDevicePoll(device, true, None);
        stage("reading the result")?;
        if status != native::WGPUMapAsyncStatus_Success as u32 {
            return Err(format!("the result was not mapped (status {status})"));
        }
        let bytes = std::slice::from_raw_parts(crate::wgpuBufferGetConstMappedRange(staging, 0, 16), 16);
        let mut words = [0u32; 4];
        for (i, w) in words.iter_mut().enumerate() {
            *w = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        }
        crate::wgpuBufferUnmap(staging);

        crate::wgpuBufferRelease(staging);
        crate::wgpuBindGroupRelease(group);
        crate::wgpuComputePipelineRelease(pipeline);
        crate::wgpuPipelineLayoutRelease(pipeline_layout);
        crate::wgpuBindGroupLayoutRelease(layout);
        crate::wgpuShaderModuleRelease(module);
        crate::wgpuBufferRelease(out);
        global.buffer_drop(out_id);
        wgpuTlasReleaseExt(tlas);
        wgpuBlasReleaseExt(blas);
        crate::wgpuBufferRelease(vertex_buffer);
        crate::wgpuQueueRelease(queue);
        crate::wgpuDeviceRelease(device);
        crate::wgpuAdapterRelease(adapter);
        crate::wgpuInstanceRelease(instance);
        Ok(words)
    }

    /// Every adapter with ray queries traces the two rays: the first hits the
    /// triangle (kind 1, the instance's custom data 7), the second misses.
    /// Skipped, with a message, on a machine with no such adapter.
    #[test]
    fn rays_are_traced_on_every_adapter_with_ray_queries() {
        let global = Arc::new(Context::new("ray-query test", wgt::InstanceDescriptor::new_without_display_handle(), None));
        let mut traced = 0;
        let mut failures = Vec::new();
        for adapter_id in global.enumerate_adapters(wgt::Backends::all(), false) {
            let info = global.adapter_get_info(adapter_id);
            let name = format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);
            if !global.adapter_features(adapter_id).contains(wgt::Features::EXPERIMENTAL_RAY_QUERY) {
                eprintln!("ray queries: {name}: no ray queries");
                continue;
            }
            match unsafe { trace(&global, adapter_id) } {
                Ok(words) => {
                    eprintln!("ray queries: {name}: traced {words:?}");
                    if words != [1, 0, 7, 0] {
                        failures.push(format!("{name}: traced {words:?}, expected [1, 0, 7, 0]"));
                    }
                    traced += 1;
                }
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        if traced == 0 {
            eprintln!("ray queries: no adapter with ray queries here: nothing verified");
        }
    }
}
