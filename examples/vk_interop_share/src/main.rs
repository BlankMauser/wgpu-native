//! Rust host on the wgpu 30 crate owns the device and a texture; a C stand-in
//! for LunarFenrir's executor renders into that texture through wgpu-native
//! (BlankMauser fork, wgpu 30, Vulkan interop extension); the host reads it
//! back and checks pixels. One VkDevice, one VkQueue, one VkImage.

use ash::vk::Handle;
use std::ffi::{c_char, c_void};
use wgpu::hal::api::Vulkan;

#[repr(C)]
struct VulkanDeviceHandles {
    get_instance_proc_addr: *const c_void,
    instance: u64,
    instance_api_version: u32,
    instance_extension_count: usize,
    instance_extensions: *const *const c_char,
    physical_device: u64,
    device: u64,
    device_extension_count: usize,
    device_extensions: *const *const c_char,
    queue_family_index: u32,
    queue_index: u32,
}

extern "C" {
    fn exec_init(h: *const VulkanDeviceHandles) -> i32;
    fn exec_import(image: u64, w: u32, h: u32) -> i32;
    fn exec_frame(frame: i32);
    fn exec_shutdown();
    fn exec_errors() -> i32;
}

const W: u32 = 64;
const H: u32 = 64;

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (W * H * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    // Hand back in the agreed state (TextureBinding == RESOURCE).
    handoff(&mut enc, tex);
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buf.slice(..).get_mapped_range().expect("mapped").to_vec();
    data
}

fn handoff(enc: &mut wgpu::CommandEncoder, tex: &wgpu::Texture) {
    enc.transition_resources(
        std::iter::empty(),
        std::iter::once(wgpu::TextureTransition {
            texture: tex,
            selector: None,
            state: if std::env::var_os("HANDOFF_RT").is_some() {
                wgpu::TextureUses::COLOR_TARGET
            } else {
                wgpu::TextureUses::RESOURCE
            },
        }),
    );
}

fn px(d: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [d[i], d[i + 1], d[i + 2], d[i + 3]]
}

fn main() {
    env_logger::init();
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::VULKAN;
    desc.flags = wgpu::InstanceFlags::from_build_config().with_env();
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    println!("adapter: {:?}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("host wgpu error: {e}")));

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shared"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    // Host side, before the first handoff: (1) write the texture once, so the
    // host's wgpu-core init tracker marks it initialized (otherwise the host's
    // first copy lazily zero-clears it, discarding the executor's pixels);
    // (2) move the host tracker to the agreed handoff state.
    let mut enc = device.create_command_encoder(&Default::default());
    if std::env::var_os("SKIP_HOST_INIT").is_none() {
        let view = tex.create_view(&Default::default());
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("host-init"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    handoff(&mut enc, &tex);
    queue.submit([enc.finish()]);

    // Raw handles of the host's device (same pattern as moon-river's Rive bridge).
    let (inst_exts, dev_exts, mut handles) = {
        let hal = unsafe { device.as_hal::<Vulkan>() }.expect("Vulkan device");
        let inst = hal.shared_instance();
        let inst_exts: Vec<*const c_char> = inst.extensions().iter().map(|e| e.as_ptr()).collect();
        let dev_exts: Vec<*const c_char> =
            hal.enabled_device_extensions().iter().map(|e| e.as_ptr()).collect();
        let h = VulkanDeviceHandles {
            get_instance_proc_addr: inst.entry().static_fn().get_instance_proc_addr as *const c_void,
            instance: inst.raw_instance().handle().as_raw(),
            instance_api_version: inst.instance_api_version(),
            instance_extension_count: inst_exts.len(),
            instance_extensions: std::ptr::null(),
            physical_device: hal.raw_physical_device().as_raw(),
            device: hal.raw_device().handle().as_raw(),
            device_extension_count: dev_exts.len(),
            device_extensions: std::ptr::null(),
            queue_family_index: hal.queue_family_index(),
            queue_index: hal.queue_index(),
        };
        (inst_exts, dev_exts, h)
    };
    handles.instance_extensions = inst_exts.as_ptr();
    handles.device_extensions = dev_exts.as_ptr();
    let image = unsafe { tex.as_hal::<Vulkan>() }.map(|t| unsafe { t.raw_handle() }.as_raw()).unwrap();
    println!("VkDevice 0x{:x}, VkImage 0x{:x}", handles.device, image);

    assert_eq!(unsafe { exec_init(&handles) }, 0, "exec_init");
    assert_eq!(unsafe { exec_import(image, W, H) }, 0, "exec_import");

    // Frame 0: executor clears blue and draws an orange triangle (top-left half).
    unsafe { exec_frame(0) };
    let d = readback(&device, &queue, &tex);
    let orange = d.chunks(4).filter(|p| p[0] == 255 && (127..=128).contains(&p[1]) && p[2] == 0).count();
    let blue = d.chunks(4).filter(|p| p == &[0, 0, 255, 255]).count();
    println!("frame 0: px(4,4)={:?} px(60,60)={:?} orange={orange} blue={blue}", px(&d, 4, 4), px(&d, 60, 60));
    assert_eq!(px(&d, 60, 60), [0, 0, 255, 255]);
    assert_eq!(&px(&d, 4, 4)[..1], &[255]);
    assert_eq!(orange + blue, (W * H) as usize);
    assert!(orange > 1900 && orange < 2200);

    // Frame 1: executor clears green; the texture went host -> executor -> host twice.
    unsafe { exec_frame(1) };
    let d = readback(&device, &queue, &tex);
    let green = d.chunks(4).filter(|p| p == &[0, 255, 0, 255]).count();
    println!("frame 1: green={green}/{}", W * H);
    assert_eq!(green, (W * H) as usize);

    assert_eq!(unsafe { exec_errors() }, 0, "wgpu-native logged errors");
    unsafe { exec_shutdown() };
    drop(tex);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    println!("OK: one VkDevice, Rust wgpu 30 texture rendered by wgpu-native, pixels verified");
}
