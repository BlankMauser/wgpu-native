fn main() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let ffi = format!("{root}/ffi");
    cc::Build::new()
        .file("executor.c")
        .include(&ffi)
        .include(format!("{ffi}/webgpu-headers"))
        .compile("executor");
    // The wgpu-native fork as a separate shared library (its own wgpu-core
    // copy), loaded like wgpu_mbt's dynamic mode would.
    // WGPU_NATIVE_DIR: the directory holding the fork's libwgpu_native.so.
    let lib = std::env::var("WGPU_NATIVE_DIR").unwrap_or(format!("{root}/target/debug"));
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-lib=dylib=wgpu_native");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rerun-if-changed=executor.c");
    println!("cargo:rerun-if-env-changed=WGPU_NATIVE_DIR");
}
