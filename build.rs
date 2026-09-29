use std::path::{Path, PathBuf};

fn prism_dir() -> String {
    if let Ok(dir) = std::env::var("PRISM_LLAMA_DIR") {
        if !dir.is_empty() {
            return dir;
        }
    }
    let rel = include_str!("prism-rel.txt").trim();
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(rel)
        .to_string_lossy()
        .into_owned()
}

fn main() {
    let dir = prism_dir();
    if !Path::new(&dir).is_dir() {
        panic!("prism llama directory not found: {dir} (set PRISM_LLAMA_DIR)");
    }

    let mtmd = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vendor/prism-llama.cpp/tools/mtmd");
    let mut builder = cc::Build::new();
    builder
        .file("c/src/shim.c")
        .include("c/include");
    if mtmd.is_dir() {
        builder.include(&mtmd);
        println!("cargo:rerun-if-changed={}", mtmd.display());
    }
    builder
        .flag_if_supported("-O2")
        .compile("jf_shim");

    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    println!("cargo:rerun-if-env-changed=PRISM_LLAMA_DIR");
    println!("cargo:rerun-if-changed=prism-rel.txt");
    println!("cargo:rerun-if-changed=c/src/shim.c");
    println!("cargo:rerun-if-changed=c/include/shim.h");
    println!("cargo:rerun-if-changed=c/include/llama.h");
    println!("cargo:rerun-if-changed=c/include/mtmd.h");
    println!("cargo:rerun-if-changed=c/include/mtmd-helper.h");
}
