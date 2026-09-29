use std::path::{Path, PathBuf};

fn require_linkable(dir: &str) {
    let root = Path::new(dir);
    if root.join("libllama.so").is_file() || root.join("libllama.dylib").is_file() {
        return;
    }
    panic!("libllama.so or libllama.dylib not found in {dir}");
}

fn prism_dir() -> String {
    if let Ok(dir) = std::env::var("PRISM_LLAMA_DIR")
        && !dir.is_empty()
    {
        return dir;
    }

    let rel = include_str!("prism-rel.txt").trim();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let primary = manifest.join(rel);
    if primary.is_dir() {
        return primary.to_string_lossy().into_owned();
    }

    let beside = manifest.join("..").join(rel);
    if beside.is_dir() {
        return beside.to_string_lossy().into_owned();
    }

    // Automatically download prebuilts if missing
    let script = manifest.join("scripts/fetch-prism.sh");
    if script.is_file() {
        println!("cargo:warning=Prism libraries not found, running scripts/fetch-prism.sh...");
        let status = std::process::Command::new("sh")
            .arg(&script)
            .current_dir(&manifest)
            .status();

        if let Ok(st) = status
            && st.success()
            && primary.is_dir()
        {
            return primary.to_string_lossy().into_owned();
        }
    }

    panic!(
        "Prism llama directory not found: {rel}\n\
         Run ./scripts/fetch-prism.sh to download prebuilt libraries,\n\
         or set PRISM_LLAMA_DIR to a directory with libllama.so or libllama.dylib."
    );
}

fn main() {
    let dir = prism_dir();
    require_linkable(&dir);

    let mtmd =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vendor/prism-llama.cpp/tools/mtmd");
    let mut builder = cc::Build::new();
    builder.file("c/src/shim.c").include("c/include");
    if mtmd.is_dir() {
        builder.include(&mtmd);
        println!("cargo:rerun-if-changed={}", mtmd.display());
    }
    builder.flag_if_supported("-O2").compile("jf_shim");

    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    if cfg!(unix) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    println!("cargo:rerun-if-env-changed=PRISM_LLAMA_DIR");
    println!("cargo:rerun-if-changed=prism-rel.txt");
    println!("cargo:rerun-if-changed=c/src/shim.c");
    println!("cargo:rerun-if-changed=c/include/shim.h");
    println!("cargo:rerun-if-changed=c/include/llama.h");
    println!("cargo:rerun-if-changed=c/include/mtmd.h");
    println!("cargo:rerun-if-changed=c/include/mtmd-helper.h");
}
