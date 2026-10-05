use std::ffi::{c_char, c_float};

/// Opaque handle owned by `jf_load` / `jf_free`.
#[repr(C)]
pub struct JfCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub unsafe fn jf_load_backends(dir: *const c_char);
    pub unsafe fn jf_load(
        path: *const c_char,
        lora: *const c_char,
        n_ctx: i32,
        n_threads: i32,
        n_gpu_layers: i32,
        out: *mut *mut JfCtx,
    ) -> i32;
    pub unsafe fn jf_free(h: *mut JfCtx);
    pub unsafe fn jf_n_vocab(h: *mut JfCtx) -> i32;
    pub unsafe fn jf_n_seq_max(h: *mut JfCtx) -> i32;
    pub unsafe fn jf_n_ctx_seq(h: *mut JfCtx) -> i32;
    pub unsafe fn jf_tokenize(
        h: *mut JfCtx,
        text: *const c_char,
        add_special: i32,
        out: *mut i32,
        cap: i32,
    ) -> i32;
    pub unsafe fn jf_token_piece(h: *mut JfCtx, id: i32, buf: *mut c_char, cap: i32) -> i32;
    pub unsafe fn jf_decode(h: *mut JfCtx, tokens: *const i32, n: i32) -> i32;
    pub unsafe fn jf_decode_many(
        h: *mut JfCtx,
        toks: *const *const i32,
        lens: *const i32,
        n_seq: i32,
        out_logits: *mut c_float,
    ) -> i32;
    pub unsafe fn jf_logits(h: *mut JfCtx) -> *const c_float;
    pub unsafe fn jf_last_keep(
        h: *mut JfCtx,
        nseq_out: *mut i32,
        keep_out: *mut i32,
        total_out: *mut i32,
        cap: i32,
    );
    pub unsafe fn jf_vision_load(h: *mut JfCtx, mmproj: *const c_char, n_threads: i32) -> i32;
    pub unsafe fn jf_media_marker(h: *mut JfCtx) -> *const c_char;
    pub unsafe fn jf_decode_rgb(
        h: *mut JfCtx,
        prompt: *const c_char,
        rgb: *const u8,
        nx: u32,
        ny: u32,
        n_tokens_out: *mut i32,
    ) -> i32;
}
