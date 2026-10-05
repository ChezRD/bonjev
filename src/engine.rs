use crate::ffi::*;
use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, CString, OsStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Once;

pub struct Engine {
    ctx: *mut JfCtx,
    pub n_vocab: usize,
    pub n_seq: usize,
    media_marker: Option<String>,
}

// SAFETY: `ctx` is owned by this value and is not shared. The type is moved between threads
// only while no borrow of it is live.
unsafe impl Send for Engine {}

/// Why an engine call failed. The server uses the kind, not the message text,
/// to decide whether retrying with a smaller context can help.
#[derive(Debug)]
pub struct EngineFault {
    pub kind: EngineFaultKind,
    detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineFaultKind {
    /// Loading the weights failed: out of memory or a bad file, which the FFI
    /// boundary does not distinguish. Retried by `load_from_ctx`, which halves
    /// the context. A vision-projector failure is [`Self::Invalid`], not this.
    Load,
    /// A text decode failed.
    Decode,
    /// Logits were not produced after a decode.
    Logits,
    /// The prompt is longer than the per-sequence context window.
    TooLong,
    /// Decoding an image failed; shrinking the context does not repair it.
    Image,
    /// A load failure that a smaller context cannot fix: a bad adapter, bad
    /// arguments, an empty vocabulary, or a bad vision projector.
    Invalid,
}

// `jf_load` status codes, mirroring the enum in `c/include/shim.h`.
const JF_LOAD_MODEL: i32 = 1;
const JF_LOAD_CONTEXT: i32 = 2;
const JF_LOAD_LORA: i32 = 3;
const JF_LOAD_ARGS: i32 = 4;

impl EngineFaultKind {
    /// Whether the server's unload-and-shrink retry can help. Only inference
    /// failures qualify: a load failure is handled inside
    /// [`LoadSpec::load_from_ctx`], which already halves the context, and a file
    /// that cannot load at all must not be retried.
    pub fn recoverable(self) -> bool {
        matches!(self, Self::Decode | Self::Logits)
    }
}

impl EngineFault {
    fn load(detail: impl Into<String>) -> Self {
        Self { kind: EngineFaultKind::Load, detail: detail.into() }
    }
    fn decode(detail: impl Into<String>) -> Self {
        Self { kind: EngineFaultKind::Decode, detail: detail.into() }
    }
    fn logits(detail: impl Into<String>) -> Self {
        Self { kind: EngineFaultKind::Logits, detail: detail.into() }
    }
    fn too_long(detail: impl Into<String>) -> Self {
        Self { kind: EngineFaultKind::TooLong, detail: detail.into() }
    }
    fn image(detail: impl Into<String>) -> Self {
        Self { kind: EngineFaultKind::Image, detail: detail.into() }
    }
    fn invalid(detail: impl Into<String>) -> Self {
        Self { kind: EngineFaultKind::Invalid, detail: detail.into() }
    }

    /// Map a `jf_load` status code to a fault. Only a weights/context failure can
    /// be out of memory; an adapter, argument, or internal failure cannot.
    fn from_load(status: i32, path: &Path) -> Self {
        let path = path.display();
        match status {
            JF_LOAD_MODEL | JF_LOAD_CONTEXT => Self::load(format!("llama: failed to load {path}")),
            JF_LOAD_LORA => Self::invalid(format!("llama: failed to load the adapter for {path}")),
            JF_LOAD_ARGS => Self::invalid(format!("llama: invalid load arguments for {path}")),
            _ => Self::invalid(format!("llama: failed to load {path} (internal)")),
        }
    }
}

impl fmt::Display for EngineFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for EngineFault {}

/// Mirrors `JF_SEQ_MAX` in `c/src/shim.c`: the shim caps sequences at 16.
pub const JF_SEQ_MAX: usize = 16;

fn prism_marker() -> &'static str {
    include_str!("../prism-rel.txt")
        .trim()
        .trim_start_matches("./")
        .trim_start_matches("../")
}

fn lib_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("PRISM_LLAMA_DIR")
        && !dir.is_empty()
    {
        let path = PathBuf::from(&dir);
        if !path.is_dir() {
            bail!("PRISM_LLAMA_DIR is not a directory: {dir}");
        }
        return Ok(path);
    }
    let marker = prism_marker();
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        roots.push(parent.to_path_buf());
    }
    for root in roots {
        let mut dir = root;
        loop {
            let candidate = dir.join(marker);
            if candidate.is_dir() {
                return Ok(candidate);
            }
            if !dir.pop() {
                break;
            }
        }
    }
    bail!("prism llama directory not found; set PRISM_LLAMA_DIR")
}

fn init_backends() -> Result<()> {
    static ONCE: Once = Once::new();
    let dir = lib_dir()?;
    let c_dir = c_string_from_os(dir.as_os_str(), "PRISM_LLAMA_DIR")?;
    // SAFETY: c_dir is NUL-terminated and outlives the call. ggml_backend_load_all_from_path
    // loads the backend plugins during the call and does not retain the pointer.
    ONCE.call_once(|| unsafe { jf_load_backends(c_dir.as_ptr()) });
    Ok(())
}

fn fit_i32(what: &str, value: u32) -> Result<i32> {
    i32::try_from(value).map_err(|_| anyhow!("{what}={value} does not fit into i32"))
}

fn fits_i32(len: usize) -> bool {
    i32::try_from(len).is_ok()
}

fn len_i32(len: usize) -> i32 {
    i32::try_from(len).expect("length checked to fit i32")
}

fn count_usize(n: i32) -> usize {
    usize::try_from(n).expect("count checked non-negative")
}

fn c_path(path: &Path) -> Result<CString> {
    c_string_from_os(path.as_os_str(), "path")
}

/// Path bytes for the C API. Unix passes the raw OS bytes; other platforms use
/// the UTF-8 form of the path.
#[cfg(unix)]
fn os_bytes(value: &OsStr) -> std::borrow::Cow<'_, [u8]> {
    use std::os::unix::ffi::OsStrExt;
    std::borrow::Cow::Borrowed(value.as_bytes())
}

#[cfg(not(unix))]
fn os_bytes(value: &OsStr) -> std::borrow::Cow<'_, [u8]> {
    std::borrow::Cow::Owned(value.to_string_lossy().into_owned().into_bytes())
}

fn c_string_from_os(value: &OsStr, what: &str) -> Result<CString> {
    CString::new(os_bytes(value).as_ref()).map_err(|_| anyhow!("{what} contains a NUL byte"))
}

/// True when a load failure could be out of memory: only a weights/context
/// failure qualifies. Adapter, argument, and internal failures are not retried.
fn is_memory_load(err: &anyhow::Error) -> bool {
    err.downcast_ref::<EngineFault>()
        .is_some_and(|fault| fault.kind == EngineFaultKind::Load)
}

/// Smallest context tried when VRAM is tight.
pub const MIN_VRAM_CTX: u32 = 4096;

/// Parameters needed to (re)load the GGUF after an explicit unload.
pub struct LoadSpec {
    pub model_path: PathBuf,
    pub ctx: u32,
    pub threads: u32,
    pub n_gpu_layers: i32,
    pub mmproj: Option<PathBuf>,
    /// Resolved adapter list (`path[,scale][;...]`), passed to `jf_load`.
    pub lora: Option<String>,
}

impl LoadSpec {
    /// Load weights and, when set, the vision projector. A load failure that
    /// looks like an out-of-memory condition is retried with a smaller context
    /// down to `MIN_VRAM_CTX`; any other failure is returned as is.
    pub fn load_from_ctx(&self, start_ctx: u32) -> Result<(Engine, u32)> {
        let mut try_ctx = start_ctx.max(MIN_VRAM_CTX);
        loop {
            match self.load_at(try_ctx) {
                Ok(engine) => {
                    if try_ctx < start_ctx {
                        eprintln!(
                            "[bonjev] loaded with ctx={try_ctx} (requested {start_ctx}) after a smaller retry"
                        );
                    }
                    return Ok((engine, try_ctx));
                }
                Err(err) if is_memory_load(&err) => {
                    eprintln!("[bonjev] ctx={try_ctx} load failed: {err}; retrying smaller");
                    if try_ctx == MIN_VRAM_CTX {
                        bail!(
                            "could not load {} with ctx down to {MIN_VRAM_CTX}: {err}",
                            self.model_path.display()
                        );
                    }
                    try_ctx = (try_ctx / 2).max(MIN_VRAM_CTX);
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn load_at(&self, ctx: u32) -> Result<Engine> {
        let mut engine = Engine::load(
            &self.model_path,
            self.lora.as_deref(),
            ctx,
            self.threads,
            self.n_gpu_layers,
        )?;
        if let Some(mmproj) = &self.mmproj {
            engine.load_vision(mmproj, self.threads)?;
        }
        Ok(engine)
    }
}

impl Engine {
    pub fn load(
        path: &Path,
        lora: Option<&str>,
        n_ctx: u32,
        n_threads: u32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let n_ctx = fit_i32("ctx", n_ctx)?;
        let n_threads = fit_i32("threads", n_threads)?;
        if n_ctx <= 0 {
            bail!("ctx must be positive");
        }
        init_backends()?;
        let c_path = c_path(path)?;
        let c_lora = lora
            .map(|value| CString::new(value).map_err(|_| anyhow!("LoRA spec contains a NUL byte")))
            .transpose()?;
        let lora_ptr = c_lora
            .as_ref()
            .map_or(std::ptr::null(), |value| value.as_ptr());
        let mut handle: *mut JfCtx = std::ptr::null_mut();
        // SAFETY: c_path and c_lora are NUL-terminated and outlive the call; jf_load
        // writes the handle to `handle` on success and leaves it null on failure.
        let status = unsafe {
            jf_load(
                c_path.as_ptr(),
                lora_ptr,
                n_ctx,
                n_threads,
                n_gpu_layers,
                &mut handle,
            )
        };
        if status != 0 || handle.is_null() {
            bail!(EngineFault::from_load(status, path));
        }
        let ctx = handle;
        // SAFETY: ctx is the non-null handle just returned by jf_load.
        let n_vocab = unsafe { jf_n_vocab(ctx) };
        if n_vocab <= 0 {
            // SAFETY: ctx came from jf_load and has not been freed.
            unsafe { jf_free(ctx) };
            bail!(EngineFault::invalid("model vocabulary is empty"));
        }
        // SAFETY: ctx is the live handle from jf_load.
        let n_seq = unsafe { jf_n_seq_max(ctx) }.max(1);
        Ok(Self {
            ctx,
            n_vocab: count_usize(n_vocab),
            n_seq: count_usize(n_seq),
            media_marker: None,
        })
    }

    pub fn load_vision(&mut self, path: &Path, n_threads: u32) -> Result<()> {
        let c_path = c_path(path)?;
        let threads = fit_i32("threads", n_threads)?;
        // SAFETY: c_path outlives the call. The projector stays owned by the context.
        let rc = unsafe { jf_vision_load(self.ctx, c_path.as_ptr(), threads) };
        if rc != 0 {
            bail!(EngineFault::invalid(format!(
                "failed to load vision projector {}",
                path.display()
            )));
        }
        // SAFETY: vision load succeeded, so the context owns a projector and its marker.
        let marker = unsafe { jf_media_marker(self.ctx) };
        if marker.is_null() {
            bail!("vision projector loaded without a media marker");
        }
        // SAFETY: marker is a non-null C string owned by the projector until the next load or drop.
        let text = unsafe { CStr::from_ptr(marker) }
            .to_str()
            .map_err(|_| anyhow!("media marker is not UTF-8"))?
            .to_string();
        self.media_marker = Some(text);
        Ok(())
    }

    pub fn media_marker(&self) -> Option<&str> {
        self.media_marker.as_deref()
    }

    pub fn decode_rgb(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<usize> {
        let expect = usize::try_from(width)
            .expect("image width fits usize")
            .checked_mul(usize::try_from(height).expect("image height fits usize"))
            .and_then(|n| n.checked_mul(3));
        if expect != Some(rgb.len()) {
            bail!("rgb length {} does not match {width}x{height}", rgb.len());
        }
        let c_prompt = CString::new(prompt)?;
        let mut n_tokens = 0i32;
        // SAFETY: prompt and rgb live for the call. The shim copies the pixels into the bitmap
        // before it returns.
        let rc = unsafe {
            jf_decode_rgb(
                self.ctx,
                c_prompt.as_ptr(),
                rgb.as_ptr(),
                width,
                height,
                &mut n_tokens,
            )
        };
        if rc != 0 {
            bail!(EngineFault::image(format!("image decode failed rc={rc}")));
        }
        if n_tokens < 0 {
            bail!("image decode reported a negative token count");
        }
        Ok(count_usize(n_tokens))
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let c_text = CString::new(text)?;
        let cap = text.len().saturating_add(16);
        if !fits_i32(cap) {
            bail!("tokenize input too long");
        }
        let mut buf = vec![0; cap];
        // SAFETY: c_text and buf live for the call. cap fits i32, so the shim writes
        // at most that many tokens into buf.
        let n = unsafe {
            jf_tokenize(
                self.ctx,
                c_text.as_ptr(),
                i32::from(add_special),
                buf.as_mut_ptr(),
                len_i32(buf.len()),
            )
        };
        if n == i32::MIN {
            bail!("tokenize overflow");
        }
        if n < 0 {
            let need = usize::try_from(-n).map_err(|_| anyhow!("tokenize overflow"))?;
            if !fits_i32(need) {
                bail!("tokenize overflow");
            }
            buf.resize(need, 0);
            // SAFETY: same as the first tokenize call; buf now holds `need` slots, which fits i32.
            let n2 = unsafe {
                jf_tokenize(
                    self.ctx,
                    c_text.as_ptr(),
                    i32::from(add_special),
                    buf.as_mut_ptr(),
                    len_i32(buf.len()),
                )
            };
            let count = token_count(n2)?;
            buf.truncate(count);
            return Ok(buf);
        }
        buf.truncate(count_usize(n));
        Ok(buf)
    }

    /// Detokenize one id. Used by the `BONJEV_TOKEN_DUMP` diagnostics.
    pub fn token_piece(&self, id: i32) -> Result<String> {
        if id < 0 {
            bail!("negative token id");
        }
        let mut buf = vec![0u8; 64];
        loop {
            // SAFETY: self.ctx is live, buf holds `buf.len()` writable bytes.
            let n = unsafe {
                jf_token_piece(self.ctx, id, buf.as_mut_ptr().cast(), len_i32(buf.len()))
            };
            if n >= 0 {
                buf.truncate(count_usize(n));
                return Ok(String::from_utf8_lossy(&buf).into_owned());
            }
            let need = usize::try_from(-n).map_err(|_| anyhow!("token piece overflow"))?;
            if need <= buf.len() || need > 4096 {
                bail!("token piece detokenization failed ({n})");
            }
            buf.resize(need, 0);
        }
    }

    fn ensure_fits_seq(&self, len: usize) -> Result<()> {
        // SAFETY: self.ctx is the handle owned by this Engine.
        let max = unsafe { jf_n_ctx_seq(self.ctx) };
        if max <= 0 {
            return Ok(());
        }
        let max = count_usize(max);
        if len > max {
            bail!(EngineFault::too_long(format!(
                "prompt has {len} tokens but the maximum context length is {max} \
                 (n_ctx is shared across {n} sequences in this process)",
                n = self.n_seq
            )));
        }
        Ok(())
    }

    pub fn decode(&mut self, tokens: &[i32]) -> Result<()> {
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        if !fits_i32(tokens.len()) {
            bail!("prompt longer than i32::MAX tokens");
        }
        self.ensure_fits_seq(tokens.len())?;
        // SAFETY: tokens is a live slice. jf_decode reads exactly `len` tokens and does not
        // keep the pointer after it returns.
        let rc = unsafe { jf_decode(self.ctx, tokens.as_ptr(), len_i32(tokens.len())) };
        if rc != 0 {
            bail!(EngineFault::decode(format!("decode failed rc={rc}")));
        }
        Ok(())
    }

    /// One logit row per sequence, in input order. A single sequence uses the same decode as `decode`.
    /// Up to three sequences that share a token prefix store that prefix once. That is how `top3` scores one question.
    pub fn decode_slices(&mut self, seqs: &[&[i32]]) -> Result<Vec<Vec<f32>>> {
        if seqs.is_empty() {
            bail!("empty batch");
        }
        if seqs.len() == 1 {
            self.decode(seqs[0])?;
            return Ok(vec![self.logits()?.to_vec()]);
        }
        if seqs.len() > self.n_seq {
            bail!(
                "batch of {} sequences exceeds {} engine sequences",
                seqs.len(),
                self.n_seq
            );
        }
        for seq in seqs {
            if seq.is_empty() {
                bail!("empty prompt");
            }
            if !fits_i32(seq.len()) {
                bail!("prompt longer than i32::MAX tokens");
            }
            self.ensure_fits_seq(seq.len())?;
        }
        let ptrs: Vec<*const i32> = seqs.iter().map(|seq| seq.as_ptr()).collect();
        let lens: Vec<i32> = seqs.iter().map(|seq| len_i32(seq.len())).collect();
        let mut out = vec![0.0f32; seqs.len() * self.n_vocab];
        // SAFETY: ptrs and lens live for this call, each sequence pointer addresses `lens[i]`
        // tokens, and `out` has n_seq * n_vocab floats. The shim copies logits out before return.
        let rc = unsafe {
            jf_decode_many(
                self.ctx,
                ptrs.as_ptr(),
                lens.as_ptr(),
                len_i32(seqs.len()),
                out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            bail!(EngineFault::decode(format!("decode failed rc={rc}")));
        }
        Ok(out.chunks(self.n_vocab).map(|row| row.to_vec()).collect())
    }

    /// Tokens reused from KV vs prompt length for the last decode call,
    /// one `(kept, total)` pair per sequence in input order.
    pub fn last_keep(&self) -> Vec<(usize, usize)> {
        let mut nseq = 0i32;
        let mut keep = [0i32; JF_SEQ_MAX];
        let mut total = [0i32; JF_SEQ_MAX];
        // SAFETY: buffers hold JF_SEQ_MAX i32 each; the shim writes at most `cap` entries.
        unsafe {
            jf_last_keep(
                self.ctx,
                &mut nseq,
                keep.as_mut_ptr(),
                total.as_mut_ptr(),
                JF_SEQ_MAX as i32,
            )
        };
        let n = count_usize(nseq.clamp(0, JF_SEQ_MAX as i32));
        (0..n)
            .map(|i| {
                (
                    usize::try_from(keep[i].max(0)).unwrap_or(0),
                    usize::try_from(total[i].max(0)).unwrap_or(0),
                )
            })
            .collect()
    }

    pub fn logits(&self) -> Result<&[f32]> {
        // SAFETY: self.ctx is the handle owned by this Engine. A null return is checked below.
        let p = unsafe { jf_logits(self.ctx) };
        if p.is_null() {
            bail!(EngineFault::logits("logits unavailable"));
        }
        // SAFETY: non-null pointer from llama for this context, length n_vocab, valid until
        // the next decode or drop. `c_float` is f32 on the platforms this binary links.
        Ok(unsafe { std::slice::from_raw_parts(p.cast::<f32>(), self.n_vocab) })
    }
}

fn token_count(n: i32) -> Result<usize> {
    if n == i32::MIN {
        bail!("tokenize overflow");
    }
    if n < 0 {
        bail!("tokenize failed ({n})");
    }
    Ok(count_usize(n))
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: ctx was returned by jf_load and this Drop runs once.
        unsafe { jf_free(self.ctx) };
    }
}

#[cfg(test)]
mod tests {
    use super::prism_marker;

    #[test]
    fn prism_marker_is_relative() {
        let marker = prism_marker();
        let path = std::path::Path::new(marker);
        assert!(path.is_relative(), "{marker}");
        assert!(
            path.components()
                .all(|part| part != std::path::Component::ParentDir),
            "{marker}"
        );
        assert!(marker.starts_with("vendor/"), "{marker}");
    }
}
