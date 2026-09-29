use crate::ffi::*;
use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Once;

pub struct Engine {
    ctx: *mut JfCtx,
    pub n_vocab: usize,
    pub n_seq: usize,
    media_marker: Option<String>,
    prior_logits: Option<Vec<f32>>,
}

// SAFETY: `ctx` is owned by this value and is not shared. The type is moved between threads
// only while no borrow of it is live.
unsafe impl Send for Engine {}

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
    let c_dir = CString::new(dir.as_os_str().as_bytes())?;
    ONCE.call_once(|| unsafe { jf_load_backends(c_dir.as_ptr()) });
    Ok(())
}

fn fit_i32(what: &str, value: u32) -> Result<i32> {
    i32::try_from(value).map_err(|_| anyhow!("{what}={value} does not fit into i32"))
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| anyhow!("path contains a NUL byte"))
}

impl Engine {
    pub fn load(path: &Path, n_ctx: u32, n_threads: u32, n_gpu_layers: i32) -> Result<Self> {
        Self::load_seqs(path, n_ctx, n_threads, n_gpu_layers, 1)
    }

    pub fn load_seqs(
        path: &Path,
        n_ctx: u32,
        n_threads: u32,
        n_gpu_layers: i32,
        n_seq: u32,
    ) -> Result<Self> {
        let n_ctx = fit_i32("ctx", n_ctx)?;
        let n_threads = fit_i32("threads", n_threads)?;
        let n_seq_i = fit_i32("parallel", n_seq)?;
        if n_ctx <= 0 {
            bail!("ctx must be positive");
        }
        if !(1..=16).contains(&n_seq_i) {
            bail!("parallel must be from 1 to 16");
        }
        init_backends()?;
        let c_path = c_path(path)?;
        // SAFETY: c_path is a NUL-terminated path that outlives the call. jf_load copies
        // what it needs before returning. A null handle is reported below.
        let ctx = unsafe { jf_load(c_path.as_ptr(), n_ctx, n_threads, n_gpu_layers, n_seq_i) };
        if ctx.is_null() {
            bail!("llama: failed to load {}", path.display());
        }
        let n_vocab = unsafe { jf_n_vocab(ctx) };
        if n_vocab <= 0 {
            unsafe { jf_free(ctx) };
            bail!("model vocabulary is empty");
        }
        Ok(Self {
            ctx,
            n_vocab: n_vocab as usize,
            n_seq: n_seq_i as usize,
            media_marker: None,
            prior_logits: None,
        })
    }

    pub fn load_vision(&mut self, path: &Path, n_threads: u32) -> Result<()> {
        let c_path = c_path(path)?;
        let threads = fit_i32("threads", n_threads)?;
        // SAFETY: c_path outlives the call. The projector stays owned by the context.
        let rc = unsafe { jf_vision_load(self.ctx, c_path.as_ptr(), threads) };
        if rc != 0 {
            bail!("failed to load vision projector {}", path.display());
        }
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
        let expect = (width as usize)
            .checked_mul(height as usize)
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
            bail!("image decode failed rc={rc}");
        }
        if n_tokens < 0 {
            bail!("image decode reported a negative token count");
        }
        Ok(n_tokens as usize)
    }

    pub fn prior_logits(&self) -> Option<&[f32]> {
        self.prior_logits.as_deref()
    }

    pub fn set_prior_logits(&mut self, logits: Vec<f32>) {
        self.prior_logits = Some(logits);
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let c_text = CString::new(text)?;
        let cap = text.len().saturating_add(16);
        if cap > i32::MAX as usize {
            bail!("tokenize input too long");
        }
        let mut buf = vec![0; cap];
        let n = unsafe {
            jf_tokenize(
                self.ctx,
                c_text.as_ptr(),
                i32::from(add_special),
                buf.as_mut_ptr(),
                buf.len() as i32,
            )
        };
        if n == i32::MIN {
            bail!("tokenize overflow");
        }
        if n < 0 {
            let need = usize::try_from(-n).map_err(|_| anyhow!("tokenize overflow"))?;
            if need > i32::MAX as usize {
                bail!("tokenize overflow");
            }
            buf.resize(need, 0);
            let n2 = unsafe {
                jf_tokenize(
                    self.ctx,
                    c_text.as_ptr(),
                    i32::from(add_special),
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                )
            };
            let count = token_count(n2)?;
            buf.truncate(count);
            return Ok(buf);
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    pub fn decode(&mut self, tokens: &[i32]) -> Result<()> {
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        if tokens.len() > i32::MAX as usize {
            bail!("prompt longer than i32::MAX tokens");
        }
        // SAFETY: tokens is a live slice. jf_decode reads exactly `len` tokens and does not
        // keep the pointer after it returns.
        let rc = unsafe { jf_decode(self.ctx, tokens.as_ptr(), tokens.len() as i32) };
        if rc != 0 {
            bail!("decode failed rc={rc}");
        }
        Ok(())
    }

    /// One logit row per sequence, in input order. A single sequence uses the same decode as `decode`.
    /// Several sequences that share a token prefix store that prefix once.
    pub fn decode_many(&mut self, seqs: &[Vec<i32>]) -> Result<Vec<Vec<f32>>> {
        let borrowed: Vec<&[i32]> = seqs.iter().map(Vec::as_slice).collect();
        self.decode_slices(&borrowed)
    }

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
                "batch of {} sequences exceeds parallel {}",
                seqs.len(),
                self.n_seq
            );
        }
        for seq in seqs {
            if seq.is_empty() {
                bail!("empty prompt");
            }
            if seq.len() > i32::MAX as usize {
                bail!("prompt longer than i32::MAX tokens");
            }
        }
        let ptrs: Vec<*const i32> = seqs.iter().map(|seq| seq.as_ptr()).collect();
        let lens: Vec<i32> = seqs.iter().map(|seq| seq.len() as i32).collect();
        let mut out = vec![0.0f32; seqs.len() * self.n_vocab];
        // SAFETY: ptrs and lens live for this call, each sequence pointer addresses `lens[i]`
        // tokens, and `out` has n_seq * n_vocab floats. The shim copies logits out before return.
        let rc = unsafe {
            jf_decode_many(
                self.ctx,
                ptrs.as_ptr(),
                lens.as_ptr(),
                seqs.len() as i32,
                out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            bail!("decode failed rc={rc}");
        }
        Ok(out.chunks(self.n_vocab).map(|row| row.to_vec()).collect())
    }

    pub fn logits(&self) -> Result<&[f32]> {
        let p = unsafe { jf_logits(self.ctx) };
        if p.is_null() {
            bail!("logits unavailable");
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
    Ok(n as usize)
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { jf_free(self.ctx) };
    }
}

#[cfg(test)]
mod tests {
    use super::prism_marker;

    #[test]
    fn prism_marker_is_relative() {
        let marker = prism_marker();
        assert!(marker.starts_with("vendor/"));
        assert!(!marker.starts_with('/'));
    }
}
