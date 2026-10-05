use anyhow::{Context, Result};
use std::path::PathBuf;

pub struct ModelId {
    /// Canonical slug (`bonsai-8b`, `ternary-bonsai-2-27b`, …).
    pub name: &'static str,
    pub repo: &'static str,
    /// Primary GGUF file. Empty for non-GGUF catalog entries.
    pub file: &'static str,
    /// Vision projector file, when the model has one. Text-only models skip vision.
    pub mmproj: Option<&'static str>,
    pub about: &'static str,
    aliases: &'static [&'static str],
}

const MODELS: &[ModelId] = &[
    ModelId {
        name: "ternary-bonsai-2-27b",
        repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
        file: "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        mmproj: Some("Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"),
        about: "Bonsai 2 (Ternary-Bonsai-2-27B) PQ2_0, ~7.2 GB",
        aliases: &["bonsai2", "bonsai-2", "bonsai-2-27b", "ternary-bonsai-2"],
    },
    ModelId {
        name: "ternary-bonsai-2-27b-ptq1",
        repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
        file: "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        mmproj: None,
        about: "Bonsai 2 27B PTQ1_0 (dense trits), ~5.5 GB",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-2-27b-f16",
        repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
        file: "Ternary-Bonsai-2-27B-F16.gguf",
        mmproj: None,
        about: "Bonsai 2 27B F16, ~50 GB (needs ~50 GB VRAM/RAM)",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-27b",
        repo: "prism-ml/Ternary-Bonsai-27B-gguf",
        file: "Ternary-Bonsai-27B-PQ2_0.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-27B PQ2_0, ~6.7 GB",
        aliases: &["ternary-27b", "ternary-bonsai27b"],
    },
    ModelId {
        name: "ternary-bonsai-27b-q2g64",
        repo: "prism-ml/Ternary-Bonsai-27B-gguf",
        file: "Ternary-Bonsai-27B-Q2_g64.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-27B Q2_0 group-64",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-27b-f16",
        repo: "prism-ml/Ternary-Bonsai-27B-gguf",
        file: "Ternary-Bonsai-27B-F16.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-27B F16, ~50 GB",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-8b",
        repo: "prism-ml/Ternary-Bonsai-8B-gguf",
        file: "Ternary-Bonsai-8B-PQ2_0.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-8B PQ2_0",
        aliases: &["ternary-8b", "ternary-bonsai8b", "8b"],
    },
    ModelId {
        name: "ternary-bonsai-8b-q2g64",
        repo: "prism-ml/Ternary-Bonsai-8B-gguf",
        file: "Ternary-Bonsai-8B-Q2_0_g64.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-8B Q2_0 group-64",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-8b-f16",
        repo: "prism-ml/Ternary-Bonsai-8B-gguf",
        file: "Ternary-Bonsai-8B-F16.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-8B F16, ~15 GB",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-4b",
        repo: "prism-ml/Ternary-Bonsai-4B-gguf",
        file: "Ternary-Bonsai-4B-PQ2_0.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-4B PQ2_0",
        aliases: &["ternary-4b", "ternary-bonsai4b", "4b"],
    },
    ModelId {
        name: "ternary-bonsai-4b-q2g64",
        repo: "prism-ml/Ternary-Bonsai-4B-gguf",
        file: "Ternary-Bonsai-4B-Q2_0_g64.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-4B Q2_0 group-64",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-4b-f16",
        repo: "prism-ml/Ternary-Bonsai-4B-gguf",
        file: "Ternary-Bonsai-4B-F16.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-4B F16",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-1.7b",
        repo: "prism-ml/Ternary-Bonsai-1.7B-gguf",
        file: "Ternary-Bonsai-1.7B-PQ2_0.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-1.7B PQ2_0",
        aliases: &["ternary-1.7b", "ternary-bonsai1.7b", "1.7b"],
    },
    ModelId {
        name: "ternary-bonsai-1.7b-q2g64",
        repo: "prism-ml/Ternary-Bonsai-1.7B-gguf",
        file: "Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-1.7B Q2_0 group-64",
        aliases: &[],
    },
    ModelId {
        name: "ternary-bonsai-1.7b-f16",
        repo: "prism-ml/Ternary-Bonsai-1.7B-gguf",
        file: "Ternary-Bonsai-1.7B-F16.gguf",
        mmproj: None,
        about: "Ternary-Bonsai-1.7B F16",
        aliases: &[],
    },
];

pub fn models() -> &'static [ModelId] {
    MODELS
}

fn norm_key(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['_', ' '], "-")
}

pub fn canonical(name: &str) -> Result<&'static ModelId> {
    let key = norm_key(name);
    MODELS
        .iter()
        .find(|model| key == model.name || model.aliases.iter().any(|alias| norm_key(alias) == key))
        .ok_or_else(|| {
            let ids: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
            anyhow::anyhow!(
                "unknown model '{name}' (try one of: {ids})",
                ids = ids.join(", ")
            )
        })
}

/// Local Hugging Face cache only. Does not download.
pub fn cached(model: &ModelId) -> Option<PathBuf> {
    hf_hub::Cache::from_env()
        .model(model.repo.to_string())
        .get(model.file)
}

pub fn any_cached() -> bool {
    MODELS.iter().any(|model| cached(model).is_some())
}

pub fn download(model: &ModelId) -> Result<PathBuf> {
    download_file(model.repo, model.file)
}

pub fn mmproj(model: &ModelId) -> Result<PathBuf> {
    let Some(file) = model.mmproj else {
        anyhow::bail!("model '{}' has no vision projector", model.name);
    };
    if let Some(path) = cached_file(model.repo, file) {
        return Ok(path);
    }
    download_file(model.repo, file)
}

fn cached_file(repo: &str, file: &str) -> Option<PathBuf> {
    hf_hub::Cache::from_env().model(repo.to_string()).get(file)
}

fn download_file(repo: &str, file: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new()?;
    api.model(repo.to_string())
        .get(file)
        .with_context(|| format!("download {repo}/{file}"))
}

// ---------------------------------------------------------------------------
// Decision LoRA adapters
// ---------------------------------------------------------------------------

/// A decision LoRA published for one or more base models.
pub struct LoraId {
    /// Short name used on the CLI and in `BONJEV_LORA` (e.g. `b10`).
    pub name: &'static str,
    /// Canonical model ids this adapter is built for.
    pub bases: &'static [&'static str],
    /// Hugging Face repository name, without the namespace.
    pub repo: &'static str,
    /// File inside the repository (may be under `alt/`).
    pub file: &'static str,
    pub about: &'static str,
    aliases: &'static [&'static str],
}

/// Default namespace for the published adapter repos. Override with
/// `BONJEV_LORA_NAMESPACE`.
const LORA_NAMESPACE: &str = "ChezRD";

const LORAS: &[LoraId] = &[
    LoraId {
        name: "m17_tiny25_single",
        bases: &["ternary-bonsai-1.7b"],
        repo: "ternary-bonsai-1.7b-decision-lora",
        file: "m17_tiny25_single.gguf",
        about: "1.7B default: Tiny-Jev x2.5 (231 124->143; 147 with BONJEV_STYLE=top3)",
        aliases: &["tiny25", "m17"],
    },
    LoraId {
        name: "b10",
        bases: &["ternary-bonsai-4b"],
        repo: "ternary-bonsai-4b-decision-lora",
        file: "B10.gguf",
        about: "4B default: early kev4b + candigate + senna (145->161; DI .4358->.4723)",
        aliases: &["default"],
    },
    LoraId {
        name: "all3_align",
        bases: &["ternary-bonsai-4b"],
        repo: "ternary-bonsai-4b-decision-lora",
        file: "alt/m4b_all3_align.gguf",
        about: "4B alt: max 231 (163 at lambda 1.25) but DI below base",
        aliases: &["m4b_all3_align"],
    },
    LoraId {
        name: "m8_inv_b04",
        bases: &["ternary-bonsai-8b"],
        repo: "ternary-bonsai-8b-decision-lora",
        file: "m8_inv_b04.gguf",
        about: "8B default: kev8b + LCT_attn 0.25 (149->159; DI ties the base)",
        aliases: &["inv", "default"],
    },
    LoraId {
        name: "r3",
        bases: &["ternary-bonsai-8b"],
        repo: "ternary-bonsai-8b-decision-lora",
        file: "alt/R3.gguf",
        about: "8B alt: 159 but over-confident (not recommended)",
        aliases: &["R3"],
    },
    LoraId {
        name: "vega_clef_plumb",
        bases: &["ternary-bonsai-2-27b", "ternary-bonsai-2-27b-ptq1"],
        repo: "ternary-bonsai-2-27b-decision-lora",
        file: "vega_clef_plumb.gguf",
        about: "27B default: Vega + clef + plumb (194->207; DI .5772->.6090)",
        aliases: &["win", "default"],
    },
    LoraId {
        name: "m27_clef_plumb_at_half",
        bases: &["ternary-bonsai-2-27b", "ternary-bonsai-2-27b-ptq1"],
        repo: "ternary-bonsai-2-27b-decision-lora",
        file: "alt/m27_clef_plumb_at_half.gguf",
        about: "27B alt: ties 207, no Vega",
        aliases: &["at_half"],
    },
    LoraId {
        name: "vega_clef_plumb_at_half",
        bases: &["ternary-bonsai-2-27b", "ternary-bonsai-2-27b-ptq1"],
        repo: "ternary-bonsai-2-27b-decision-lora",
        file: "alt/vega_clef_plumb_at_half.gguf",
        about: "27B alt: ties 207",
        aliases: &["vega_half"],
    },
    LoraId {
        name: "clef_plumb_vegaffn",
        bases: &["ternary-bonsai-2-27b", "ternary-bonsai-2-27b-ptq1"],
        repo: "ternary-bonsai-2-27b-decision-lora",
        file: "alt/clef_plumb_vegaffn.gguf",
        about: "27B alt: 206, Vega-FFN only",
        aliases: &["vegaffn"],
    },
];

pub fn loras() -> &'static [LoraId] {
    LORAS
}

/// Printed by `bonjev loras`: the multi-adapter behavior is real but research-only.
pub const LORA_NOTE: &str = "\
Multiple --lora entries are attached additively (up to 8, research use only). Tested runtime \
composites did not beat the single published adapter per model, so ship one adapter per model. \
Measured (Stage MA): 1.7B tiny,2.5 + xuhao_ffn,0.25 = 142 (<144); 8B kev,1.5 + lct_attn,0.25 = \
159 (<160); 4B kev4b + candigate + senna = 158 (<161).";

/// Adapters built for a given canonical model id.
pub fn loras_for(model_name: &str) -> impl Iterator<Item = &'static LoraId> + '_ {
    LORAS
        .iter()
        .filter(move |lora| lora.bases.contains(&model_name))
}

fn namespace() -> String {
    std::env::var("BONJEV_LORA_NAMESPACE").unwrap_or_else(|_| LORA_NAMESPACE.to_string())
}

/// Full `namespace/repo` for an adapter.
pub fn lora_repo(lora: &LoraId) -> String {
    format!("{}/{}", namespace(), lora.repo)
}

/// Resolve a short LoRA name against a model's adapters.
pub fn find_lora(model_name: &str, name: &str) -> Result<&'static LoraId> {
    let key = norm_key(name);
    let mut names: Vec<&str> = Vec::new();
    for lora in loras_for(model_name) {
        if key == norm_key(lora.name) || lora.aliases.iter().any(|a| norm_key(a) == key) {
            return Ok(lora);
        }
        names.push(lora.name);
    }
    if names.is_empty() {
        anyhow::bail!("model '{model_name}' has no published LoRA adapters");
    }
    anyhow::bail!(
        "unknown LoRA '{name}' for '{model_name}' (try one of: {})",
        names.join(", ")
    )
}

/// A CLI/env entry is a filesystem path if it is a `.gguf`, absolute/relative,
/// or an existing file; otherwise it is a short LoRA name.
fn looks_like_path(entry: &str) -> bool {
    entry.ends_with(".gguf")
        || entry.starts_with('/')
        || entry.starts_with('.')
        || entry.starts_with('~')
        || PathBuf::from(entry).is_file()
}

/// Resolve one `target[,scale]` entry to a `path[,scale]` string.
fn resolve_lora_target(model_name: &str, target: &str) -> Result<String> {
    if looks_like_path(target) {
        let path = PathBuf::from(target);
        if !path.is_file() {
            anyhow::bail!("LoRA file not found: {}", path.display());
        }
        return Ok(path.display().to_string());
    }
    let lora = find_lora(model_name, target)?;
    let repo = lora_repo(lora);
    let path = match cached_file(&repo, lora.file) {
        Some(path) => path,
        None => {
            eprintln!("[bonjev] downloading LoRA {} ({})", lora.name, lora.about);
            download_file(&repo, lora.file)?
        }
    };
    Ok(path.display().to_string())
}

/// Maximum adapters the C shim can attach (`JF_LORA_MAX` in `c/src/shim.c`).
pub const MAX_LORAS: usize = 8;

/// Resolve a `--lora` / `BONJEV_LORA` value into a shim-ready
/// `path[,scale][;path[,scale]...]` list. Short names are downloaded through the
/// Hugging Face client; local `.gguf` paths are kept as-is.
pub fn resolve_lora_spec(model_name: &str, raw: &str) -> Result<String> {
    let mut out: Vec<String> = Vec::new();
    for entry in raw.split(';') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (target, scale) = match entry.rsplit_once(',') {
            Some((target, scale)) if scale.trim().parse::<f32>().is_ok() => {
                (target.trim(), Some(scale.trim()))
            }
            _ => (entry, None),
        };
        let path = resolve_lora_target(model_name, target)?;
        out.push(match scale {
            Some(scale) => format!("{path},{scale}"),
            None => path,
        });
    }
    if out.is_empty() {
        anyhow::bail!("empty LoRA spec");
    }
    if out.len() > MAX_LORAS {
        anyhow::bail!("too many LoRA adapters: {} (max {MAX_LORAS})", out.len());
    }
    Ok(out.join(";"))
}

/// Flatten repeated `--lora` values and the `BONJEV_LORA` env into non-empty
/// entries. CLI values take precedence; each value may itself be a
/// `;`-separated list.
pub fn split_lora_entries(values: &[String], env: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for value in values {
        out.extend(
            value
                .split(';')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(str::to_string),
        );
    }
    if out.is_empty()
        && let Some(value) = env
    {
        out.extend(
            value
                .split(';')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(str::to_string),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        canonical, find_lora, lora_repo, norm_key, resolve_lora_spec, resolve_lora_target,
        split_lora_entries,
    };

    #[test]
    fn norm_key_collapses_separators() {
        assert_eq!(norm_key("Bonsai_8B"), "bonsai-8b");
        assert_eq!(norm_key("Bonsai-2-27B"), "bonsai-2-27b");
    }

    #[test]
    fn readable_aliases_map_to_commercial_ids() {
        assert_eq!(canonical("bonsai2").unwrap().name, "ternary-bonsai-2-27b");
        assert_eq!(
            canonical("Bonsai-2-27B").unwrap().name,
            "ternary-bonsai-2-27b"
        );
        assert_eq!(canonical("8b").unwrap().name, "ternary-bonsai-8b");
        assert_eq!(canonical("ternary-8b").unwrap().name, "ternary-bonsai-8b");
        assert_eq!(canonical("4b").unwrap().name, "ternary-bonsai-4b");
        assert_eq!(
            canonical("ternary-bonsai-4b").unwrap().name,
            "ternary-bonsai-4b"
        );
        assert_eq!(canonical("1.7b").unwrap().name, "ternary-bonsai-1.7b");
        assert_eq!(
            canonical("Ternary-Bonsai-1.7B").unwrap().name,
            "ternary-bonsai-1.7b"
        );
        assert_eq!(canonical("ternary-27b").unwrap().name, "ternary-bonsai-27b");
    }

    #[test]
    fn old_quant_aliases_are_gone() {
        assert!(canonical("q1").is_err());
        assert!(canonical("q2").is_err());
        assert!(canonical("q1_0").is_err());
        assert!(canonical("pq2_0").is_err());
    }

    #[test]
    fn only_bonsai_2_27b_has_a_vision_projector() {
        assert_eq!(
            canonical("ternary-bonsai-2-27b").unwrap().mmproj,
            Some("Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf")
        );
        assert_eq!(canonical("ternary-bonsai-8b").unwrap().mmproj, None);
        assert_eq!(canonical("ternary-bonsai-1.7b").unwrap().mmproj, None);
    }

    #[test]
    fn unknown_model_errors() {
        assert!(canonical("q3").is_err());
        assert!(canonical("bonsai-32b").is_err());
    }

    #[test]
    fn lora_names_are_scoped_to_their_model() {
        assert_eq!(
            find_lora("ternary-bonsai-4b", "b10").unwrap().file,
            "B10.gguf"
        );
        assert_eq!(
            find_lora("ternary-bonsai-4b", "default").unwrap().name,
            "b10"
        );
        assert_eq!(
            find_lora("ternary-bonsai-4b", "m4b_all3_align").unwrap().name,
            "all3_align"
        );
        assert!(find_lora("ternary-bonsai-8b", "b10").is_err());
        assert_eq!(
            find_lora("ternary-bonsai-2-27b", "vega_clef_plumb")
                .unwrap()
                .file,
            "vega_clef_plumb.gguf"
        );
    }

    #[test]
    fn twenty_seven_b_loras_cover_both_quants() {
        for name in [
            "vega_clef_plumb",
            "m27_clef_plumb_at_half",
            "vega_clef_plumb_at_half",
            "clef_plumb_vegaffn",
        ] {
            assert!(find_lora("ternary-bonsai-2-27b", name).is_ok());
            assert!(find_lora("ternary-bonsai-2-27b-ptq1", name).is_ok());
        }
    }

    #[test]
    fn lora_repo_uses_the_published_namespace() {
        let lora = find_lora("ternary-bonsai-8b", "inv").unwrap();
        assert!(lora_repo(lora).ends_with("/ternary-bonsai-8b-decision-lora"));
    }

    #[test]
    fn missing_local_lora_path_is_rejected() {
        assert!(resolve_lora_target("ternary-bonsai-4b", "/nope/missing.gguf").is_err());
    }

    #[test]
    fn split_lora_entries_merges_repeats_and_env() {
        let cli = vec!["b10,1.25".to_string(), "all3_align,0.5".to_string()];
        assert_eq!(
            split_lora_entries(&cli, None),
            vec!["b10,1.25", "all3_align,0.5"]
        );
        let cli = vec!["a;b".to_string()];
        assert_eq!(split_lora_entries(&cli, None), vec!["a", "b"]);
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            split_lora_entries(&empty, Some("x,1.0; y,0.5")),
            vec!["x,1.0", "y,0.5"]
        );
        // CLI values win over the env.
        assert_eq!(
            split_lora_entries(&["z".to_string()], Some("x")),
            vec!["z"]
        );
    }

    #[test]
    fn multiple_local_loras_keep_scales() {
        let dir = std::env::temp_dir().join(format!("bonjev-lora-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        std::fs::write(&a, b"x").unwrap();
        std::fs::write(&b, b"x").unwrap();
        let spec = format!("{},1.25;{},0.5", a.display(), b.display());
        let out = resolve_lora_spec("ternary-bonsai-4b", &spec).unwrap();
        assert_eq!(out, format!("{},1.25;{},0.5", a.display(), b.display()));
        std::fs::remove_dir_all(&dir).ok();
    }
}
