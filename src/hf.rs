use anyhow::{Context, Result};
use std::path::PathBuf;

pub struct ModelId {
    pub name: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub about: &'static str,
}

const MODELS: &[ModelId] = &[
    ModelId {
        name: "q2",
        repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
        file: "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        about: "Ternary-Bonsai-2-27B PQ2_0, 7.2 ГБ",
    },
    ModelId {
        name: "q1",
        repo: "prism-ml/Bonsai-27B-gguf",
        file: "Bonsai-27B-Q1_0.gguf",
        about: "Bonsai-27B Q1_0, 3.6 ГБ",
    },
];

pub fn models() -> &'static [ModelId] {
    MODELS
}

pub fn canonical(name: &str) -> Result<&'static ModelId> {
    let key = name.to_ascii_lowercase();
    MODELS
        .iter()
        .find(|model| match key.as_str() {
            "q2" | "bonsai-q2" | "pq2" | "pq2_0" => model.name == "q2",
            "q1" | "bonsai-q1" | "q1_0" => model.name == "q1",
            _ => false,
        })
        .ok_or_else(|| anyhow::anyhow!("unknown model '{name}' (use q2|q1)"))
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

pub const Q2_MMPROJ: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";

pub fn q2_mmproj() -> Result<PathBuf> {
    let repo = MODELS[0].repo;
    if let Some(path) = cached_file(repo, Q2_MMPROJ) {
        return Ok(path);
    }
    download_file(repo, Q2_MMPROJ)
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

#[cfg(test)]
mod tests {
    use super::canonical;

    #[test]
    fn aliases_map_to_one_id() {
        assert_eq!(canonical("PQ2_0").unwrap().name, "q2");
        assert_eq!(canonical("q1_0").unwrap().name, "q1");
        assert!(canonical("q3").is_err());
    }
}
