use crate::engine::Engine;
use crate::picture::{self, Rgb};
use crate::prompt::{labels_for, prior_alpha, prior_enabled, render, with_media, Axis, Row};
use crate::readout::{apply_prior_calibration, forms, slot_logits, softmax_slots};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::hash::{Hash, Hasher};

#[derive(Debug, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub state: StateValue,
    pub questions: Questions,
    /// Base64 image, or a data URL. jpeg, png, gif, webp or bmp.
    #[serde(default)]
    pub image: Option<String>,
    /// Raw file bytes from the CLI. Not part of the JSON body.
    #[serde(skip)]
    pub image_bytes: Option<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
pub struct Questions {
    pub decision: Question,
}

#[derive(Debug, Deserialize)]
pub struct Question {
    #[serde(rename = "type", default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub criteria: Criteria,
}

fn default_kind() -> String {
    "choice".to_string()
}

const IMAGE_STATE_KEYS: &[&str] = &["image", "image_base64", "image_b64", "image_url"];

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StateValue {
    Text(String),
    Other(Value),
}

impl Default for StateValue {
    fn default() -> Self {
        Self::Other(Value::Null)
    }
}

fn state_value_text(val: &Value) -> String {
    match val {
        Value::String(s) => s.trim().to_string(),
        Value::Object(map) => {
            let mut lines = Vec::with_capacity(map.len());
            for (k, v) in map {
                if IMAGE_STATE_KEYS.contains(&k.as_str()) {
                    continue;
                }
                let body = match v {
                    Value::String(s) => s.trim().to_string(),
                    other => state_value_text(other),
                };
                lines.push(format!("{k}: {body}"));
            }
            lines.join("\n")
        }
        Value::Array(arr) => {
            let mut parts = Vec::with_capacity(arr.len());
            for v in arr {
                let body = match v {
                    Value::String(s) => s.trim().to_string(),
                    other => state_value_text(other),
                };
                parts.push(body);
            }
            parts.join("\n")
        }
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

impl StateValue {
    fn text(&self) -> String {
        match self {
            Self::Text(text) => text.trim().to_string(),
            Self::Other(value) => state_value_text(value),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Criteria {
    Map(Map<String, Value>),
    List(Vec<Value>),
    Other(Value),
}

impl Default for Criteria {
    fn default() -> Self {
        Self::Other(Value::Null)
    }
}

fn criterion_text(value: &Value) -> String {
    value.as_str().unwrap_or("").to_string()
}

fn axis_of(kind: &str) -> Axis {
    match kind {
        "noul" => Axis::Noul,
        "score" => Axis::Score,
        _ => Axis::Choice,
    }
}

fn options_from(criteria: &Criteria, axis: Axis) -> Vec<(String, String)> {
    match (axis, criteria) {
        (Axis::Score, Criteria::List(items)) => items
            .iter()
            .enumerate()
            .map(|(i, value)| (i.to_string(), criterion_text(value)))
            .collect(),
        (Axis::Score, _) => Vec::new(),
        (_, Criteria::Map(map)) => map
            .iter()
            .map(|(key, value)| (key.clone(), criterion_text(value)))
            .collect(),
        (_, Criteria::Other(other)) => {
            let _ = other;
            Vec::new()
        }
        _ => Vec::new(),
    }
}

pub fn from_options(state: &str, question: &str, kind: &str, options: &[String]) -> Request {
    let mut map = Map::new();
    for (i, text) in options.iter().enumerate() {
        map.insert(format!("option_{i}"), Value::String(text.clone()));
    }
    Request {
        state: StateValue::Text(state.to_string()),
        questions: Questions {
            decision: Question {
                kind: kind.to_string(),
                instructions: question.to_string(),
                criteria: Criteria::Map(map),
            },
        },
        image: None,
        image_bytes: None,
    }
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: Answers,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Answers {
    pub decision: DecisionOut,
}

#[derive(Debug, Serialize)]
pub struct DecisionOut {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub probabilities: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noul: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: u32,
}

fn positive_argmax(probs: &[f32]) -> Result<usize> {
    if !probs.iter().any(|p| p.is_finite() && *p > 0.0) {
        bail!("no label token produced a finite logit");
    }
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .ok_or_else(|| anyhow::anyhow!("no label token produced a finite logit"))
}

fn noul_true_probability(options: &[(String, String)], probs: &[f32]) -> Result<f32> {
    if !probs.iter().any(|p| p.is_finite() && *p > 0.0) {
        bail!("no label token produced a finite logit");
    }
    let i = options
        .iter()
        .position(|(key, _)| key.eq_ignore_ascii_case("true") || key.eq_ignore_ascii_case("yes"))
        .ok_or_else(|| anyhow::anyhow!("noul criteria need a true or yes key"))?;
    probs
        .get(i)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("probability count does not match options"))
}

fn prior_cache_key(axis: Axis, labels: &[String], picture: Option<&Rgb>) -> String {
    let axis_tag = match axis {
        Axis::Choice => "choice",
        Axis::Noul => "noul",
        Axis::Score => "score",
    };
    let mut key = format!("{}:{}", axis_tag, labels.join(","));
    if let Some(picture) = picture {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        picture.width.hash(&mut hasher);
        picture.height.hash(&mut hasher);
        picture.pixels.hash(&mut hasher);
        key.push_str(&format!(":img:{:016x}", hasher.finish()));
    }
    key
}

fn null_prior_slots(
    engine: &mut Engine,
    row: &Row,
    labels: &[String],
    merge_space: bool,
    picture: Option<&Rgb>,
) -> Result<Vec<f32>> {
    let key = prior_cache_key(row.axis, labels, picture);
    if let Some(cached) = engine.prior_get(&key) {
        return Ok(cached);
    }
    let options: Vec<(String, String)> = row
        .options
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (id.clone(), format!("{}: Option {}.", labels[i], labels[i])))
        .collect();
    let null_row = Row {
        state: "N/A".to_string(),
        question: "Which option follows?".to_string(),
        options,
        axis: row.axis,
    };
    let prompt = render(&null_row, labels);
    let null_forms = forms(engine, labels, merge_space)?;
    let prior = if let Some(picture) = picture {
        let marker = engine
            .media_marker()
            .ok_or_else(|| anyhow!("vision projector is not loaded"))?;
        let prompt = with_media(&prompt, marker);
        engine.decode_rgb(&prompt, &picture.pixels, picture.width, picture.height)?;
        slot_logits(engine.logits()?, &null_forms)
    } else {
        let tokens = engine.tokenize(&prompt, false)?;
        engine.decode(&tokens)?;
        slot_logits(engine.logits()?, &null_forms)
    };
    engine.prior_put(key, prior.clone());
    Ok(prior)
}

pub fn run(engine: &mut Engine, model_name: &str, req: Request) -> Result<Response> {
    let prep = prepare(engine, req)?;
    let (n_tokens, logits) = if let Some(picture) = &prep.picture {
        let n = engine.decode_rgb(&prep.prompt, &picture.pixels, picture.width, picture.height)?;
        (n, engine.logits()?.to_vec())
    } else {
        engine.decode(&prep.tokens)?;
        (prep.tokens.len(), engine.logits()?.to_vec())
    };
    decide_from_logits(
        model_name,
        &prep.row,
        n_tokens,
        &prep.forms,
        &logits,
        prep.prior.as_deref(),
    )
}

struct Prepared {
    row: Row,
    tokens: Vec<i32>,
    prompt: String,
    picture: Option<Rgb>,
    forms: Vec<Vec<i32>>,
    prior: Option<Vec<f32>>,
}

fn prepare(engine: &mut Engine, req: Request) -> Result<Prepared> {
    let picture = take_picture(&req)?;
    let question = req.questions.decision;
    let axis = axis_of(&question.kind);
    let options = options_from(&question.criteria, axis);
    if options.is_empty() {
        bail!("no options in questions.decision.criteria");
    }
    let row = Row {
        state: req.state.text(),
        question: question.instructions,
        options,
        axis,
    };
    let labels = labels_for(row.options.len())?;
    let mut prompt = render(&row, &labels);
    let tokens = if picture.is_some() {
        let marker = engine
            .media_marker()
            .ok_or_else(|| anyhow!("vision projector is not loaded"))?;
        prompt = with_media(&prompt, marker);
        Vec::new()
    } else {
        engine.tokenize(&prompt, false)?
    };
    let merge_space = row.axis != Axis::Score;
    let forms = forms(engine, &labels, merge_space)?;
    let prior = if prior_enabled() && row.axis == Axis::Choice {
        Some(null_prior_slots(engine, &row, &labels, merge_space, picture.as_ref())?)
    } else {
        None
    };
    Ok(Prepared {
        row,
        tokens,
        prompt,
        picture,
        forms,
        prior,
    })
}

fn take_picture(req: &Request) -> Result<Option<Rgb>> {
    if let Some(bytes) = &req.image_bytes {
        return Ok(Some(picture::decode_bytes(bytes)?));
    }
    if let Some(text) = req.image.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(Some(picture::decode_base64(text)?));
    }
    if let StateValue::Other(Value::Object(map)) = &req.state {
        for key in IMAGE_STATE_KEYS {
            if let Some(Value::String(text)) = map.get(*key)
                && !text.trim().is_empty() {
                    return Ok(Some(picture::decode_base64(text)?));
                }
        }
    }
    Ok(None)
}

fn request_has_image(req: &Request) -> bool {
    req.image_bytes.is_some()
        || req.image.as_deref().is_some_and(|s| !s.trim().is_empty())
        || matches!(&req.state, StateValue::Other(Value::Object(map)) if IMAGE_STATE_KEYS.iter().any(|k| map.contains_key(*k)))
}

fn decide_from_logits(
    model_name: &str,
    row: &Row,
    input_tokens: usize,
    label_forms: &[Vec<i32>],
    logits: &[f32],
    prior: Option<&[f32]>,
) -> Result<Response> {
    let mut slots = slot_logits(logits, label_forms);
    if let Some(p) = prior
        && p.len() == slots.len() {
            slots = apply_prior_calibration(&slots, p, prior_alpha());
        }
    let probs = softmax_slots(&slots);
    if probs.len() != row.options.len() {
        bail!("probability count does not match options");
    }

    let mut probabilities = Map::new();
    for (i, (key, _)) in row.options.iter().enumerate() {
        probabilities.insert(key.clone(), serde_json::json!(probs[i]));
    }
    let kind = match row.axis {
        Axis::Choice => "choice",
        Axis::Noul => "noul",
        Axis::Score => "score",
    };
    let mut decision = DecisionOut {
        kind,
        probabilities,
        choice: None,
        noul: None,
        probability: None,
    };
    if row.axis == Axis::Noul {
        let p_true = noul_true_probability(&row.options, &probs)?;
        decision.noul = Some(p_true);
        decision.probability = Some(p_true);
    } else {
        let argmax = positive_argmax(&probs)?;
        decision.choice = Some(row.options[argmax].0.clone());
    }
    Ok(Response {
        model: model_name.to_string(),
        answers: Answers { decision },
        usage: Usage {
            input_tokens,
            output_tokens: 1,
        },
    })
}

/// Score several requests. One request takes the single-sequence path.
/// A wider batch decodes independent prompts together.
pub fn run_many(engine: &mut Engine, model_name: &str, reqs: Vec<Request>) -> Vec<Result<Response>> {
    if reqs.len() == 1 || reqs.iter().any(request_has_image) {
        return reqs.into_iter().map(|req| run(engine, model_name, req)).collect();
    }
    let n = reqs.len();
    let mut slots: Vec<Option<Prepared>> = Vec::with_capacity(n);
    let mut results: Vec<Option<Result<Response>>> = (0..n).map(|_| None).collect();
    for req in reqs {
        match prepare(engine, req) {
            Ok(prep) => slots.push(Some(prep)),
            Err(err) => {
                results[slots.len()] = Some(Err(err));
                slots.push(None);
            }
        }
    }
    let ready_idx: Vec<usize> = slots.iter().enumerate().filter_map(|(i, p)| p.as_ref().map(|_| i)).collect();
    let seqs: Vec<Vec<i32>> = ready_idx.iter().map(|&i| slots[i].as_ref().unwrap().tokens.clone()).collect();
    let decoded = if seqs.is_empty() {
        Ok(Vec::new())
    } else {
        engine.decode_many(&seqs)
    };
    match decoded {
        Err(err) => {
            let message = err.to_string();
            for i in ready_idx {
                results[i] = Some(Err(anyhow!(message.clone())));
            }
        }
        Ok(rows) => {
            for (row, i) in rows.into_iter().zip(ready_idx) {
                let prep = slots[i].take().unwrap();
                results[i] = Some(decide_from_logits(
                    model_name,
                    &prep.row,
                    prep.tokens.len(),
                    &prep.forms,
                    &row,
                    prep.prior.as_deref(),
                ));
            }
        }
    }
    results.into_iter().map(|item| item.expect("every slot filled")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_criteria_are_shared_by_choice_and_noul() {
        let raw = r#"{"state":"s","questions":{"decision":{"type":"noul","instructions":"q","criteria":{"true":"yes","false":"no"}}}}"#;
        let req: Request = serde_json::from_str(raw).unwrap();
        let axis = axis_of(&req.questions.decision.kind);
        let options = options_from(&req.questions.decision.criteria, axis);
        assert_eq!(
            options,
            vec![("true".into(), "yes".into()), ("false".into(), "no".into())]
        );
    }

    #[test]
    fn criteria_and_state_keep_json_key_order() {
        let raw = r#"{"state":{"z":"2","a":"1"},"questions":{"decision":{"type":"choice","instructions":"q","criteria":{"z":"last","a":"first"}}}}"#;
        let req: Request = serde_json::from_str(raw).unwrap();
        let text = req.state.text();
        assert!(text.find("z:").unwrap() < text.find("a:").unwrap());
        let options = options_from(&req.questions.decision.criteria, Axis::Choice);
        assert_eq!(options[0].0, "z");
        assert_eq!(options[1].0, "a");
    }

    #[test]
    fn noul_without_true_key_is_an_error() {
        let options = vec![("a".into(), "x".into()), ("b".into(), "y".into())];
        let err = noul_true_probability(&options, &[0.2, 0.8]).unwrap_err();
        assert!(err.to_string().contains("true or yes"));
    }

    #[test]
    fn zero_probabilities_are_not_a_choice() {
        let err = positive_argmax(&[0.0, 0.0]).unwrap_err();
        assert!(err.to_string().contains("finite logit"));
        let options = vec![("true".into(), "yes".into())];
        let err = noul_true_probability(&options, &[0.0]).unwrap_err();
        assert!(err.to_string().contains("finite logit"));
    }

    #[test]
    fn score_uses_list_indexes() {
        let raw = r#"{"state":{"ticket":1},"questions":{"decision":{"type":"score","criteria":["low","high"]}}}"#;
        let req: Request = serde_json::from_str(raw).unwrap();
        assert!(req.state.text().contains("ticket"));
        let options = options_from(&req.questions.decision.criteria, Axis::Score);
        assert_eq!(options[0].0, "0");
        assert_eq!(options[1].1, "high");
    }
}
