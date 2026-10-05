//! YAML through `yaml_serde`, the maintained `serde_yaml` fork.
//! `state` is always a literal block. Question and options stay YAML nodes.

use serde_json::{Map, Value};

use crate::prompt::StateValue;

/// State object keys that hold raw image payloads. They ride along outside
/// the prompt: skipped when rendering, read separately for vision input.
pub const IMAGE_STATE_KEYS: &[&str] = &[
    "image",
    "images",
    "__media__",
    "__images__",
    "frame",
    "frames",
];

/// The state as one node. A string stays the raw source. It is not parsed
/// as YAML, so a ticket that looks like a mapping stays a string.
pub fn state_node(state: &StateValue) -> Value {
    match state {
        StateValue::Text(text) => Value::String(text.clone()),
        StateValue::Other(value) => value.clone(),
    }
}

/// The state node without raw image payloads.
pub fn state_node_public(state: &StateValue) -> Value {
    match state_node(state) {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(key, _)| !IMAGE_STATE_KEYS.contains(&key.as_str()))
                .collect(),
        ),
        node => node,
    }
}

/// Blank instructions/criterion: null, an empty string, or an empty
/// mapping/sequence. Numbers and booleans always count as content.
pub fn is_blank(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.trim().is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

/// The user turn: `state` is always a literal block, then `question` and
/// `options`. The shared state head stays first so one request reuses it.
pub fn user_doc(
    state: &Value,
    question: &Value,
    options: &[(String, Value)],
    labels: &[String],
) -> anyhow::Result<String> {
    if labels.len() != options.len() {
        anyhow::bail!(
            "answer labels ({}) do not match options ({})",
            labels.len(),
            options.len()
        );
    }
    let mut opts = Map::with_capacity(options.len());
    for (label, (_, value)) in labels.iter().zip(options.iter()) {
        opts.insert(label.clone(), value.clone());
    }
    let mut tail = Map::with_capacity(2);
    tail.insert("question".to_string(), question.clone());
    tail.insert("options".to_string(), Value::Object(opts));
    let tail_yaml =
        yaml_serde::to_string(&Value::Object(tail)).expect("a JSON value always renders as YAML");
    let mut doc = literal_block("state", &literal_text(state));
    doc.push_str(tail_yaml.trim_end());
    Ok(doc)
}

/// Text inside `state: |`. A string is copied. A JSON value is rendered as
/// YAML text so the block stays a scalar, not a nested mapping.
fn literal_text(state: &Value) -> String {
    match state {
        Value::String(text) => text.clone(),
        other => yaml_serde::to_string(other)
            .expect("a JSON value always renders as YAML")
            .trim_end_matches('\n')
            .to_string(),
    }
}

fn literal_block(key: &str, text: &str) -> String {
    let mut out = format!("{key}: |\n");
    if text.is_empty() {
        return out;
    }
    let body = text.strip_suffix('\n').unwrap_or(text);
    for line in body.split('\n') {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    out
}
