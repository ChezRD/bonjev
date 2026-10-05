use crate::prompt::Axis;
use serde::de::{self, DeserializeOwned, Deserializer};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;

pub use crate::prompt::{IMAGE_STATE_KEYS, StateValue};

#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub state: StateValue,
    pub questions: ObjMap<Question>,
    /// Base64 image, or a data URL. jpeg, png, gif, webp or bmp.
    #[serde(default)]
    pub image: Option<String>,
    /// Raw file bytes from the CLI. Not part of the JSON body.
    #[serde(skip)]
    pub image_bytes: Option<Vec<u8>>,
    /// Prompt style for this request. One name, `top3`, or up to three names
    /// separated by commas. `top3` is `w2_word3_desc`, `native_user`, and
    /// `native_user_theanswer`, scored together. Unset uses `BONJEV_STYLE`, then `answer`.
    #[serde(default)]
    pub prompt_style: Option<String>,
}

impl Request {
    pub(super) fn state_image(&self) -> Option<&str> {
        let StateValue::Other(Value::Object(map)) = &self.state else {
            return None;
        };
        for key in IMAGE_STATE_KEYS {
            if let Some(Value::String(text)) = map.get(*key) {
                let text = text.trim();
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Question {
    /// Missing `type` is choice. `boolean` is accepted and answered as `noul`.
    #[serde(rename = "type", default)]
    pub kind: String,
    /// What the model should decide. A string, an object, an array, or null,
    /// exactly as the TypeSafe contract allows.
    #[serde(default)]
    pub instructions: Value,
    /// Alias used by the original System One body when `instructions` is empty.
    #[serde(default)]
    pub criterion: Value,
    #[serde(default)]
    pub criteria: Criteria,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QuestionKind {
    Choice,
    Score,
    Noul,
}

impl QuestionKind {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "" | "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" | "boolean" => Some(Self::Noul),
            _ => None,
        }
    }

    pub(super) fn axis(self) -> Axis {
        match self {
            Self::Choice => Axis::Choice,
            Self::Score => Axis::Score,
            Self::Noul => Axis::Noul,
        }
    }
}

/// JSON object that keeps the key order from the request.
/// `serde_json::Map` only implements serde for `Value`, and `BTreeMap` sorts keys.
#[derive(Clone, Debug)]
pub(crate) struct ObjMap<T>(Vec<(String, T)>);

impl<T> ObjMap<T> {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    fn from_pair(key: String, value: T) -> Self {
        Self(vec![(key, value)])
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &(String, T)> {
        self.0.iter()
    }

    #[cfg(test)]
    pub(super) fn get(&self, key: &str) -> Option<&T> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub(super) fn insert(&mut self, key: String, value: T) {
        self.0.push((key, value));
    }
}

impl<T> IntoIterator for ObjMap<T> {
    type Item = (String, T);
    type IntoIter = std::vec::IntoIter<(String, T)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'de, T> Deserialize<'de> for ObjMap<T>
where
    T: DeserializeOwned,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Map::<String, Value>::deserialize(deserializer)?;
        let mut out = Vec::with_capacity(raw.len());
        for (key, value) in raw {
            let item = serde_json::from_value(value).map_err(de::Error::custom)?;
            out.push((key, item));
        }
        Ok(Self(out))
    }
}

impl<T: Serialize> Serialize for ObjMap<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

#[derive(Clone, Debug, Deserialize)]
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

pub(super) fn question_value(question: &Question) -> Value {
    if !crate::yaml_emit::is_blank(&question.instructions) {
        return question.instructions.clone();
    }
    if !crate::yaml_emit::is_blank(&question.criterion) {
        return question.criterion.clone();
    }
    Value::String(String::new())
}

#[derive(Debug)]
pub struct WireError {
    field: String,
    message: String,
}

impl WireError {
    pub(super) fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }

    pub fn field(&self) -> &str {
        &self.field
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for WireError {}

#[derive(Debug)]
pub enum RunError {
    Wire(WireError),
    Engine(anyhow::Error),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wire(err) => write!(f, "{err}"),
            Self::Engine(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Wire(err) => Some(err),
            Self::Engine(err) => Some(err.as_ref()),
        }
    }
}

impl From<WireError> for RunError {
    fn from(err: WireError) -> Self {
        Self::Wire(err)
    }
}

impl From<anyhow::Error> for RunError {
    fn from(err: anyhow::Error) -> Self {
        Self::Engine(err)
    }
}

#[cfg(test)]
pub(super) fn check_questions(req: &Request) -> Result<(), WireError> {
    if req.questions.is_empty() {
        return Err(WireError::new("questions", "questions is empty"));
    }
    for (qid, question) in req.questions.iter() {
        checked_options(qid, question)?;
    }
    Ok(())
}

pub(super) fn checked_options(
    qid: &str,
    question: &Question,
) -> Result<(QuestionKind, Vec<(String, Value)>), WireError> {
    let Some(kind) = QuestionKind::parse(&question.kind) else {
        return Err(WireError::new(
            format!("questions.{qid}.type"),
            format!("unknown question type '{}'", question.kind),
        ));
    };
    let options = match kind {
        QuestionKind::Choice => choice_options(qid, &question.criteria)?,
        QuestionKind::Score => score_options(qid, &question.criteria)?,
        QuestionKind::Noul => noul_options(qid, &question.criteria)?,
    };
    Ok((kind, options))
}

fn choice_options(qid: &str, criteria: &Criteria) -> Result<Vec<(String, Value)>, WireError> {
    let Criteria::Map(map) = criteria else {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            "choice criteria is a map of options",
        ));
    };
    if map.len() < 2 {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            format!("a choice needs at least two options (got {})", map.len()),
        ));
    }
    if map.len() > crate::prompt::MAX_LABELS {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            format!(
                "options per choice is limited to {} (got {})",
                crate::prompt::MAX_LABELS,
                map.len()
            ),
        ));
    }
    Ok(map
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

fn score_options(qid: &str, criteria: &Criteria) -> Result<Vec<(String, Value)>, WireError> {
    let Criteria::List(items) = criteria else {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            "score criteria is a list of levels",
        ));
    };
    if !(2..=10).contains(&items.len()) {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            format!("a score takes 2 to 10 levels (got {})", items.len()),
        ));
    }
    Ok(items
        .iter()
        .enumerate()
        .map(|(index, value)| (index.to_string(), value.clone()))
        .collect())
}

fn noul_options(qid: &str, criteria: &Criteria) -> Result<Vec<(String, Value)>, WireError> {
    let mut true_desc = Value::String("true".to_string());
    let mut false_desc = Value::String("false".to_string());
    match criteria {
        Criteria::Other(Value::Null) => {}
        Criteria::Map(map) => {
            for (key, value) in map {
                match key.as_str() {
                    "true" => true_desc = value.clone(),
                    "false" => false_desc = value.clone(),
                    _ => {
                        return Err(WireError::new(
                            format!("questions.{qid}.criteria"),
                            "noul criteria only allows true and false",
                        ))
                    }
                }
            }
        }
        _ => {
            return Err(WireError::new(
                format!("questions.{qid}.criteria"),
                "noul criteria is a true/false map",
            ))
        }
    }
    Ok(vec![
        ("true".to_string(), true_desc),
        ("false".to_string(), false_desc),
    ])
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: ObjMap<AnswerOut>,
    pub usage: Usage,
}

#[derive(Debug)]
pub enum AnswerOut {
    Choice {
        choice: String,
        probabilities: Map<String, Value>,
        confidence: f32,
        margin: f32,
        entropy: f32,
        answer_first: Option<bool>,
    },
    Score {
        score: f32,
        legend: ObjMap<Value>,
        probabilities: Map<String, Value>,
        confidence: f32,
        margin: f32,
        entropy: f32,
        answer_first: Option<bool>,
    },
    Noul {
        noul: f32,
        margin: f32,
        entropy: f32,
        answer_first: Option<bool>,
    },
}

impl AnswerOut {
    pub fn set_answer_first(&mut self, value: Option<bool>) {
        match self {
            Self::Choice { answer_first, .. }
            | Self::Score { answer_first, .. }
            | Self::Noul { answer_first, .. } => *answer_first = value,
        }
    }
}

impl Serialize for AnswerOut {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        match self {
            Self::Choice {
                choice,
                probabilities,
                confidence,
                margin,
                entropy,
                answer_first,
            } => {
                map.serialize_entry("type", "choice")?;
                map.serialize_entry("choice", choice)?;
                if !probabilities.is_empty() {
                    map.serialize_entry("probabilities", probabilities)?;
                }
                map.serialize_entry("confidence", confidence)?;
                map.serialize_entry("margin", margin)?;
                map.serialize_entry("entropy", entropy)?;
                if let Some(value) = answer_first {
                    map.serialize_entry("answer_first", value)?;
                }
            }
            Self::Score {
                score,
                legend,
                probabilities,
                confidence,
                margin,
                entropy,
                answer_first,
            } => {
                map.serialize_entry("type", "score")?;
                if !probabilities.is_empty() {
                    map.serialize_entry("probabilities", probabilities)?;
                }
                map.serialize_entry("confidence", confidence)?;
                map.serialize_entry("margin", margin)?;
                map.serialize_entry("entropy", entropy)?;
                if let Some(value) = answer_first {
                    map.serialize_entry("answer_first", value)?;
                }
                map.serialize_entry("score", score)?;
                map.serialize_entry("legend", legend)?;
            }
            Self::Noul {
                noul,
                margin,
                entropy,
                answer_first,
            } => {
                map.serialize_entry("type", "noul")?;
                map.serialize_entry("noul", noul)?;
                map.serialize_entry("margin", margin)?;
                map.serialize_entry("entropy", entropy)?;
                if let Some(value) = answer_first {
                    map.serialize_entry("answer_first", value)?;
                }
            }
        }
        map.end()
    }
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: u32,
    /// Prompt tokens reused from KV cache, not recomputed in this call.
    pub cache_hit_tokens: usize,
    /// Prompt tokens actually prefilled in this call.
    pub cache_miss_tokens: usize,
}

pub fn from_options(state: &str, question: &str, kind: &str, options: &[String]) -> Request {
    let criteria = if kind == "score" {
        Criteria::List(options.iter().cloned().map(Value::String).collect())
    } else if kind == "noul" || kind == "boolean" {
        Criteria::default()
    } else {
        let mut map = Map::new();
        for (index, text) in options.iter().enumerate() {
            map.insert(format!("option_{index}"), Value::String(text.clone()));
        }
        Criteria::Map(map)
    };
    Request {
        state: StateValue::Text(state.to_string()),
        questions: ObjMap::from_pair(
            "decision".to_string(),
            Question {
                kind: kind.to_string(),
                instructions: Value::String(question.to_string()),
                criterion: Value::Null,
                criteria,
            },
        ),
        image: None,
        image_bytes: None,
        prompt_style: None,
    }
}
