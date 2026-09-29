use crate::prompt::Axis;
use serde::de::{self, DeserializeOwned, Deserializer};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;

pub use crate::prompt::{IMAGE_STATE_KEYS, StateValue};

#[derive(Debug, Deserialize)]
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
    /// Choice scores subtract a fraction of the null-prompt prior.
    /// Missing field means calibrated. `false` scores the raw choice logits.
    #[serde(default = "default_calibrated")]
    pub scores_are_calibrated: bool,
}

fn default_calibrated() -> bool {
    true
}

impl Request {
    pub(super) fn has_image(&self) -> bool {
        self.image_bytes.is_some()
            || self
                .image
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
            || self.state_image().is_some()
    }

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

#[derive(Debug, Deserialize)]
pub struct Question {
    /// Missing `type` is choice. `boolean` is accepted and answered as `noul`.
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub instructions: String,
    /// Alias used by the original System One body when `instructions` is empty.
    #[serde(default)]
    pub criterion: String,
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
#[derive(Debug)]
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

pub(super) fn question_text(question: &Question) -> String {
    let instructions = question.instructions.trim();
    if !instructions.is_empty() {
        return instructions.to_string();
    }
    question.criterion.trim().to_string()
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
) -> Result<(QuestionKind, Vec<(String, String)>), WireError> {
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

fn choice_options(qid: &str, criteria: &Criteria) -> Result<Vec<(String, String)>, WireError> {
    let Criteria::Map(map) = criteria else {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            "choice criteria is a map of options",
        ));
    };
    if !(2..=255).contains(&map.len()) {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            "choice needs 2 to 255 options",
        ));
    }
    Ok(map
        .iter()
        .map(|(key, value)| (key.clone(), criterion_text(value)))
        .collect())
}

fn score_options(qid: &str, criteria: &Criteria) -> Result<Vec<(String, String)>, WireError> {
    let Criteria::List(items) = criteria else {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            "score criteria is a list of levels",
        ));
    };
    if !(2..=10).contains(&items.len()) {
        return Err(WireError::new(
            format!("questions.{qid}.criteria"),
            "score needs 2 to 10 levels",
        ));
    }
    let mut options = Vec::with_capacity(items.len());
    for (index, value) in items.iter().enumerate() {
        let Value::String(text) = value else {
            return Err(WireError::new(
                format!("questions.{qid}.criteria"),
                "score levels are strings",
            ));
        };
        options.push((index.to_string(), text.clone()));
    }
    Ok(options)
}

fn noul_options(qid: &str, criteria: &Criteria) -> Result<Vec<(String, String)>, WireError> {
    match criteria {
        Criteria::Other(Value::Null) => Ok(vec![
            ("true".to_string(), "true".to_string()),
            ("false".to_string(), "false".to_string()),
        ]),
        Criteria::Map(map) => {
            if !map.contains_key("true") {
                return Err(WireError::new(
                    format!("questions.{qid}.criteria"),
                    "noul criteria need a true key",
                ));
            }
            if map.keys().any(|key| key != "true" && key != "false") {
                return Err(WireError::new(
                    format!("questions.{qid}.criteria"),
                    "noul criteria only allows true and false",
                ));
            }
            Ok(map
                .iter()
                .map(|(key, value)| (key.clone(), criterion_text(value)))
                .collect())
        }
        _ => Err(WireError::new(
            format!("questions.{qid}.criteria"),
            "noul criteria is a true/false map",
        )),
    }
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: ObjMap<AnswerOut>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct AnswerOut {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub probabilities: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noul: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legend: Option<ObjMap<String>>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: u32,
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
                instructions: question.to_string(),
                criterion: String::new(),
                criteria,
            },
        ),
        image: None,
        image_bytes: None,
        scores_are_calibrated: true,
    }
}
