use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};

use crate::prompt::Axis;

/// Post-hoc readout calibration: temperature and per-slot logit bias.
///
/// `BONJEV_TEMP=<f32>` sets a global temperature (softmax(vals / temp)).
/// `BONJEV_BIAS=<path.json>` adds a per-slot logit bias; the file is either a
/// bare array `[b0, b1, ...]` or an object `{"bias": [...], "temp": 1.2,
/// "per_axis": {"choice": 1.1}}`. Temperature scales probabilities (argmax is
/// unchanged); bias can move the argmax and is the real lever on the exam.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub temp: f32,
    pub bias: Vec<f32>,
    pub per_axis_temp: HashMap<Axis, f32>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            temp: 1.0,
            bias: Vec::new(),
            per_axis_temp: HashMap::new(),
        }
    }
}

fn positive_finite(value: f32, what: &str) -> Result<f32> {
    if value > 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err(anyhow!("{what} must be a positive, finite number"))
    }
}

fn positive_number(value: &serde_json::Value, what: &str) -> Result<f32> {
    let number = value
        .as_f64()
        .ok_or_else(|| anyhow!("{what} is not a number"))?;
    positive_finite(number as f32, what)
}

/// Parse a JSON array of numbers. Fails on the first non-number element instead
/// of dropping it: a hole would shift every later per-slot bias.
fn floats(value: &serde_json::Value, what: &str) -> Result<Vec<f32>> {
    let items = value
        .as_array()
        .ok_or_else(|| anyhow!("{what} is not an array"))?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| anyhow!("{what}[{index}] is not a number"))
        })
        .collect()
}

fn axis_from_key(key: &str) -> Result<Axis> {
    match key {
        "choice" => Ok(Axis::Choice),
        "noul" => Ok(Axis::Noul),
        "score" => Ok(Axis::Score),
        other => Err(anyhow!("unknown per_axis key '{other}'")),
    }
}

impl Calibration {
    /// Read calibration from `BONJEV_TEMP` and `BONJEV_BIAS`. A bad value is an
    /// error, not a silent fallback to the defaults.
    pub fn from_env() -> Result<Self> {
        let mut calib = Self::default();
        if let Ok(text) = std::env::var("BONJEV_TEMP") {
            let value = text
                .trim()
                .parse::<f64>()
                .with_context(|| format!("BONJEV_TEMP={text:?} is not a number"))?;
            calib.temp = positive_finite(value as f32, "BONJEV_TEMP")?;
        }
        let Ok(path) = std::env::var("BONJEV_BIAS") else {
            return Ok(calib);
        };
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read BONJEV_BIAS={path}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parse BONJEV_BIAS={path}"))?;
        if value.is_array() {
            calib.bias = floats(&value, "BONJEV_BIAS")?;
            return Ok(calib);
        }
        if let Some(bias) = value.get("bias") {
            calib.bias = floats(bias, "bias")?;
        }
        if let Some(temp) = value.get("temp") {
            calib.temp = positive_number(temp, "temp")?;
        }
        if let Some(map) = value.get("per_axis") {
            let map = map
                .as_object()
                .ok_or_else(|| anyhow!("per_axis is not an object"))?;
            for (key, item) in map {
                let axis = axis_from_key(key)?;
                calib.per_axis_temp.insert(axis, positive_number(item, key)?);
            }
        }
        Ok(calib)
    }

    pub fn apply_bias<'a>(&self, raw: &'a [f32]) -> Cow<'a, [f32]> {
        if self.bias.is_empty() {
            return Cow::Borrowed(raw);
        }
        let mut out = raw.to_vec();
        for (index, value) in out.iter_mut().enumerate() {
            if let Some(bias) = self.bias.get(index) {
                *value += bias;
            }
        }
        Cow::Owned(out)
    }

    pub fn temp_for(&self, axis: Axis) -> f32 {
        self.per_axis_temp
            .get(&axis)
            .copied()
            .filter(|t| *t > 0.0)
            .unwrap_or(self.temp)
    }
}

/// Process-wide calibration, read once from the environment. A bad value is
/// reported once and ignored, so the defaults are used.
pub fn global() -> &'static Calibration {
    static CALIB: OnceLock<Calibration> = OnceLock::new();
    CALIB.get_or_init(|| match Calibration::from_env() {
        Ok(calib) => calib,
        Err(err) => {
            eprintln!("[bonjev] calibration ignored: {err:#}; using defaults");
            Calibration::default()
        }
    })
}
