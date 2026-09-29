use crate::prompt::{Axis, Row, prior_alpha};
use crate::readout::{apply_prior_calibration, slot_logits, softmax_slots};
use anyhow::{Result, anyhow, bail};

use super::wire::{AnswerOut, ObjMap, RunError};

/// `(max - 1/k) / (1 - 1/k)` over the option distribution.
fn jev_confidence(probs: &[f32]) -> f32 {
    let k = probs.len() as f32;
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (max - 1.0 / k) / (1.0 - 1.0 / k)
}

pub(super) fn positive_argmax(probs: &[f32]) -> Result<usize> {
    if !probs.iter().any(|p| p.is_finite() && *p > 0.0) {
        bail!("no label token produced a finite logit");
    }
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .ok_or_else(|| anyhow!("no label token produced a finite logit"))
}

pub(super) fn noul_true_probability(options: &[(String, String)], probs: &[f32]) -> Result<f32> {
    if !probs.iter().any(|p| p.is_finite() && *p > 0.0) {
        bail!("no label token produced a finite logit");
    }
    let index = options
        .iter()
        .position(|(key, _)| key.eq_ignore_ascii_case("true") || key.eq_ignore_ascii_case("yes"))
        .ok_or_else(|| anyhow!("noul criteria need a true or yes key"))?;
    probs
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("probability count does not match options"))
}

pub(super) fn pack_answer(
    row: &Row,
    label_forms: &[Vec<i32>],
    logits: &[f32],
    prior: Option<&[f32]>,
) -> Result<AnswerOut, RunError> {
    let mut slots = slot_logits(logits, label_forms);
    if let Some(prior_slots) = prior
        && prior_slots.len() == slots.len()
    {
        slots = apply_prior_calibration(&slots, prior_slots, prior_alpha());
    }
    let probs = softmax_slots(&slots);
    if probs.len() != row.options.len() {
        return Err(anyhow!("probability count does not match options").into());
    }

    if row.axis == Axis::Noul {
        let p_true = noul_true_probability(&row.options, &probs)?;
        return Ok(AnswerOut {
            kind: "noul",
            choice: None,
            probabilities: serde_json::Map::new(),
            noul: Some(p_true),
            confidence: None,
            score: None,
            legend: None,
        });
    }

    let mut probabilities = serde_json::Map::new();
    for (index, (key, _)) in row.options.iter().enumerate() {
        probabilities.insert(key.clone(), serde_json::json!(probs[index]));
    }
    let confidence = jev_confidence(&probs);
    if row.axis == Axis::Score {
        let score = probs
            .iter()
            .enumerate()
            .map(|(index, probability)| index as f32 * probability)
            .sum();
        let mut legend = ObjMap::new();
        for (key, text) in &row.options {
            legend.insert(key.clone(), text.clone());
        }
        return Ok(AnswerOut {
            kind: "score",
            choice: None,
            probabilities,
            noul: None,
            confidence: Some(confidence),
            score: Some(score),
            legend: Some(legend),
        });
    }
    let argmax = positive_argmax(&probs)?;
    Ok(AnswerOut {
        kind: "choice",
        choice: Some(row.options[argmax].0.clone()),
        probabilities,
        noul: None,
        confidence: Some(confidence),
        score: None,
        legend: None,
    })
}
