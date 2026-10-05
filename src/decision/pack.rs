use crate::prompt::{Axis, Row};
use crate::readout::{argmax_finite, slot_logits, softmax_slots_scaled};
use anyhow::{Result, anyhow, bail};
use serde_json::Value;

use super::wire::{AnswerOut, ObjMap, RunError};

/// `(max - 1/k) / (1 - 1/k)` over the option distribution.
fn jev_confidence(probs: &[f32]) -> f32 {
    let k = probs.len() as f32;
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (max - 1.0 / k) / (1.0 - 1.0 / k)
}

/// `(top1 - top2, -sum p ln p)` over the option distribution.
fn margin_entropy(probs: &[f32]) -> (f32, f32) {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let margin = if sorted.len() >= 2 {
        sorted[0] - sorted[1]
    } else {
        0.0
    };
    let entropy = -probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| p * p.ln())
        .sum::<f32>();
    (margin, entropy)
}

pub(super) fn positive_argmax(probs: &[f32]) -> Result<usize> {
    argmax_finite(probs)
        .filter(|index| probs[*index] > 0.0)
        .ok_or_else(|| anyhow!("no label token produced a finite logit"))
}

pub(super) fn noul_true_probability(options: &[(String, Value)], probs: &[f32]) -> Result<f32> {
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

pub(super) fn option_probs(label_forms: &[Vec<i32>], logits: &[f32], axis: Axis) -> Vec<f32> {
    let calib = crate::decision::calib::global();
    let raw = if crate::readout::contextual_enabled() {
        match crate::readout::prior() {
            Some(prior) => crate::readout::slot_logits_with_prior(logits, &prior, label_forms),
            None => slot_logits(logits, label_forms),
        }
    } else {
        slot_logits(logits, label_forms)
    };
    let biased = calib.apply_bias(&raw);
    softmax_slots_scaled(&biased, calib.temp_for(axis))
}

/// Mean of per-style option distributions. Slot `i` is the same wire option in every pass.
pub(super) fn average_passes(passes: &[Vec<f32>]) -> Result<Vec<f32>> {
    let Some(first) = passes.first() else {
        bail!("ensemble has no passes");
    };
    let mut sum = vec![0.0f32; first.len()];
    for pass in passes {
        if pass.len() != sum.len() {
            bail!("ensemble passes do not share an option count");
        }
        for (slot, value) in sum.iter_mut().zip(pass) {
            *slot += value;
        }
    }
    let width = passes.len() as f32;
    for value in &mut sum {
        *value /= width;
    }
    Ok(sum)
}

/// How several style passes of one question are merged.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EnsembleMode {
    /// Mean of the option distributions (default).
    Avg,
    /// Majority of each pass's argmax; ties fall back to the mean.
    Vote,
    /// Mean weighted by each pass's top1-top2 margin.
    Weighted,
}

/// `BONJEV_ENSEMBLE_MODE=avg|vote|weighted` (default `avg`).
pub(super) fn ensemble_mode() -> EnsembleMode {
    match std::env::var("BONJEV_ENSEMBLE_MODE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("vote") => EnsembleMode::Vote,
        Some("weighted") => EnsembleMode::Weighted,
        _ => EnsembleMode::Avg,
    }
}

fn pass_margin(probs: &[f32]) -> f32 {
    let mut top = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &value in probs {
        if value > top {
            second = top;
            top = value;
        } else if value > second {
            second = value;
        }
    }
    if second.is_finite() {
        (top - second).max(0.0)
    } else {
        0.0
    }
}

/// Merge passes into one distribution plus an optional explicit winner slot
/// (vote mode). Weighted and vote modes still report the merged distribution;
/// only vote needs the explicit slot because the majority can differ from the
/// mean's argmax.
pub(super) fn combine_passes(
    passes: &[Vec<f32>],
    mode: EnsembleMode,
) -> Result<(Vec<f32>, Option<usize>)> {
    let mean = average_passes(passes)?;
    match mode {
        EnsembleMode::Avg => Ok((mean, None)),
        EnsembleMode::Weighted => {
            let mut weighted = vec![0.0f32; mean.len()];
            let mut total = 0.0f32;
            for pass in passes {
                let weight = pass_margin(pass);
                total += weight;
                for (slot, value) in weighted.iter_mut().zip(pass) {
                    *slot += value * weight;
                }
            }
            if total <= 0.0 {
                return Ok((mean, None));
            }
            for value in &mut weighted {
                *value /= total;
            }
            Ok((weighted, None))
        }
        EnsembleMode::Vote => {
            let mut votes = vec![0usize; mean.len()];
            for pass in passes {
                let winner = argmax_finite(pass);
                if let Some(index) = winner {
                    votes[index] += 1;
                }
            }
            let best = votes.iter().copied().max().unwrap_or(0);
            let winners: Vec<usize> = votes
                .iter()
                .enumerate()
                .filter(|(_, count)| **count == best)
                .map(|(index, _)| index)
                .collect();
            if winners.len() == 1 {
                Ok((mean, Some(winners[0])))
            } else {
                Ok((mean, None))
            }
        }
    }
}

#[cfg(test)]
pub(super) fn pack_from_probs(row: &Row, probs: &[f32]) -> Result<AnswerOut, RunError> {
    pack_from_probs_with(row, probs, None)
}

/// `choice` overrides the argmax slot for choice and yes/no answers (vote mode).
pub(super) fn pack_from_probs_with(
    row: &Row,
    probs: &[f32],
    choice: Option<usize>,
) -> Result<AnswerOut, RunError> {
    if probs.len() != row.options.len() {
        return Err(anyhow!("probability count does not match options").into());
    }
    let (margin, entropy) = margin_entropy(probs);

    if row.axis == Axis::Noul {
        let mut p_true = noul_true_probability(&row.options, probs)?;
        if let Some(index) = choice {
            let winner_is_true = row
                .options
                .get(index)
                .map(|(key, _)| key.eq_ignore_ascii_case("true") || key.eq_ignore_ascii_case("yes"))
                .unwrap_or(true);
            if winner_is_true != (p_true >= 0.5) {
                p_true = 1.0 - p_true;
            }
        }
        return Ok(AnswerOut::Noul {
            noul: p_true,
            margin,
            entropy,
            answer_first: None,
        });
    }

    let mut probabilities = serde_json::Map::new();
    for (index, (key, _)) in row.options.iter().enumerate() {
        probabilities.insert(key.clone(), serde_json::json!(probs[index]));
    }
    let confidence = jev_confidence(probs);
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
        return Ok(AnswerOut::Score {
            score,
            legend,
            probabilities,
            confidence,
            margin,
            entropy,
            answer_first: None,
        });
    }
    let argmax = match choice {
        Some(index) if index < probs.len() => index,
        _ => positive_argmax(probs)?,
    };
    Ok(AnswerOut::Choice {
        choice: row.options[argmax].0.clone(),
        probabilities,
        confidence,
        margin,
        entropy,
        answer_first: None,
    })
}

#[cfg(test)]
pub(super) fn pack_answer(
    row: &Row,
    label_forms: &[Vec<i32>],
    logits: &[f32],
) -> Result<AnswerOut, RunError> {
    pack_from_probs(row, &option_probs(label_forms, logits, row.axis))
}
