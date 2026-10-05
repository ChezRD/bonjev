use crate::engine::Engine;
use anyhow::Result;

/// Last id of `full`, when `full` is exactly `lead` plus one token.
/// That token is `lead + space + letter` tokenized as one piece.
pub fn one_token_after(lead: &[i32], full: &[i32]) -> Option<i32> {
    let (&last, stem) = full.split_last()?;
    if stem == lead { Some(last) } else { None }
}

/// The decoded sequence ends with the lead's own token ids, and something
/// stands before that lead. A merged colon or newline fails this check.
pub fn lead_ends_sequence(tokens: &[i32], lead: &[i32]) -> bool {
    !lead.is_empty() && tokens.len() > lead.len() && tokens.ends_with(lead)
}

/// The prompt without the lead and the full prompt share a non-empty token
/// prefix, and the lead adds at least one token. A swallowed lead fails this.
pub fn lead_added_tokens(prefix: &[i32], full: &[i32]) -> bool {
    let mut n = 0;
    while n < prefix.len() && n < full.len() && prefix[n] == full[n] {
        n += 1;
    }
    n > 0 && n < full.len()
}

/// One id per letter: the single token added by `"{lead} {letter}"`.
pub fn letter_ids(engine: &Engine, lead: &str, labels: &[String]) -> Result<Vec<Vec<i32>>> {
    let lead_ids = engine.tokenize(lead, false)?;
    if lead_ids.is_empty() {
        anyhow::bail!("answer lead tokenizes to nothing");
    }
    let mut out = Vec::with_capacity(labels.len());
    let mut seen = Vec::with_capacity(labels.len());
    for label in labels {
        let full = engine.tokenize(&format!("{lead} {label}"), false)?;
        let Some(id) = one_token_after(&lead_ids, &full) else {
            anyhow::bail!("letter '{label}' is not one token after the lead");
        };
        if seen.contains(&id) {
            anyhow::bail!("letters share token id {id}");
        }
        seen.push(id);
        out.push(vec![id]);
    }
    Ok(out)
}

/// Index of the largest finite value; `None` when every value is non-finite.
/// Uses `total_cmp` so the order is total even with NaN, and skips non-finite
/// values so a NaN logit can never win the argmax.
pub fn argmax_finite(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
}

/// True when the raw argmax token is one of the label (option) tokens: the model
/// answers first instead of producing a reasoning/special token.
pub fn top_is_label(logits: &[f32], forms: &[Vec<i32>]) -> bool {
    let Some(top) = argmax_finite(logits) else {
        return false;
    };
    let top = top as i32;
    forms.iter().any(|ids| ids.contains(&top))
}

use std::sync::{Arc, Mutex};

static PRIOR: Mutex<Option<Arc<[f32]>>> = Mutex::new(None);

fn prior_lock() -> std::sync::MutexGuard<'static, Option<Arc<[f32]>>> {
    PRIOR
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Content-free (null prompt) full-vocabulary logits, used as the option prior.
/// Process-wide; reset with [`reset_prior`] when the engine is reloaded. A
/// reader keeps its own `Arc`, so a reset never frees the buffer underneath it.
pub fn set_prior(values: Vec<f32>) {
    *prior_lock() = Some(Arc::from(values));
}

/// Drop the cached prior so the next contextual-calibration run recomputes it.
pub fn reset_prior() {
    *prior_lock() = None;
}

pub fn prior() -> Option<Arc<[f32]>> {
    prior_lock().clone()
}

/// Contextual calibration is opt-in: `BONJEV_CONTEXT_CALIB=1`.
pub fn contextual_enabled() -> bool {
    std::env::var("BONJEV_CONTEXT_CALIB").map(|v| v == "1").unwrap_or(false)
}

/// Per-slot logit with the content-free prior subtracted per token id.
pub fn slot_logits_with_prior(logits: &[f32], prior: &[f32], forms: &[Vec<i32>]) -> Vec<f32> {
    forms
        .iter()
        .map(|ids| {
            let ls: Vec<f32> = ids
                .iter()
                .filter_map(|&i| {
                    let index = usize::try_from(i).ok()?;
                    let value = logits.get(index).copied()?;
                    Some(value - prior.get(index).copied().unwrap_or(0.0))
                })
                .collect();
            if ls.is_empty() {
                f32::NEG_INFINITY
            } else {
                logaddexp(&ls)
            }
        })
        .collect()
}

fn logaddexp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + values.iter().map(|x| (x - max).exp()).sum::<f32>().ln()
}

/// Per-slot logit = log-sum-exp over all token forms for that slot.
pub fn slot_logits(logits: &[f32], forms: &[Vec<i32>]) -> Vec<f32> {
    forms
        .iter()
        .map(|ids| {
            let ls: Vec<f32> = ids
                .iter()
                .filter_map(|&i| usize::try_from(i).ok().and_then(|index| logits.get(index)).copied())
                .collect();
            if ls.is_empty() {
                f32::NEG_INFINITY
            } else {
                logaddexp(&ls)
            }
        })
        .collect()
}

pub fn softmax_slots(vals: &[f32]) -> Vec<f32> {
    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; vals.len()];
    }
    let sum: f32 = vals.iter().map(|v| (v - max).exp()).sum();
    if sum == 0.0 {
        return vec![0.0; vals.len()];
    }
    vals.iter().map(|v| (v - max).exp() / sum).collect()
}

/// Softmax with a temperature: `softmax(vals / temp)`. `temp <= 0` or `temp == 1`
/// falls back to the plain softmax.
pub fn softmax_slots_scaled(vals: &[f32], temp: f32) -> Vec<f32> {
    if temp > 0.0 && (temp - 1.0).abs() > f32::EPSILON {
        let scaled: Vec<f32> = vals.iter().map(|v| v / temp).collect();
        softmax_slots(&scaled)
    } else {
        softmax_slots(vals)
    }
}

/// Merge full-vocabulary logits onto the label set and normalise.
#[cfg(test)]
pub fn probs_from_logits(logits: &[f32], forms: &[Vec<i32>]) -> Vec<f32> {
    softmax_slots(&slot_logits(logits, forms))
}

#[cfg(test)]
mod tests {
    use super::{lead_added_tokens, lead_ends_sequence, one_token_after, probs_from_logits};

    #[test]
    fn one_token_after_keeps_only_the_last_id() {
        assert_eq!(one_token_after(&[1, 2], &[1, 2, 9]), Some(9));
        assert_eq!(one_token_after(&[1, 2], &[1, 3, 9]), None);
        assert_eq!(one_token_after(&[1, 2], &[1, 2]), None);
        assert_eq!(one_token_after(&[1], &[1, 2, 3]), None);
    }

    #[test]
    fn lead_must_end_the_sequence_and_add_tokens() {
        assert!(lead_ends_sequence(&[7, 1, 2], &[1, 2]));
        assert!(!lead_ends_sequence(&[7, 1, 3], &[1, 2]));
        assert!(!lead_ends_sequence(&[1, 2], &[1, 2]));
        assert!(!lead_ends_sequence(&[7, 1, 2], &[]));
        assert!(lead_added_tokens(&[1, 2, 3], &[1, 2, 3, 4, 5]));
        assert!(!lead_added_tokens(&[1, 2, 3], &[1, 2, 3]));
        assert!(!lead_added_tokens(&[1, 2, 3], &[9, 2, 3, 4]));
    }

    #[test]
    fn equal_logits_split_evenly() {
        let mut logits = vec![0.0; 4];
        logits[1] = 1.0;
        logits[2] = 1.0;
        let probs = probs_from_logits(&logits, &[vec![1], vec![2]]);
        assert!((probs[0] - 0.5).abs() < 1e-5);
        assert!((probs[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn missing_token_ids_are_zero() {
        let probs = probs_from_logits(&[0.0, 1.0], &[vec![9], vec![8]]);
        assert_eq!(probs, vec![0.0, 0.0]);
    }
}
