use crate::engine::Engine;
use anyhow::Result;

fn push_single_token(ids: &mut Vec<i32>, engine: &Engine, text: &str) -> Result<()> {
    let bare = engine.tokenize(text, false)?;
    if bare.len() == 1 && !ids.contains(&bare[0]) {
        ids.push(bare[0]);
    }
    Ok(())
}

fn push_token_variants(
    ids: &mut Vec<i32>,
    engine: &Engine,
    variants: &[&str],
    merge_space: bool,
) -> Result<()> {
    for v in variants {
        push_single_token(ids, engine, v)?;
        if merge_space {
            push_single_token(ids, engine, &format!(" {v}"))?;
        }
    }
    Ok(())
}

/// Token ids per label: bare form and, when merging, the " X" form.
pub fn forms(engine: &Engine, labels: &[String], merge_space: bool) -> Result<Vec<Vec<i32>>> {
    let mut out = Vec::with_capacity(labels.len());
    for lab in labels {
        let mut ids = Vec::new();
        push_token_variants(&mut ids, engine, &[lab.as_str()], merge_space)?;
        out.push(ids);
    }
    Ok(out)
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
                .filter_map(|&i| logits.get(i as usize).copied())
                .collect();
            if ls.is_empty() {
                f32::NEG_INFINITY
            } else {
                logaddexp(&ls)
            }
        })
        .collect()
}

pub fn apply_prior_calibration(slots: &[f32], prior: &[f32], alpha: f32) -> Vec<f32> {
    slots
        .iter()
        .zip(prior.iter())
        .map(|(z, z0)| z - alpha * z0)
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

/// Merge full-vocabulary logits onto the label set and normalise.
#[cfg(test)]
pub fn probs_from_logits(logits: &[f32], forms: &[Vec<i32>]) -> Vec<f32> {
    softmax_slots(&slot_logits(logits, forms))
}

#[cfg(test)]
mod tests {
    use super::probs_from_logits;

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
