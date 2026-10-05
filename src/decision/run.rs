use std::sync::Arc;

use crate::engine::Engine;
use crate::picture::{self, Rgb};
use crate::prompt::{Row, Style, auto_style, labels_for, render_style, styles_for, with_media};
use crate::readout::{lead_added_tokens, lead_ends_sequence, letter_ids};
use anyhow::anyhow;
use serde_json::Value;

use super::pack::{combine_passes, ensemble_mode, option_probs, pack_from_probs_with};
use super::wire::{
    AnswerOut, ObjMap, QuestionKind, Request, Response, RunError, WireError, checked_options,
    question_value,
};

struct Prepared {
    row: Row,
    tokens: Vec<i32>,
    prompt: String,
    forms: Vec<Vec<i32>>,
    style: Style,
}

/// One question parsed and validated once at the request boundary.
struct CheckedQuestion<'a> {
    qid: &'a str,
    question: &'a super::wire::Question,
    kind: QuestionKind,
    options: &'a [(String, Value)],
}

/// `BONJEV_TOKEN_DUMP=1` prints one JSON line per scored pass: the top raw
/// vocabulary tokens and the per-label slot logits.
fn dump_enabled() -> bool {
    std::env::var("BONJEV_TOKEN_DUMP")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
        .unwrap_or(false)
}

fn dump_top(engine: &Engine, qid: &str, prep: &Prepared, logits: &[f32]) {
    const TOP: usize = 8;
    let mut best: Vec<(i32, f32)> = Vec::with_capacity(TOP);
    for (index, &value) in logits.iter().enumerate() {
        if best.len() < TOP {
            best.push((index as i32, value));
            best.sort_by(|a, b| b.1.total_cmp(&a.1));
        } else if value > best[TOP - 1].1 {
            best[TOP - 1] = (index as i32, value);
            best.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
    }
    let top: Vec<serde_json::Value> = best
        .into_iter()
        .map(|(id, logit)| {
            serde_json::json!({
                "id": id,
                "piece": engine.token_piece(id).unwrap_or_default(),
                "logit": logit,
            })
        })
        .collect();
    let slots: Vec<serde_json::Value> = prep
        .forms
        .iter()
        .enumerate()
        .map(|(index, ids)| {
            let id = ids.first().copied().unwrap_or(-1);
            let logit = usize::try_from(id)
                .ok()
                .and_then(|index| logits.get(index))
                .copied()
                .unwrap_or(f32::NAN);
            serde_json::json!({
                "label": crate::prompt::LETTERS
                    .chars()
                    .nth(index)
                    .map(String::from)
                    .unwrap_or_default(),
                "wire": prep.row.options.get(index).map(|(key, _)| key.clone()).unwrap_or_default(),
                "id": id,
                "piece": engine.token_piece(id).unwrap_or_default(),
                "logit": logit,
            })
        })
        .collect();
    eprintln!(
        "[bonjev] dump {}",
        serde_json::json!({
            "qid": qid,
            "style": prep.style.name,
            "top": top,
            "slots": slots,
        })
    );
}

pub fn run(engine: &mut Engine, model_name: &str, req: &Request) -> Result<Response, RunError> {
    if req.questions.is_empty() {
        return Err(WireError::new("questions", "questions is empty").into());
    }
    if crate::readout::contextual_enabled() {
        ensure_prior(engine)?;
    }
    let requested = req
        .prompt_style
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let explicit = match requested {
        Some(raw) => Some(
            styles_for(Some(raw)).map_err(|err| WireError::new("prompt_style", err.to_string()))?,
        ),
        None => None,
    };
    let env_style = std::env::var("BONJEV_STYLE")
        .ok()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty());
    let picture = take_picture(req)?;
    let state = Arc::new(req.state.clone());
    let output_tokens = u32::try_from(req.questions.len()).unwrap_or(u32::MAX);
    // Parse and validate every question once, at the request boundary.
    let mut checked = Vec::with_capacity(req.questions.len());
    for (qid, question) in req.questions.iter() {
        let (kind, options) = checked_options(qid, question)?;
        checked.push((qid.as_str(), question, kind, options));
    }
    let mut answers = ObjMap::new();
    let mut input_tokens = 0usize;
    let mut cache_hit_tokens = 0usize;
    for (qid, question, kind, options) in checked {
        // A request style wins; then `BONJEV_STYLE`; then the per-axis default
        // picked by the exam sweep (`auto_style`).
        let styles = match &explicit {
            Some(styles) => styles.clone(),
            None => match &env_style {
                Some(name) => styles_for(Some(name))
                    .map_err(|err| WireError::new("prompt_style", err.to_string()))?,
                None => vec![
                    auto_style(kind.axis())
                        .map_err(|err| WireError::new("prompt_style", err.to_string()))?,
                ],
            },
        };
        let (n_tokens, kept, answer) = score_styles(
            engine,
            &state,
            &CheckedQuestion {
                qid,
                question,
                kind,
                options: &options,
            },
            picture.as_ref(),
            &styles,
        )?;
        input_tokens += n_tokens;
        cache_hit_tokens += kept;
        answers.insert(qid.to_string(), answer);
    }
    Ok(Response {
        model: model_name.to_string(),
        answers,
        usage: super::wire::Usage {
            input_tokens,
            output_tokens,
            cache_hit_tokens,
            cache_miss_tokens: input_tokens.saturating_sub(cache_hit_tokens),
        },
    })
}

fn prepare_one(
    engine: &mut Engine,
    state: &Arc<crate::prompt::StateValue>,
    checked: &CheckedQuestion<'_>,
    picture: Option<&Rgb>,
    style: Style,
) -> Result<Prepared, RunError> {
    let row = Row {
        state: Arc::clone(state),
        question: question_value(checked.question),
        options: checked.options.to_vec(),
        axis: checked.kind.axis(),
    };
    let labels = labels_for(engine, row.options.len())?;
    let mut prompt = render_style(&row, &labels, style)?;
    if picture.is_some() {
        let marker = engine
            .media_marker()
            .ok_or_else(|| anyhow!("vision projector is not loaded"))?;
        prompt = with_media(&prompt, marker);
    }
    let Some(prefix) = prompt.strip_suffix(style.lead) else {
        return Err(anyhow!("prompt does not end with the answer lead").into());
    };
    let lead_ids = engine.tokenize(style.lead, false)?;
    let prefix_tokens = engine.tokenize(prefix, false)?;
    let decoded = engine.tokenize(&prompt, false)?;
    if !lead_added_tokens(&prefix_tokens, &decoded) {
        return Err(anyhow!("answer lead added no tokens").into());
    }
    if !lead_ends_sequence(&decoded, &lead_ids) {
        return Err(anyhow!("answer lead is not intact at the end of the prompt").into());
    }
    let tokens = if picture.is_some() {
        Vec::new()
    } else {
        decoded
    };
    let forms = letter_ids(engine, style.lead, &labels)?;
    Ok(Prepared {
        row,
        tokens,
        prompt,
        forms,
        style,
    })
}

fn score_styles(
    engine: &mut Engine,
    state: &Arc<crate::prompt::StateValue>,
    checked: &CheckedQuestion<'_>,
    picture: Option<&Rgb>,
    styles: &[Style],
) -> Result<(usize, usize, AnswerOut), RunError> {
    let mut preps = Vec::with_capacity(styles.len());
    for style in styles {
        preps.push(prepare_one(engine, state, checked, picture, *style)?);
    }
    if picture.is_some() {
        let mut passes = Vec::with_capacity(preps.len());
        let mut tokens = 0usize;
        let mut kept_sum = 0usize;
        let mut answer_first = None;
        for prep in &preps {
            let (n_tokens, kept, logits) = logits_of(engine, prep, picture)?;
            tokens += n_tokens;
            kept_sum += kept.min(n_tokens);
            passes.push(option_probs(&prep.forms, &logits, prep.row.axis));
            if answer_first.is_none() {
                answer_first = Some(crate::readout::top_is_label(&logits, &prep.forms));
            }
            if dump_enabled() {
                dump_top(engine, checked.qid, prep, &logits);
            }
        }
        let mut answer = pack_group(&preps[0].row, &passes)?;
        answer.set_answer_first(answer_first);
        return Ok((tokens, kept_sum, answer));
    }
    let tokens = preps.iter().map(|prep| prep.tokens.len()).sum();
    let scored = decode_questions(engine, &preps)?;
    let mut kept_sum = 0usize;
    let mut passes = Vec::with_capacity(preps.len());
    let mut answer_first = None;
    for (prep, (logits, kept, _)) in preps.iter().zip(&scored) {
        kept_sum += (*kept).min(prep.tokens.len());
        passes.push(option_probs(&prep.forms, logits, prep.row.axis));
        if answer_first.is_none() {
            answer_first = Some(crate::readout::top_is_label(logits, &prep.forms));
        }
        if dump_enabled() {
            dump_top(engine, checked.qid, prep, logits);
        }
    }
    let mut answer = pack_group(&preps[0].row, &passes)?;
    answer.set_answer_first(answer_first);
    Ok((tokens, kept_sum, answer))
}

fn ensure_prior(engine: &mut Engine) -> Result<(), RunError> {
    if crate::readout::prior().is_some() {
        return Ok(());
    }
    let mut tokens = engine.tokenize("", true)?;
    if tokens.is_empty() {
        tokens = engine.tokenize("N/A", true)?;
    }
    if tokens.is_empty() {
        return Err(anyhow!("context calibration: the null prompt tokenizes to nothing").into());
    }
    engine.decode(&tokens)?;
    let logits = engine.logits()?;
    crate::readout::set_prior(logits.to_vec());
    eprintln!(
        "[bonjev] context prior set ({} tokens, {} vocab)",
        tokens.len(),
        logits.len()
    );
    Ok(())
}

fn pack_group(row: &Row, passes: &[Vec<f32>]) -> Result<AnswerOut, RunError> {
    let (probs, choice) = combine_passes(passes, ensemble_mode())?;
    pack_from_probs_with(row, &probs, choice)
}

fn decode_questions(
    engine: &mut Engine,
    preps: &[Prepared],
) -> Result<Vec<(Vec<f32>, usize, usize)>, RunError> {
    let width = engine.n_seq.max(1);
    let mut out = Vec::with_capacity(preps.len());
    for chunk in preps.chunks(width) {
        let seqs: Vec<&[i32]> = chunk.iter().map(|prep| prep.tokens.as_slice()).collect();
        let rows = engine.decode_slices(&seqs)?;
        let keeps = engine.last_keep();
        for (index, row) in rows.into_iter().enumerate() {
            let (kept, total) = keeps
                .get(index)
                .copied()
                .unwrap_or((0, chunk[index].tokens.len()));
            out.push((row, kept, total));
        }
    }
    Ok(out)
}

fn logits_of(
    engine: &mut Engine,
    prep: &Prepared,
    picture: Option<&Rgb>,
) -> Result<(usize, usize, Vec<f32>), RunError> {
    if let Some(picture) = picture {
        let n = engine.decode_rgb(&prep.prompt, &picture.pixels, picture.width, picture.height)?;
        let kept = engine.last_keep().first().map(|pair| pair.0).unwrap_or(0);
        return Ok((n, kept, engine.logits()?.to_vec()));
    }
    engine.decode(&prep.tokens)?;
    let kept = engine.last_keep().first().map(|pair| pair.0).unwrap_or(0);
    Ok((prep.tokens.len(), kept, engine.logits()?.to_vec()))
}

fn take_picture(req: &Request) -> Result<Option<Rgb>, RunError> {
    if let Some(bytes) = &req.image_bytes {
        return Ok(Some(picture::decode_bytes(bytes).map_err(image_err)?));
    }
    if let Some(text) = req
        .image
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return Ok(Some(picture::decode_base64(text).map_err(image_err)?));
    }
    if let Some(text) = req.state_image() {
        return Ok(Some(picture::decode_base64(text).map_err(image_err)?));
    }
    Ok(None)
}

fn image_err(err: anyhow::Error) -> RunError {
    RunError::Wire(WireError::new("image", err.to_string()))
}
