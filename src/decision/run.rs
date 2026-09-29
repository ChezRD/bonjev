use crate::engine::Engine;
use crate::picture::{self, Rgb};
use crate::prompt::{Axis, Row, labels_for, letter_bank, prior_enabled, render, with_media};
use crate::readout::forms;
use anyhow::anyhow;

use super::pack::pack_answer;
use super::wire::{
    AnswerOut, ObjMap, Request, Response, RunError, WireError, check_questions, checked_options,
    question_text,
};

struct Prepared {
    qid: String,
    row: Row,
    tokens: Vec<i32>,
    prompt: String,
    forms: Vec<Vec<i32>>,
    prior: Option<Vec<f32>>,
}

pub fn run(engine: &mut Engine, model_name: &str, req: Request) -> Result<Response, RunError> {
    check_questions(&req)?;
    let calibrate = req.scores_are_calibrated && prior_enabled();
    let picture = take_picture(&req)?;
    let state = req.state;
    let output_tokens = req.questions.len() as u32;
    let mut answers = ObjMap::new();
    let mut input_tokens = 0usize;
    if picture.is_some() {
        for (qid, question) in req.questions {
            let (n_tokens, answer) =
                score_question(engine, &qid, &state, question, picture.as_ref(), calibrate)?;
            input_tokens += n_tokens;
            answers.insert(qid, answer);
        }
    } else {
        let mut preps = Vec::new();
        for (qid, question) in req.questions {
            preps.push(prepare_one(engine, qid, state.clone(), question, None)?);
        }
        for prep in &preps {
            input_tokens += prep.tokens.len();
        }
        let logit_rows = decode_questions(engine, &preps)?;
        for prep in &mut preps {
            attach_choice_prior(engine, prep, calibrate)?;
        }
        for (prep, logits) in preps.into_iter().zip(logit_rows) {
            let answer = pack_answer(&prep.row, &prep.forms, &logits, prep.prior.as_deref())?;
            answers.insert(prep.qid, answer);
        }
    }
    Ok(Response {
        model: model_name.to_string(),
        answers,
        usage: super::wire::Usage {
            input_tokens,
            output_tokens,
        },
    })
}

fn prepare(engine: &mut Engine, req: Request) -> Result<Prepared, RunError> {
    if req.questions.len() != 1 {
        return Err(anyhow!("batch decode expects one question per request").into());
    }
    check_questions(&req)?;
    let calibrate = req.scores_are_calibrated && prior_enabled();
    let picture = take_picture(&req)?;
    let state = req.state;
    let (qid, question) = req
        .questions
        .into_iter()
        .next()
        .expect("BUG: length check guarantees one question");
    let mut prep = prepare_one(engine, qid, state, question, picture.as_ref())?;
    attach_choice_prior(engine, &mut prep, calibrate)?;
    Ok(prep)
}

fn prepare_one(
    engine: &mut Engine,
    qid: String,
    state: crate::prompt::StateValue,
    question: super::wire::Question,
    picture: Option<&Rgb>,
) -> Result<Prepared, RunError> {
    let (kind, options) = checked_options(&qid, &question)?;
    let row = Row {
        state,
        question: question_text(&question),
        options,
        axis: kind.axis(),
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
    Ok(Prepared {
        qid,
        row,
        tokens,
        prompt,
        forms,
        prior: None,
    })
}

fn attach_choice_prior(
    engine: &Engine,
    prep: &mut Prepared,
    calibrate: bool,
) -> Result<(), RunError> {
    if !(calibrate && prep.row.axis == Axis::Choice) {
        return Ok(());
    }
    prep.prior = engine
        .prior_logits()
        .map(|logits| crate::readout::slot_logits(logits, &prep.forms));
    Ok(())
}

fn score_question(
    engine: &mut Engine,
    qid: &str,
    state: &crate::prompt::StateValue,
    question: super::wire::Question,
    picture: Option<&Rgb>,
    calibrate: bool,
) -> Result<(usize, AnswerOut), RunError> {
    let mut prep = prepare_one(engine, qid.to_string(), state.clone(), question, picture)?;
    attach_choice_prior(engine, &mut prep, calibrate)?;
    let (n_tokens, logits) = logits_of(engine, &prep, picture)?;
    let answer = pack_answer(&prep.row, &prep.forms, &logits, prep.prior.as_deref())?;
    Ok((n_tokens, answer))
}

fn decode_questions(engine: &mut Engine, preps: &[Prepared]) -> Result<Vec<Vec<f32>>, RunError> {
    let width = engine.n_seq.max(1);
    let mut rows = Vec::with_capacity(preps.len());
    for chunk in preps.chunks(width) {
        let seqs: Vec<&[i32]> = chunk.iter().map(|prep| prep.tokens.as_slice()).collect();
        rows.extend(engine.decode_slices(&seqs)?);
    }
    Ok(rows)
}

fn logits_of(
    engine: &mut Engine,
    prep: &Prepared,
    picture: Option<&Rgb>,
) -> Result<(usize, Vec<f32>), RunError> {
    if let Some(picture) = picture {
        let n = engine.decode_rgb(&prep.prompt, &picture.pixels, picture.width, picture.height)?;
        return Ok((n, engine.logits()?.to_vec()));
    }
    engine.decode(&prep.tokens)?;
    Ok((prep.tokens.len(), engine.logits()?.to_vec()))
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

pub fn load_choice_prior(engine: &mut Engine, seq_tokens: usize) -> Result<(), RunError> {
    if !prior_enabled() {
        return Ok(());
    }
    let n = letter_bank().len();
    let labels = labels_for(n)?;
    let prompt = equivalent_prompt(&labels);
    let tokens = engine.tokenize(&prompt, false)?;
    if tokens.len() > seq_tokens {
        return Err(anyhow!(
            "choice prior prompt is {} tokens and one sequence holds {seq_tokens}",
            tokens.len()
        )
        .into());
    }
    engine.decode(&tokens)?;
    let logits = engine.logits()?.to_vec();
    engine.set_prior_logits(logits);
    eprintln!("[bonjev] choice prior: one pass, {n} equal options");
    Ok(())
}

fn equivalent_prompt(labels: &[String]) -> String {
    let options = labels
        .iter()
        .map(|label| (label.clone(), format!("{label}: Option {label}.")))
        .collect();
    let row = Row {
        state: "N/A".into(),
        question: "Which option follows?".to_string(),
        options,
        axis: Axis::Choice,
    };
    render(&row, labels)
}

/// Score several requests. One request takes the single-sequence path.
/// A wider batch decodes independent prompts together.
pub fn run_many(
    engine: &mut Engine,
    model_name: &str,
    reqs: Vec<Request>,
) -> Vec<Result<Response, RunError>> {
    if reqs.len() == 1
        || reqs
            .iter()
            .any(|req| req.has_image() || req.questions.len() != 1)
    {
        return reqs
            .into_iter()
            .map(|req| run(engine, model_name, req))
            .collect();
    }
    let n = reqs.len();
    let mut slots: Vec<Option<Prepared>> = Vec::with_capacity(n);
    let mut results: Vec<Option<Result<Response, RunError>>> = (0..n).map(|_| None).collect();
    for req in reqs {
        match prepare(engine, req) {
            Ok(prep) => slots.push(Some(prep)),
            Err(err) => {
                results[slots.len()] = Some(Err(err));
                slots.push(None);
            }
        }
    }
    let ready_idx: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, prep)| prep.as_ref().map(|_| index))
        .collect();
    let seqs: Vec<Vec<i32>> = ready_idx
        .iter()
        .map(|&index| {
            slots[index]
                .as_ref()
                .expect("BUG: ready index points at a prepared slot")
                .tokens
                .clone()
        })
        .collect();
    let decoded = if seqs.is_empty() {
        Ok(Vec::new())
    } else {
        engine.decode_many(&seqs)
    };
    match decoded {
        Err(err) => {
            let message = err.to_string();
            for index in ready_idx {
                results[index] = Some(Err(anyhow!(message.clone()).into()));
            }
        }
        Ok(rows) => {
            for (row, index) in rows.into_iter().zip(ready_idx) {
                let prep = slots[index]
                    .take()
                    .expect("BUG: ready index points at a prepared slot");
                let answer = pack_answer(&prep.row, &prep.forms, &row, prep.prior.as_deref());
                results[index] = Some(answer.map(|body| {
                    let mut answers = ObjMap::new();
                    answers.insert(prep.qid, body);
                    Response {
                        model: model_name.to_string(),
                        answers,
                        usage: super::wire::Usage {
                            input_tokens: prep.tokens.len(),
                            output_tokens: 1,
                        },
                    }
                }));
            }
        }
    }
    results
        .into_iter()
        .map(|item| item.expect("BUG: every batch slot is filled before return"))
        .collect()
}
