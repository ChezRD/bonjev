use serde_json::{Value, json};

use super::pack::{
    EnsembleMode, average_passes, combine_passes, noul_true_probability, option_probs, pack_answer,
    positive_argmax,
};
use super::wire::{AnswerOut, QuestionKind, Request, check_questions, checked_options};
use crate::prompt::{Axis, Row};

fn keys_of(answer: &impl serde::Serialize) -> Vec<String> {
    let value = serde_json::to_value(answer).unwrap();
    let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    keys
}

fn request(raw: &str) -> Request {
    serde_json::from_str(raw).unwrap()
}

#[test]
fn official_examples_deserialize() {
    let noul = request(
        r#"{"state":"Help! My payouts have been failing for 3 days.","model":"jev-latest","questions":{"is_urgent":{"type":"noul","instructions":"Does this convey urgency?"}}}"#,
    );
    check_questions(&noul).unwrap();
    let (_, options) =
        checked_options("is_urgent", noul.questions.get("is_urgent").unwrap()).unwrap();
    assert_eq!(options.len(), 2);

    let choice = request(
        r#"{"state":"Help! My payouts have been failing for 3 days.","model":"jev-latest","questions":{"department":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments, invoicing, refunds","technical":"Bugs, outages, integrations","sales":"Pricing, upgrades, new accounts"}}}}"#,
    );
    check_questions(&choice).unwrap();

    let score = request(
        r#"{"state":"Help! My payouts have been failing for 3 days.","model":"jev-latest","questions":{"frustration":{"type":"score","instructions":"How frustrated is the customer?","criteria":["Calm","Frustrated","Very angry"]}}}"#,
    );
    check_questions(&score).unwrap();
    let (_, levels) =
        checked_options("frustration", score.questions.get("frustration").unwrap()).unwrap();
    assert_eq!(levels[0].0, "0");
    assert_eq!(levels[2].1, json!("Very angry"));

    // Structured levels and object instructions are TypeOne entries, not 422.
    let structured = request(
        r#"{"state":{"ticket":"x"},"questions":{"heat":{"type":"score","instructions":{"question":"Rate it","note":"facts only"},"criteria":[{"summary":"low","signals":["ok"]},{"summary":"high","signals":["bad"]}]}}}"#,
    );
    check_questions(&structured).unwrap();
    let (_, structured_levels) =
        checked_options("heat", structured.questions.get("heat").unwrap()).unwrap();
    assert_eq!(structured_levels.len(), 2);
}

#[test]
fn answer_shapes_match_the_wire() {
    let noul = Row {
        state: std::sync::Arc::new("x".into()),
        question: "q".into(),
        options: vec![("true".into(), "yes".into()), ("false".into(), "no".into())],
        axis: Axis::Noul,
    };
    let answer = pack_answer(&noul, &[vec![0], vec![1]], &[2.0, 0.0]).unwrap();
    assert_eq!(keys_of(&answer), ["entropy", "margin", "noul", "type"]);
    match answer {
        AnswerOut::Noul { noul, .. } => assert!(noul > 0.5),
        other => panic!("expected noul, got {other:?}"),
    }

    let choice = Row {
        state: std::sync::Arc::new("x".into()),
        question: "q".into(),
        options: vec![("a".into(), "A".into()), ("b".into(), "B".into())],
        axis: Axis::Choice,
    };
    let answer = pack_answer(&choice, &[vec![0], vec![1]], &[2.0, 0.0]).unwrap();
    assert_eq!(
        keys_of(&answer),
        ["choice", "confidence", "entropy", "margin", "probabilities", "type"]
    );
    match answer {
        AnswerOut::Choice { choice, .. } => assert_eq!(choice, "a"),
        other => panic!("expected choice, got {other:?}"),
    }

    let score = Row {
        state: std::sync::Arc::new("x".into()),
        question: "q".into(),
        options: vec![("0".into(), "low".into()), ("1".into(), "high".into())],
        axis: Axis::Score,
    };
    let answer = pack_answer(&score, &[vec![0], vec![1]], &[0.0, 0.0]).unwrap();
    assert_eq!(
        keys_of(&answer),
        ["confidence", "entropy", "legend", "margin", "probabilities", "score", "type"]
    );
    match answer {
        AnswerOut::Score {
            score,
            legend,
            confidence,
            ..
        } => {
            assert!((score - 0.5).abs() < 1e-5);
            assert_eq!(legend.get("1"), Some(&json!("high")));
            assert_eq!(confidence, 0.0);
        }
        other => panic!("expected score, got {other:?}"),
    }
}

#[test]
fn wire_errors_name_the_field() {
    let unknown = request(r#"{"questions":{"route":{"type":"essay","instructions":"q"}}}"#);
    let err = check_questions(&unknown).unwrap_err();
    assert_eq!(err.field(), "questions.route.type");

    let one_choice = request(
        r#"{"questions":{"decision":{"type":"choice","instructions":"q","criteria":{"a":"A"}}}}"#,
    );
    let err = check_questions(&one_choice).unwrap_err();
    assert_eq!(err.field(), "questions.decision.criteria");
    assert!(err.to_string().contains("at least two options"));

    let one_level = request(
        r#"{"questions":{"heat":{"type":"score","instructions":"q","criteria":["only"]}}}"#,
    );
    let err = check_questions(&one_level).unwrap_err();
    assert_eq!(err.field(), "questions.heat.criteria");
    assert!(err.to_string().contains("2 to 10"));

    let empty = request(r#"{"questions":{}}"#);
    assert_eq!(check_questions(&empty).unwrap_err().field(), "questions");
}

#[test]
fn boolean_alias_answers_as_noul() {
    let req = request(r#"{"questions":{"flag":{"type":"boolean","instructions":"q"}}}"#);
    let (kind, options) = checked_options("flag", req.questions.get("flag").unwrap()).unwrap();
    assert_eq!(kind, QuestionKind::Noul);
    assert_eq!(
        options,
        vec![
            ("true".into(), "true".into()),
            ("false".into(), "false".into())
        ]
    );
}

#[test]
fn criteria_and_state_keep_json_key_order() {
    let req = request(
        r#"{"state":{"z":"2","a":"1"},"questions":{"decision":{"type":"choice","instructions":"q","criteria":{"z":"last","a":"first"}}}}"#,
    );
    let state = crate::yaml_emit::state_node_public(&req.state);
    let back: Value =
        yaml_serde::from_str(&yaml_serde::to_string(&state).unwrap()).expect("state round-trips");
    let keys: Vec<_> = back
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["z", "a"]);
    assert_eq!(back["z"], json!("2"));
    assert_eq!(back["a"], json!("1"));
    let (_, options) = checked_options("decision", req.questions.get("decision").unwrap()).unwrap();
    assert_eq!(options[0].0, "z");
    assert_eq!(options[1].0, "a");
    assert_eq!(options[0].1, json!("last"));
}

#[test]
fn noul_without_true_key_is_an_error() {
    let req = request(
        r#"{"questions":{"flag":{"type":"noul","instructions":"q","criteria":{"a":"x","b":"y"}}}}"#,
    );
    let err = check_questions(&req).unwrap_err();
    assert_eq!(err.field(), "questions.flag.criteria");
    let options = vec![("a".into(), "x".into()), ("b".into(), "y".into())];
    let err = noul_true_probability(&options, &[0.2, 0.8]).unwrap_err();
    assert!(err.to_string().contains("true or yes"));
}

#[test]
fn zero_probabilities_are_not_a_choice() {
    let err = positive_argmax(&[0.0, 0.0]).unwrap_err();
    assert!(err.to_string().contains("finite logit"));
    let options = vec![("true".into(), "yes".into())];
    let err = noul_true_probability(&options, &[0.0]).unwrap_err();
    assert!(err.to_string().contains("finite logit"));
}

#[test]
fn several_questions_keep_their_ids() {
    let req = request(
        r#"{"state":{"text":"x"},"questions":{"route":{"type":"choice","instructions":"q","criteria":{"a":"A","b":"B"}},"flag":{"type":"noul","instructions":"q"},"heat":{"type":"score","instructions":"q","criteria":["low","high"]}}}"#,
    );
    assert_eq!(req.questions.len(), 3);
    let ids: Vec<_> = req.questions.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(ids, ["route", "flag", "heat"]);
    check_questions(&req).unwrap();
    let state = crate::yaml_emit::state_node_public(&req.state);
    let back: Value =
        yaml_serde::from_str(&yaml_serde::to_string(&state).unwrap()).expect("state round-trips");
    assert_eq!(back, json!({"text": "x"}));
}

#[test]
fn ignored_request_model_does_not_fail_parse() {
    let with_model = request(
        r#"{"model":"jev-latest","state":"ticket","questions":{"decision":{"type":"choice","instructions":"q","criteria":{"a":"A","b":"B"}}}}"#,
    );
    let without = request(
        r#"{"state":"ticket","questions":{"decision":{"type":"choice","instructions":"q","criteria":{"a":"A","b":"B"}}}}"#,
    );
    assert_eq!(with_model.state, without.state);
    let flagged =
        checked_options("decision", with_model.questions.get("decision").unwrap()).unwrap();
    let plain = checked_options("decision", without.questions.get("decision").unwrap()).unwrap();
    assert_eq!(flagged, plain);
}

#[test]
fn prompt_style_is_optional() {
    let named = request(
        r#"{"state":"ticket","prompt_style":"native_user","questions":{"decision":{"type":"choice","instructions":"q","criteria":{"a":"A","b":"B"}}}}"#,
    );
    assert_eq!(named.prompt_style.as_deref(), Some("native_user"));
    let plain = request(
        r#"{"state":"ticket","questions":{"decision":{"type":"choice","instructions":"q","criteria":{"a":"A","b":"B"}}}}"#,
    );
    assert_eq!(plain.prompt_style, None);
}

#[test]
fn vote_picks_the_majority_slot() {
    let (probs, choice) = combine_passes(
        &[vec![0.9, 0.1], vec![0.4, 0.6], vec![0.8, 0.2]],
        EnsembleMode::Vote,
    )
    .unwrap();
    assert_eq!(choice, Some(0));
    assert!((probs[0] - 0.7).abs() < 1e-6);
}

#[test]
fn vote_tie_falls_back_to_the_mean() {
    let (_probs, choice) =
        combine_passes(&[vec![0.9, 0.1], vec![0.1, 0.9]], EnsembleMode::Vote).unwrap();
    assert_eq!(choice, None);
}

#[test]
fn weighted_gives_the_confident_pass_more_weight() {
    let (probs, choice) =
        combine_passes(&[vec![0.95, 0.05], vec![0.6, 0.4]], EnsembleMode::Weighted).unwrap();
    assert_eq!(choice, None);
    assert!(probs[0] > 0.85, "{probs:?}");
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn ensemble_average_keeps_the_same_slots() {
    let avg = average_passes(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
    assert!((avg[0] - 2.0 / 3.0).abs() < 1e-6);
    assert!((avg[1] - 1.0 / 3.0).abs() < 1e-6);
    let slots = option_probs(&[vec![1], vec![2]], &[0.0, 1.0, 1.0], Axis::Choice);
    assert!((slots[0] - 0.5).abs() < 1e-5);
    assert!((slots[1] - 0.5).abs() < 1e-5);
}
