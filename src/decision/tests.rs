use serde_json::{Value, json};

use super::pack::{noul_true_probability, pack_answer, positive_argmax};
use super::wire::{QuestionKind, Request, check_questions, checked_options};
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
    assert_eq!(levels[2].1, "Very angry");
}

#[test]
fn answer_shapes_match_the_wire() {
    let noul = Row {
        state: "x".into(),
        question: "q".into(),
        options: vec![("true".into(), "yes".into()), ("false".into(), "no".into())],
        axis: Axis::Noul,
    };
    let answer = pack_answer(&noul, &[vec![0], vec![1]], &[2.0, 0.0], None).unwrap();
    assert_eq!(keys_of(&answer), ["noul", "type"]);
    assert!(answer.noul.unwrap() > 0.5);

    let choice = Row {
        state: "x".into(),
        question: "q".into(),
        options: vec![("a".into(), "A".into()), ("b".into(), "B".into())],
        axis: Axis::Choice,
    };
    let answer = pack_answer(&choice, &[vec![0], vec![1]], &[2.0, 0.0], None).unwrap();
    assert_eq!(
        keys_of(&answer),
        ["choice", "confidence", "probabilities", "type"]
    );
    assert_eq!(answer.choice.as_deref(), Some("a"));

    let score = Row {
        state: "x".into(),
        question: "q".into(),
        options: vec![("0".into(), "low".into()), ("1".into(), "high".into())],
        axis: Axis::Score,
    };
    let answer = pack_answer(&score, &[vec![0], vec![1]], &[0.0, 0.0], None).unwrap();
    assert_eq!(
        keys_of(&answer),
        ["confidence", "legend", "probabilities", "score", "type"]
    );
    assert!((answer.score.unwrap() - 0.5).abs() < 1e-5);
    assert_eq!(
        answer.legend.unwrap().get("1").map(String::as_str),
        Some("high")
    );
    assert_eq!(answer.confidence.unwrap(), 0.0);
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
    assert!(err.to_string().contains("2 to 255"));

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
    let scalars = crate::yaml_emit::parse_scalars(&req.state.text()).unwrap();
    let keys: Vec<_> = scalars
        .iter()
        .step_by(2)
        .map(|scalar| scalar.text.as_str())
        .collect();
    assert_eq!(keys, ["z", "a"]);
    let values: Vec<_> = scalars
        .iter()
        .skip(1)
        .step_by(2)
        .map(|scalar| scalar.text.trim_end())
        .collect();
    assert_eq!(values, ["2", "1"]);
    let (_, options) = checked_options("decision", req.questions.get("decision").unwrap()).unwrap();
    assert_eq!(options[0].0, "z");
    assert_eq!(options[1].0, "a");
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
    assert!(req.state.text().contains('x'));
}

#[test]
fn calibration_flag_defaults_on() {
    let on =
        request(r#"{"questions":{"decision":{"instructions":"q","criteria":{"a":"A","b":"B"}}}}"#);
    assert!(on.scores_are_calibrated);
    assert_eq!(
        checked_options("decision", on.questions.get("decision").unwrap())
            .unwrap()
            .0,
        QuestionKind::Choice
    );
    let off = request(
        r#"{"scores_are_calibrated":false,"questions":{"decision":{"instructions":"q","criteria":{"a":"A","b":"B"}}}}"#,
    );
    assert!(!off.scores_are_calibrated);
}

#[test]
fn ignored_request_model_does_not_fail_parse() {
    let value = json!({"model": "jev-latest", "questions": {}});
    let raw = serde_json::to_string(&value).unwrap();
    let req = request(&raw);
    assert!(req.questions.is_empty());
    let _: Value = serde_json::from_str(&raw).unwrap();
}
