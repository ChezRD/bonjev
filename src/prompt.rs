#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Choice,
    Noul,
    Score,
}

pub struct Row {
    pub state: String,
    pub question: String,
    pub options: Vec<(String, String)>,
    pub axis: Axis,
}

pub const DEFAULT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. \
Respond with only the listed answer token, with no explanation or reasoning.";

pub fn system_prompt() -> String {
    std::env::var("BONJEV_SYSTEM").unwrap_or_else(|_| DEFAULT_SYSTEM.to_string())
}

const THINK_PREFILL_CAREFUL: &str =
    "Carefully verifying the evidence against the criteria to determine the exact outcome.";
const THINK_PREFILL_REVIEW: &str =
    "Reviewing the facts, checking constraints, and selecting the strictly supported option.";

fn think_prefill_for_state(state: &str) -> Option<String> {
    let raw = std::env::var("BONJEV_THINK_PREFILL").unwrap_or_default();
    match raw.trim() {
        "" => None,
        "auto" => {
            if crate::features::use_think_prefill(state) {
                Some(THINK_PREFILL_CAREFUL.to_string())
            } else {
                None
            }
        }
        "1" | "careful" => Some(THINK_PREFILL_CAREFUL.to_string()),
        "2" | "review" => Some(THINK_PREFILL_REVIEW.to_string()),
        other if !other.is_empty() => Some(other.to_string()),
        _ => None,
    }
}

pub fn assistant_prefix_for_state(state: &str) -> String {
    match std::env::var("BONJEV_THINK_TAGS").as_deref() {
        Ok("0") => String::new(),
        _ => {
            let inner = think_prefill_for_state(state)
                .map(|t| format!("{}\n", t))
                .unwrap_or_else(|| "\n".to_string());
            format!("<think>\n{inner}</think>\n\n")
        }
    }
}

pub fn prior_enabled() -> bool {
    std::env::var("BONJEV_PRIOR").as_deref() != Ok("0")
}

pub fn prior_alpha() -> f32 {
    std::env::var("BONJEV_PRIOR_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.10)
}

pub fn letter_bank() -> &'static [u8] {
    match std::env::var("BONJEV_LETTERS")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "wxy" => b"WXYZABCDEFGHIJKLMNOPQRSTUV",
        "ijk" => b"IJKLMNOPQRSTUVWXYZABCDEFGH",
        "fgh" => b"FGHIJKLMNOPQRSTUVWXYZABCDE",
        _ => b"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    }
}

/// Two-letter labels after A–Z. Uppercase pairs like `AA` are not here:
/// writing them letter by letter emits token `A`, which is already slot A.
/// These are lowercase pairs (`aa`, `ab`, …). Spelling `aa` emits `a`, and
/// that id is not a letter slot. Each pair is one Bonsai token both bare and
/// with a leading space, and the piece is exactly the pair. Same list on q1 and q2.
/// Slots 0..26 stay the letter bank, so a question with at most 26 options
/// does not use this table.
const EXTRA_LABELS: &[&str] = &[
    "aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj",
    "ak", "al", "am", "an", "ao", "ap", "ar", "as", "at", "au",
    "av", "aw", "ax", "ay", "az", "ba", "bb", "bc", "bd", "be",
    "bf", "bg", "bh", "bi", "bj", "bk", "bl", "bm", "bn", "bo",
    "bp", "br", "bs", "bt", "bu", "bv", "bw", "bx", "by", "bz",
    "ca", "cb", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj",
    "ck", "cl", "cm", "cn", "co", "cp", "cq", "cr", "cs", "ct",
    "cu", "cv", "cw", "cx", "cy", "cz", "da", "db", "dc", "dd",
    "de", "df", "dg", "dh", "di", "dj", "dk", "dl", "dm", "dn",
    "do", "dp", "dq", "dr", "ds", "dt", "du", "dv", "dw", "dx",
    "dy", "dz", "ea", "eb", "ec", "ed", "ee", "ef", "eg", "eh",
    "ei", "ej", "ek", "el", "em", "en", "eo", "ep", "eq", "er",
    "es", "et", "eu", "ev", "ew", "ex", "ey", "ez", "fa", "fb",
    "fc", "fd", "fe", "ff", "fg", "fh", "fi", "fk", "fl", "fm",
    "fn", "fo", "fp", "fq", "fr", "fs", "ft", "fu", "fv", "fw",
    "fx", "fy", "ga", "gb", "gc", "gd", "ge", "gf", "gg", "gh",
    "gi", "gj", "gl", "gm", "gn", "go", "gp", "gr", "gs", "gt",
    "gu", "gv", "gw", "gx", "gy", "gz", "ha", "hb", "hc", "hd",
    "he", "hf", "hg", "hh", "hi", "hj", "hk", "hl", "hm", "hn",
    "ho", "hp", "hr", "hs", "ht", "hu", "hv", "hw", "hx", "hy",
    "ia", "ib", "ic", "id", "ie", "if", "ig", "ih", "ii", "ij",
    "ik", "il", "im", "in", "io", "ip", "iq", "ir", "is", "it",
    "iv", "iw", "ix", "iy", "iz", "ja", "jb", "jc", "jd",
];

pub const MAX_LABELS: usize = 255;

const _: () = assert!(26 + EXTRA_LABELS.len() == MAX_LABELS);

pub fn labels_for(n: usize) -> anyhow::Result<Vec<String>> {
    let bank = letter_bank();
    let cap = bank.len() + EXTRA_LABELS.len();
    if n > cap || n > MAX_LABELS {
        anyhow::bail!("need {n} answer labels but only {MAX_LABELS} are supported");
    }
    let mut out = Vec::with_capacity(n);
    for &byte in bank.iter().take(n) {
        out.push((byte as char).to_string());
    }
    if n > bank.len() {
        out.extend(EXTRA_LABELS[..n - bank.len()].iter().map(|s| (*s).to_string()));
    }
    Ok(out)
}

fn yaml_block(text: &str, indent: usize) -> String {
    let pad = " ".repeat(indent);
    text.lines()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn push_task_block(out: &mut String, question: &str) {
    out.push_str("task: |\n");
    out.push_str(&yaml_block(question.trim(), 2));
    out.push('\n');
}

fn push_data_block(out: &mut String, state: &str) {
    out.push_str("data: |\n");
    out.push_str(&yaml_block(state.trim(), 2));
    out.push('\n');
}

pub fn yaml_user(row: &Row, labels: &[String]) -> String {
    let mut out = String::new();
    let task_first = std::env::var("BONJEV_TASK_FIRST").as_deref() == Ok("1");
    let sandwich = crate::features::use_sandwich_layout(&row.state);
    let q = row.question.trim();

    if sandwich {
        push_task_block(&mut out, q);
        push_data_block(&mut out, row.state.trim());
        push_task_block(&mut out, q);
    } else if task_first {
        push_task_block(&mut out, q);
        push_data_block(&mut out, row.state.trim());
    } else {
        push_data_block(&mut out, row.state.trim());
        push_task_block(&mut out, q);
    }

    out.push_str("options:\n");
    for (i, (_key, text)) in row.options.iter().enumerate() {
        out.push_str(&format!("  {}: |\n", labels[i]));
        out.push_str(&yaml_block(text.trim(), 4));
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// Put the vision marker once, at the start of the user turn.
pub fn with_media(prompt: &str, marker: &str) -> String {
    if marker.is_empty() || prompt.contains(marker) {
        return prompt.to_string();
    }
    const USER: &str = "<|im_start|>user\n";
    if let Some(i) = prompt.find(USER) {
        let at = i + USER.len();
        let mut out = String::with_capacity(prompt.len() + marker.len() + 1);
        out.push_str(&prompt[..at]);
        out.push_str(marker);
        out.push('\n');
        out.push_str(&prompt[at..]);
        return out;
    }
    format!("{marker}\n{prompt}")
}

pub fn answer_suffix() -> String {
    std::env::var("BONJEV_SUFFIX").unwrap_or_else(|_| "Answer:".to_string())
}

/// Render model prompt: system -> user (yaml) -> assistant prefix -> answer slot.
pub fn render(row: &Row, labels: &[String]) -> String {
    let user = yaml_user(row, labels);
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(&system_prompt());
    prompt.push_str("<|im_end|>\n<|im_start|>user\n");
    prompt.push_str(&user);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt.push_str(&assistant_prefix_for_state(&row.state));
    prompt.push_str(&answer_suffix());
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_structure_matches_specification() {
        let row = Row {
            state: "Ticket".into(),
            question: "Which intent?".into(),
            options: vec![("a".into(), "refund".into())],
            axis: Axis::Choice,
        };
        let prompt = render(&row, &labels_for(1).unwrap());
        assert!(prompt.contains(DEFAULT_SYSTEM));
        assert!(prompt.contains("<|im_start|>assistant\n<think>\n"));
        assert!(prompt.contains("</think>\n\nAnswer:"));
        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("data: |\n  Ticket"));
    }

    #[test]
    fn media_marker_lands_once_in_the_user_turn() {
        let row = Row {
            state: "Ticket".into(),
            question: "Which intent?".into(),
            options: vec![("a".into(), "refund".into())],
            axis: Axis::Choice,
        };
        let prompt = with_media(&render(&row, &labels_for(1).unwrap()), "<__media__>");
        assert_eq!(prompt.matches("<__media__>").count(), 1);
        let user = prompt.find("<|im_start|>user\n<__media__>\n").expect("user turn");
        assert!(prompt.find("<|im_start|>system\n").unwrap() < user);
    }

    #[test]
    fn labels_use_letters_then_vocab_pairs() {
        let bank = letter_bank();
        let letters = labels_for(bank.len()).unwrap();
        assert_eq!(letters.len(), bank.len());
        assert_eq!(letters[0], (bank[0] as char).to_string());
        let extended = labels_for(bank.len() + 1).unwrap();
        assert_eq!(&extended[..bank.len()], &letters[..]);
        assert_eq!(extended[bank.len()], "aa");
        assert!(EXTRA_LABELS.iter().all(|lab| {
            let bytes = lab.as_bytes();
            bytes.len() == 2 && bytes[0].is_ascii_lowercase() && bytes[1].is_ascii_lowercase()
        }));
        let all = labels_for(MAX_LABELS).unwrap();
        assert_eq!(all.len(), MAX_LABELS);
        assert_eq!(all.last().map(String::as_str), Some("jd"));
        assert_eq!(all.iter().collect::<std::collections::HashSet<_>>().len(), MAX_LABELS);
        let err = labels_for(MAX_LABELS + 1).unwrap_err().to_string();
        assert!(err.contains("255"));
    }
}
