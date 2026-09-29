//! Observable task features for prompt routing (no task-id hardcoding).
//! Thresholds tuned on frozen runs lev-baseline vs lev-think-careful / lev-sandwich (231 public tasks).

/// Prose documents at or above this length benefit from task→data→task sandwich (net +1 vs baseline at 8000).
pub fn sandwich_chars_threshold() -> usize {
    std::env::var("BONJEV_SANDWICH_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000)
}

pub fn sandwich_disabled() -> bool {
    std::env::var("BONJEV_SANDWICH").as_deref() == Ok("0")
        || sandwich_chars_threshold() == 0
}

/// Flattened judge/request-evidence pairs (request:/response: blocks), not long prose policies.
pub fn state_is_structured_evidence(state: &str) -> bool {
    let s = state.trim();
    s.starts_with("request:")
        || (s.contains("request:") && s.contains("response:"))
}

pub fn state_char_len(state: &str) -> usize {
    state.trim().len()
}

pub fn use_sandwich_layout(state: &str) -> bool {
    !sandwich_disabled() && state_char_len(state) >= sandwich_chars_threshold()
}

/// Static think prefill on long prose only (net +1 vs always-on think on this bench).
pub fn think_chars_threshold() -> usize {
    std::env::var("BONJEV_THINK_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000)
}

pub fn use_think_prefill(state: &str) -> bool {
    if state_is_structured_evidence(state) {
        return false;
    }
    state_char_len(state) >= think_chars_threshold()
}
