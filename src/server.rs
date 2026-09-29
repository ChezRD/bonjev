use crate::decision::{self, Request, Response, RunError};
use crate::engine::Engine;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde::Serialize;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub struct AppState {
    pub model_name: String,
    slot: Slot,
}

enum Slot {
    Direct(Mutex<Engine>),
    Batch(Mutex<Sender<Job>>),
}

struct Job {
    req: Request,
    reply: Sender<Result<Response, RunError>>,
}

pub fn direct(engine: Engine, model_name: String) -> AppState {
    AppState {
        model_name,
        slot: Slot::Direct(Mutex::new(engine)),
    }
}

pub fn batch(engine: Engine, model_name: String, parallel: usize) -> AppState {
    let (tx, rx) = std::sync::mpsc::channel();
    let name = model_name.clone();
    std::thread::Builder::new()
        .name("bonjev-batch".to_string())
        .spawn(move || batch_loop(engine, name, rx, parallel))
        .expect("batch thread");
    AppState {
        model_name,
        slot: Slot::Batch(Mutex::new(tx)),
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .with_state(state)
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
}

async fn models(State(st): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: st.model_name.clone(),
            object: "model",
        }],
    })
}

async fn systemone(
    State(st): State<Arc<AppState>>,
    payload: Result<Json<Request>, JsonRejection>,
) -> HttpResponse {
    let Json(req) = match payload {
        Ok(json) => json,
        Err(err) => {
            return error_json(StatusCode::UNPROCESSABLE_ENTITY, err.to_string(), None);
        }
    };
    let result = if matches!(st.slot, Slot::Batch(_)) {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        {
            let Slot::Batch(tx) = &st.slot else {
                unreachable!("matched batch");
            };
            if tx
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .send(Job {
                    req,
                    reply: reply_tx,
                })
                .is_err()
            {
                return error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "batch worker stopped".to_string(),
                    None,
                );
            }
        }
        match tokio::task::spawn_blocking(move || reply_rx.recv()).await {
            Ok(Ok(body)) => body,
            Ok(Err(_)) => {
                return error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "batch worker dropped the reply".to_string(),
                    None,
                );
            }
            Err(err) => {
                return error_json(StatusCode::INTERNAL_SERVER_ERROR, err.to_string(), None);
            }
        }
    } else {
        match tokio::task::spawn_blocking(move || decide(&st, req)).await {
            Ok(body) => body,
            Err(err) => {
                return error_json(StatusCode::INTERNAL_SERVER_ERROR, err.to_string(), None);
            }
        }
    };
    match result {
        Ok(body) => Json(body).into_response(),
        Err(err) => decide_error(err),
    }
}

fn decide_error(err: RunError) -> HttpResponse {
    match err {
        RunError::Wire(err) => error_json(
            StatusCode::UNPROCESSABLE_ENTITY,
            err.to_string(),
            Some(err.field().to_string()),
        ),
        RunError::Engine(err) => {
            error_json(StatusCode::INTERNAL_SERVER_ERROR, err.to_string(), None)
        }
    }
}

fn error_json(status: StatusCode, message: String, field: Option<String>) -> HttpResponse {
    #[derive(Serialize)]
    struct Body {
        error: Detail,
    }
    #[derive(Serialize)]
    struct Detail {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        field: Option<String>,
    }
    (
        status,
        Json(Body {
            error: Detail { message, field },
        }),
    )
        .into_response()
}

fn batch_loop(mut engine: Engine, model_name: String, rx: Receiver<Job>, parallel: usize) {
    while let Some(jobs) = gather(&rx, parallel) {
        if jobs.len() > 1 {
            eprintln!("[bonjev] batch {}", jobs.len());
        }
        let mut replies = Vec::with_capacity(jobs.len());
        let mut reqs = Vec::with_capacity(jobs.len());
        for job in jobs {
            replies.push(job.reply);
            reqs.push(job.req);
        }
        let results = decision::run_many(&mut engine, &model_name, reqs);
        for (reply, result) in replies.into_iter().zip(results) {
            let _ = reply.send(result);
        }
    }
}

fn gather(rx: &Receiver<Job>, parallel: usize) -> Option<Vec<Job>> {
    let first = rx.recv().ok()?;
    let mut jobs = vec![first];
    if parallel <= 1 {
        return Some(jobs);
    }
    let deadline = Instant::now() + Duration::from_millis(40);
    while jobs.len() < parallel {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match rx.recv_timeout(left) {
            Ok(job) => jobs.push(job),
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    Some(jobs)
}

fn decide(st: &AppState, req: Request) -> Result<Response, RunError> {
    let Slot::Direct(engine) = &st.slot else {
        return Err(anyhow::anyhow!("direct decide on a batch server").into());
    };
    let mut engine = engine_lock(engine);
    decision::run(&mut engine, &st.model_name, req)
}

fn engine_lock(engine: &Mutex<Engine>) -> MutexGuard<'_, Engine> {
    engine.lock().unwrap_or_else(|err| err.into_inner())
}
