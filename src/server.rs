use crate::decision::{self, Request, Response};
use crate::engine::Engine;
use anyhow::Result;
use axum::http::StatusCode;
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
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
    reply: Sender<Result<Response>>,
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
    Json(req): Json<Request>,
) -> Result<Json<Response>, (StatusCode, String)> {
    let result = if matches!(st.slot, Slot::Batch(_)) {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        {
            let Slot::Batch(tx) = &st.slot else {
                unreachable!("matched batch");
            };
            if tx
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .send(Job { req, reply: reply_tx })
                .is_err()
            {
                return Err((StatusCode::INTERNAL_SERVER_ERROR, "batch worker stopped".to_string()));
            }
        }
        match tokio::task::spawn_blocking(move || reply_rx.recv()).await {
            Ok(Ok(body)) => body,
            Ok(Err(_)) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "batch worker dropped the reply".to_string(),
                ))
            }
            Err(err) => return Err((StatusCode::INTERNAL_SERVER_ERROR, err.to_string())),
        }
    } else {
        match tokio::task::spawn_blocking(move || decide(&st, req)).await {
            Ok(body) => body,
            Err(err) => return Err((StatusCode::INTERNAL_SERVER_ERROR, err.to_string())),
        }
    };
    match result {
        Ok(body) => Ok(Json(body)),
        Err(err) => Err((StatusCode::BAD_REQUEST, err.to_string())),
    }
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

fn decide(st: &AppState, req: Request) -> Result<Response> {
    let Slot::Direct(engine) = &st.slot else {
        anyhow::bail!("direct decide on a batch server");
    };
    let mut engine = engine_lock(engine);
    decision::run(&mut engine, &st.model_name, req)
}

fn engine_lock(engine: &Mutex<Engine>) -> MutexGuard<'_, Engine> {
    engine.lock().unwrap_or_else(|err| err.into_inner())
}
