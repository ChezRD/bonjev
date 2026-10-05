use crate::decision::{self, Request, Response, RunError};
use crate::engine::{Engine, EngineFault, EngineFaultKind, LoadSpec, MIN_VRAM_CTX};
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde::Serialize;
use std::sync::{Arc, Mutex, MutexGuard};

struct Runtime {
    spec: LoadSpec,
    engine: Option<Engine>,
    active_ctx: u32,
}

impl Runtime {
    fn ensure_loaded(&mut self) -> Result<(), RunError> {
        if self.engine.is_some() {
            return Ok(());
        }
        let (engine, ctx) = self
            .spec
            .load_from_ctx(self.active_ctx)
            .map_err(RunError::Engine)?;
        self.active_ctx = ctx;
        eprintln!(
            "[bonjev] loaded: vocab={} ctx={}",
            engine.n_vocab, self.active_ctx
        );
        self.engine = Some(engine);
        Ok(())
    }

    fn unload(&mut self) {
        if self.engine.is_some() {
            eprintln!("[bonjev] unloading active model from VRAM");
        }
        self.engine = None;
        crate::readout::reset_prior();
    }

    fn engine_mut(&mut self) -> Result<&mut Engine, RunError> {
        self.ensure_loaded()?;
        Ok(self.engine.as_mut().expect("engine loaded"))
    }

    fn recoverable(err: &RunError) -> bool {
        match err {
            RunError::Engine(inner) => inner
                .downcast_ref::<EngineFault>()
                .is_some_and(|fault| fault.kind.recoverable()),
            _ => false,
        }
    }

    fn run(&mut self, model_name: &str, req: &Request) -> Result<Response, RunError> {
        self.ensure_loaded()?;
        match decision::run(self.engine_mut()?, model_name, req) {
            ok @ Ok(_) => ok,
            Err(err) if Self::recoverable(&err) && self.active_ctx > MIN_VRAM_CTX => {
                eprintln!(
                    "[bonjev] inference error, unloading and retrying with smaller ctx: {err}"
                );
                self.unload();
                self.active_ctx = (self.active_ctx / 2).max(MIN_VRAM_CTX);
                self.ensure_loaded()?;
                decision::run(self.engine_mut()?, model_name, req)
            }
            Err(err) => Err(err),
        }
    }
}

pub struct AppState {
    model_name: String,
    runtime: Mutex<Runtime>,
}

pub fn direct(spec: LoadSpec, model_name: String) -> Result<AppState, RunError> {
    let start_ctx = spec.ctx;
    let mut runtime = Runtime {
        spec,
        engine: None,
        active_ctx: start_ctx,
    };
    runtime.ensure_loaded()?;
    Ok(AppState {
        model_name,
        runtime: Mutex::new(runtime),
    })
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
    let result = match tokio::task::spawn_blocking(move || decide(&st, &req)).await {
        Ok(body) => body,
        Err(err) => {
            return error_json(StatusCode::INTERNAL_SERVER_ERROR, err.to_string(), None);
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
            let status = match err.downcast_ref::<EngineFault>() {
                Some(fault) if fault.kind == EngineFaultKind::TooLong => {
                    StatusCode::UNPROCESSABLE_ENTITY
                }
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            error_json(status, err.to_string(), None)
        }
    }
}

fn error_json(status: StatusCode, message: String, field: Option<String>) -> HttpResponse {
    #[derive(Serialize)]
    struct Body {
        detail: Vec<Detail>,
    }
    #[derive(Serialize)]
    struct Detail {
        loc: Vec<String>,
        msg: String,
        #[serde(rename = "type")]
        kind: &'static str,
    }
    // Official 422 shape: {"detail": [{"loc": ["body", ...], "msg", "type"}]}.
    let loc = match field {
        Some(path) => std::iter::once("body".to_string())
            .chain(path.split('.').map(str::to_string))
            .collect(),
        None => vec!["body".to_string()],
    };
    (
        status,
        Json(Body {
            detail: vec![Detail {
                loc,
                msg: message,
                kind: "value_error",
            }],
        }),
    )
        .into_response()
}

fn decide(st: &AppState, req: &Request) -> Result<Response, RunError> {
    runtime_lock(&st.runtime).run(&st.model_name, req)
}

fn runtime_lock(runtime: &Mutex<Runtime>) -> MutexGuard<'_, Runtime> {
    runtime.lock().unwrap_or_else(|err| err.into_inner())
}
