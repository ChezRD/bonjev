mod decision;
mod engine;
mod features;
mod ffi;
mod hf;
mod picture;
mod prompt;
mod readout;
mod server;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "bonjev",
    version,
    about = "BonJev: one-step decisions on Bonsai GGUF (llama.cpp)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve POST /v1/systemone and GET /v1/models
    Serve(ServeArgs),
    /// One decision from the command line
    Ask(AskArgs),
    /// List known models and their Hugging Face cache status
    Models,
}

#[derive(Args)]
struct ModelArgs {
    /// q2 = Ternary-Bonsai-2-27B PQ2_0 (7.2 GB), q1 = Bonsai-27B Q1_0 (3.6 GB)
    #[arg(long, default_value = "q2")]
    model: String,
    #[arg(long, default_value_t = 32768)]
    ctx: u32,
    #[arg(long, default_value_t = 6)]
    threads: u32,
    #[arg(long, default_value_t = 99)]
    ngl: i32,
    /// Use a local GGUF instead of the HF cache
    #[arg(long)]
    model_path: Option<PathBuf>,
    /// Vision projector. For q2 the Q8 mmproj is downloaded when this is omitted.
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Do not load a vision projector.
    #[arg(long, default_value_t = false)]
    no_vision: bool,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    common: ModelArgs,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// How many prompts to score at once. 1 is one sequence.
    #[arg(long, default_value_t = 1)]
    parallel: u32,
}

#[derive(Args)]
struct AskArgs {
    #[command(flatten)]
    common: ModelArgs,
    #[arg(long)]
    state: String,
    #[arg(long)]
    question: String,
    /// Repeatable: --option "text with, comma" --option "another"
    #[arg(long = "option")]
    options: Vec<String>,
    #[arg(long, default_value = "choice")]
    kind: String,
    /// Image file. jpeg, png, gif, webp or bmp.
    #[arg(long)]
    image: Option<PathBuf>,
}

fn ask_model(default_model: &'static hf::ModelId) -> Result<&'static hf::ModelId> {
    eprintln!();
    eprintln!("Модели в кэше нет. Какую поставить?");
    for model in hf::models() {
        eprintln!("  {}  {}", model.name, model.about);
    }
    if !io::stdin().is_terminal() {
        eprintln!("stdin не терминал, ставлю {}", default_model.name);
        return Ok(default_model);
    }
    loop {
        eprint!("Модель [{}]: ", default_model.name);
        io::stderr().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            return Ok(default_model);
        }
        let picked = line.trim();
        let name = if picked.is_empty() {
            default_model.name
        } else {
            picked
        };
        match hf::canonical(name) {
            Ok(model) => return Ok(model),
            Err(err) => eprintln!("{err}"),
        }
    }
}

fn prepare_model(name: &str, model_path: Option<PathBuf>) -> Result<(PathBuf, &'static str)> {
    let requested = hf::canonical(name)?;
    if let Some(path) = model_path {
        if !path.is_file() {
            bail!("model file not found: {}", path.display());
        }
        return Ok((path, requested.name));
    }
    if let Some(path) = hf::cached(requested) {
        return Ok((path, requested.name));
    }
    let chosen = if hf::any_cached() {
        requested
    } else {
        ask_model(requested)?
    };
    eprintln!("[bonjev] downloading {} ({})", chosen.name, chosen.about);
    Ok((hf::download(chosen)?, chosen.name))
}

fn attach_vision(engine: &mut engine::Engine, args: &ModelArgs, model_name: &str) -> Result<()> {
    if args.no_vision {
        return Ok(());
    }
    let path = if let Some(path) = &args.mmproj {
        if !path.is_file() {
            bail!("vision projector not found: {}", path.display());
        }
        path.clone()
    } else if model_name == "q2" {
        eprintln!("[bonjev] vision projector: {}", hf::Q2_MMPROJ);
        hf::q2_mmproj()?
    } else {
        return Ok(());
    };
    eprintln!("[bonjev] vision: {}", path.display());
    engine.load_vision(&path, args.threads)?;
    eprintln!("[bonjev] vision marker: {}", engine.media_marker().unwrap_or(""));
    Ok(())
}

fn load_engine(args: &ModelArgs) -> Result<(engine::Engine, String)> {
    let (path, name) = prepare_model(&args.model, args.model_path.clone())?;
    eprintln!("[bonjev] model: {}", path.display());
    let mut engine = engine::Engine::load(&path, args.ctx, args.threads, args.ngl)?;
    attach_vision(&mut engine, args, name)?;
    eprintln!(
        "[bonjev] loaded: vocab={} ctx={}",
        engine.n_vocab, args.ctx
    );
    Ok((engine, format!("bonjev-{name}")))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve(args) => {
            let parallel = args.parallel.max(1);
            let (engine, model_name) = if parallel == 1 {
                load_engine(&args.common)?
            } else {
                let (path, name) = prepare_model(&args.common.model, args.common.model_path.clone())?;
                eprintln!("[bonjev] model: {}", path.display());
                let mut engine = engine::Engine::load_seqs(
                    &path,
                    args.common.ctx,
                    args.common.threads,
                    args.common.ngl,
                    parallel,
                )?;
                attach_vision(&mut engine, &args.common, name)?;
                eprintln!(
                    "[bonjev] loaded: vocab={} ctx={} parallel={parallel}",
                    engine.n_vocab, args.common.ctx
                );
                (engine, format!("bonjev-{name}"))
            };
            let state = Arc::new(if parallel == 1 {
                server::direct(engine, model_name)
            } else {
                server::batch(engine, model_name, parallel as usize)
            });
            let app = server::router(state);
            let addr = format!("127.0.0.1:{}", args.port);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            eprintln!("[bonjev] listening on http://{addr}  (POST /v1/systemone)");
            axum::serve(listener, app).await?;
        }
        Cmd::Ask(args) => {
            let (mut engine, model_name) = load_engine(&args.common)?;
            let mut req =
                decision::from_options(&args.state, &args.question, &args.kind, &args.options);
            if let Some(path) = &args.image {
                req.image_bytes = Some(std::fs::read(path)?);
            }
            let out = decision::run(&mut engine, &model_name, req)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Cmd::Models => {
            for model in hf::models() {
                let status = match hf::cached(model) {
                    Some(path) => format!("cached: {}", path.display()),
                    None => "not cached".to_string(),
                };
                println!(
                    "{:3}  {}/{}\n     {status}",
                    model.name, model.repo, model.file
                );
            }
        }
    }
    Ok(())
}
