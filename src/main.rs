mod decision;
mod engine;
mod ffi;
mod hf;
mod picture;
mod prompt;
mod readout;
mod server;
mod yaml_emit;

use anyhow::{Result, bail};
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
    /// List the decision LoRA adapters for each model
    Loras(LorasArgs),
}

#[derive(Args)]
struct LorasArgs {
    /// Only show adapters for this model (canonical id or alias)
    model: Option<String>,
}

#[derive(Args)]
struct ModelArgs {
    /// Commercial Bonsai / Ternary-Bonsai GGUF ids (see `bonjev models`).
    /// Examples: ternary-bonsai-2-27b (default), bonsai-27b, bonsai-8b, bonsai-4b,
    /// bonsai-1.7b, ternary-bonsai-8b, ternary-bonsai-4b, ternary-bonsai-1.7b,
    /// ternary-bonsai-27b. Legacy aliases: bonsai2, bonsai, bonsai8, q1, q2, 8b, …
    #[arg(long, default_value = "bonsai2")]
    model: String,
    #[arg(long, default_value_t = 32768)]
    ctx: u32,
    #[arg(long, default_value_t = 4)]
    threads: u32,
    #[arg(long, default_value_t = 99)]
    ngl: i32,
    /// Use a local GGUF instead of the HF cache
    #[arg(long)]
    model_path: Option<PathBuf>,
    /// Vision projector. The model's Q8 mmproj is downloaded when this is omitted.
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Do not load a vision projector.
    #[arg(long, default_value_t = false)]
    no_vision: bool,
    /// LoRA adapter(s): a short name (see `bonjev loras`) or a local `.gguf` path.
    /// Repeatable; append `,scale` to amplify (task arithmetic). Up to 8
    /// adapters; overrides `BONJEV_LORA`.
    #[arg(long)]
    lora: Vec<String>,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    common: ModelArgs,
    #[arg(long, default_value_t = 8080)]
    port: u16,
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
    eprintln!("No model in the cache. Which one should I download?");
    for model in hf::models() {
        eprintln!("  {}  {}", model.name, model.about);
    }
    if !io::stdin().is_terminal() {
        eprintln!("stdin is not a terminal; using {}", default_model.name);
        return Ok(default_model);
    }
    loop {
        eprint!("Model [{}]: ", default_model.name);
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

/// Resolve `--lora` (or `BONJEV_LORA`) into a shim-ready path list.
fn resolve_lora(model_name: &str, cli: &[String]) -> Result<Option<String>> {
    let env = std::env::var("BONJEV_LORA").ok();
    let entries = hf::split_lora_entries(cli, env.as_deref());
    if entries.is_empty() {
        return Ok(None);
    }
    Ok(Some(hf::resolve_lora_spec(model_name, &entries.join(";"))?))
}

fn resolve_mmproj(args: &ModelArgs, model_name: &str) -> Result<Option<PathBuf>> {
    if args.no_vision {
        return Ok(None);
    }
    let path = if let Some(path) = &args.mmproj {
        if !path.is_file() {
            bail!("vision projector not found: {}", path.display());
        }
        path.clone()
    } else {
        let model = hf::canonical(model_name)?;
        let Some(projector) = model.mmproj else {
            eprintln!(
                "[bonjev] '{name}' has no vision projector, vision disabled",
                name = model.name
            );
            return Ok(None);
        };
        eprintln!("[bonjev] vision projector: {projector}");
        hf::mmproj(model)?
    };
    eprintln!("[bonjev] vision: {}", path.display());
    Ok(Some(path))
}

fn load_engine(args: &ModelArgs) -> Result<(engine::Engine, String)> {
    let (path, name) = prepare_model(&args.model, args.model_path.clone())?;
    eprintln!("[bonjev] model: {}", path.display());
    let lora = resolve_lora(name, &args.lora)?;
    let spec = engine::LoadSpec {
        model_path: path,
        ctx: args.ctx,
        threads: args.threads,
        n_gpu_layers: args.ngl,
        mmproj: resolve_mmproj(args, name)?,
        lora,
    };
    let (engine, ctx) = spec.load_from_ctx(args.ctx)?;
    eprintln!("[bonjev] loaded: vocab={} ctx={}", engine.n_vocab, ctx);
    Ok((engine, name.to_string()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve(args) => {
            let (path, name) = prepare_model(&args.common.model, args.common.model_path.clone())?;
            eprintln!("[bonjev] model: {}", path.display());
            let lora = resolve_lora(name, &args.common.lora)?;
            let spec = engine::LoadSpec {
                model_path: path,
                ctx: args.common.ctx,
                threads: args.common.threads,
                n_gpu_layers: args.common.ngl,
                mmproj: resolve_mmproj(&args.common, name)?,
                lora,
            };
            let model_name = name.to_string();
            let state = Arc::new(server::direct(spec, model_name)?);
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
            let out = decision::run(&mut engine, &model_name, &req)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Cmd::Models => {
            for model in hf::models() {
                let status = match hf::cached(model) {
                    Some(path) => format!("cached: {}", path.display()),
                    None => "not cached".to_string(),
                };
                println!(
                    "{}  {}\n     {}/{}\n     {status}",
                    model.name,
                    model.about,
                    model.repo,
                    model.file
                );
            }
        }
        Cmd::Loras(args) => {
            let filter = match &args.model {
                Some(name) => Some(hf::canonical(name)?.name),
                None => None,
            };
            for lora in hf::loras() {
                if filter.is_some_and(|name| !lora.bases.contains(&name)) {
                    continue;
                }
                println!(
                    "{}  [{}]\n     {}/{}\n     {}",
                    lora.name,
                    lora.bases.join(", "),
                    hf::lora_repo(lora),
                    lora.file,
                    lora.about
                );
            }
            println!();
            println!("{}", hf::LORA_NOTE);
        }
    }
    Ok(())
}
