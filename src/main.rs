// unc — Universal Neural Compiler
// Compiles HuggingFace models into optimised Metal inference binaries.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "unc", about = "Universal Neural Compiler", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a HuggingFace model to a .unc bundle or standalone binary
    Compile {
        #[arg(short, long)]
        model: String,
        #[arg(short, long, default_value = "f16")]
        quant: QuantArg,
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
        /// Produce a standalone Mach-O binary instead of a .unc bundle
        #[arg(long)]
        binary: bool,
        /// Generate unfused dispatches (MLX-style) with concurrent encoder + lazy barriers
        #[arg(long)]
        unfused: bool,
    },
    /// Run inference on a compiled .unc model
    Run {
        unc_file: PathBuf,
        #[arg(short, long, default_value = "Hello, ")]
        prompt: String,
        #[arg(short, long, default_value_t = 200)]
        max_tokens: usize,
    },
    /// Start an OpenAI-compatible API server
    Serve {
        unc_file: PathBuf,
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
    /// Inspect a .unc file (dump IR, memory plan, kernel list)
    Inspect {
        unc_file: PathBuf,
        #[arg(long)]
        dump_ir: bool,
    },
    /// List supported model architectures
    ListArchitectures,
}

#[derive(Clone, ValueEnum, Debug)]
enum QuantArg {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    match cli.command {
        Commands::Compile { model, quant, max_seq_len, output, hf_token, binary, unfused } => {
            cmd_compile(&model, quant, max_seq_len, output, hf_token, binary, unfused)
        }
        Commands::Run { unc_file, prompt, max_tokens } => cmd_run(&unc_file, &prompt, max_tokens),
        Commands::Serve { unc_file, port } => cmd_serve(&unc_file, port),
        Commands::Inspect { unc_file, dump_ir } => cmd_inspect(&unc_file, dump_ir),
        Commands::ListArchitectures => cmd_list_architectures(),
    }
}

// ---------------------------------------------------------------------------
// Subcommand handlers
// ---------------------------------------------------------------------------

fn cmd_compile(
    model_id: &str,
    quant_arg: QuantArg,
    max_seq_len: usize,
    mut output: PathBuf,
    hf_token: Option<String>,
    binary: bool,
    unfused: bool,
) -> anyhow::Result<()> {
    use unc::compile::PassPipeline;
    use unc::emit::metal::emit_metal;
    use unc::frontend::{config::parse_config, huggingface::ModelLoader, templates};
    use unc::ir::graph::ArchitectureFamily;
    use unc::kernel::{metal_mlx::register_metal_mlx_kernels, registry::KernelRegistry};
    use unc::target::Target;
    use unc::unc_format::write_unc;

    if !binary && output.extension().map_or(true, |e| e != "unc") {
        output.set_extension("unc");
    }

    println!("unc: compiling {model_id} → {}", output.display());

    // 1. Download model files
    let loader = if let Some(token) = hf_token {
        let cache = dirs::cache_dir().unwrap_or_default().join("huggingface");
        ModelLoader::with_cache(cache, Some(token))?
    } else {
        ModelLoader::new()?
    };
    let model_files = loader.load(model_id)?;

    // 2. Parse config.json
    let (arch, mut params, bos_token_id, eos_token_id) = parse_config(&model_files.config_json)?;
    params.max_position_embeddings = max_seq_len;

    println!(
        "  arch={:?}, layers={}, heads={} (kv={}), hidden={}, ffn={}",
        arch, params.num_hidden_layers, params.num_attention_heads,
        params.num_kv_heads, params.hidden_size, params.intermediate_size
    );

    // 3. Build computation graph
    let tokenizer_path = model_files.tokenizer_json.as_deref()
        .map(|p| p.to_string_lossy().into_owned());
    let tok_path = tokenizer_path.as_deref();
    let quant_str = match quant_arg {
        QuantArg::F32 => "f32", QuantArg::F16 => "f16", QuantArg::Bf16 => "bf16",
        QuantArg::Q8_0 => "q8_0",
        QuantArg::Q4_0 => "q4_0",
    };
    let mut graph = match arch {
        ArchitectureFamily::LLaMA    => templates::llama::lower_llama(&params, &model_files.weight_files, model_id, tok_path, bos_token_id, eos_token_id),
        ArchitectureFamily::Mistral  => templates::mistral::lower_mistral(&params, &model_files.weight_files, model_id, tok_path, bos_token_id, eos_token_id),
        ArchitectureFamily::Qwen     => templates::qwen::lower_qwen(&params, &model_files.weight_files, model_id, tok_path, bos_token_id, eos_token_id),
        ArchitectureFamily::Phi      => templates::phi::lower_phi(&params, &model_files.weight_files, model_id, tok_path, bos_token_id, eos_token_id),
        ArchitectureFamily::Gemma    => templates::gemma::lower_gemma(&params, &model_files.weight_files, model_id, tok_path, bos_token_id, eos_token_id),
        ArchitectureFamily::GPTNeoX  => anyhow::bail!("GPTNeoX not yet supported"),
    };
    graph.metadata.quant = quant_str.to_string();
    println!("  graph: {} nodes, quant: {quant_str}", graph.num_nodes());

    // 4. Detect Metal target and build kernel registry
    let target: Target = {
        #[cfg(target_os = "macos")]
        {
            use unc::target::detect::detect_metal_target;
            Target::Metal(detect_metal_target()?)
        }
        #[cfg(not(target_os = "macos"))]
        anyhow::bail!("Metal target requires macOS")
    };

    let mut registry = KernelRegistry::new();
    match &target {
        Target::Metal(m) => register_metal_mlx_kernels(&mut registry, m),
    }
    println!("  registry: {} kernels for {}", registry.len(), target.name());

    // 5. Compile
    let pipeline = PassPipeline::default();
    let result = pipeline.run(graph, &registry, &target, &model_files.weight_files)?;
    println!(
        "  compile: {} nodes, {} kernels, activation={} MB, kv={} MB",
        result.stats.nodes_after,
        result.stats.unique_kernels,
        result.memory_plan.total_activation_bytes / (1024 * 1024),
        result.memory_plan.total_kv_cache_bytes / (1024 * 1024),
    );

    if binary {
        // AOT: produce standalone Mach-O binary
        use unc::emit::aot::emit_metal_aot;
        emit_metal_aot(&result, &model_files, &output)?;
        println!("Done → {} (standalone binary)", output.display());
    } else {
        // Bundle: produce .unc file
        let out_dir = output.with_extension("unc_build");
        let artifact = emit_metal(&result, &out_dir, unfused)?;
        write_unc(&result, &artifact, &output)?;
        println!("Done → {}", output.display());
    }
    Ok(())
}

fn cmd_run(unc_file: &PathBuf, prompt: &str, max_tokens: usize) -> anyhow::Result<()> {
    use unc::runtime::generate::run_generation;
    use unc::unc_format::read_unc;

    let bundle = read_unc(unc_file)?;

    match bundle.target_tag.trim() {
        "METAL" => {
            #[cfg(target_os = "macos")]
            {
                // Resolve metallib path relative to the .unc file's directory
                let unc_dir = unc_file.canonicalize()
                    .unwrap_or_else(|_| unc_file.clone())
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .to_path_buf();

                let metallib_path = bundle
                    .metallib_path
                    .as_deref()
                    .map(|p| {
                        let pb = std::path::PathBuf::from(p);
                        if pb.is_absolute() { pb } else { unc_dir.join(pb) }
                    })
                    .or_else(|| std::env::var("UNC_METALLIB_PATH").ok().map(std::path::PathBuf::from))
                    .unwrap_or_else(|| unc_file.with_extension("metallib"));
                run_generation(&bundle, &metallib_path, prompt, max_tokens)?;
            }
            #[cfg(not(target_os = "macos"))]
            anyhow::bail!("Metal target requires macOS");
        }
        tag => anyhow::bail!("unknown target tag: {tag:?}"),
    }
    Ok(())
}

fn cmd_serve(unc_file: &PathBuf, port: u16) -> anyhow::Result<()> {
    println!("unc serve {} on port {} (not yet implemented)", unc_file.display(), port);
    Ok(())
}

fn cmd_inspect(unc_file: &PathBuf, dump_ir: bool) -> anyhow::Result<()> {
    use unc::unc_format::read_unc;
    use unc::ir::printer::dump_graph;

    let bundle = read_unc(unc_file)?;
    println!("=== {} ===", unc_file.display());
    println!("model:  {}", bundle.graph.metadata.model_id);
    println!("arch:   {:?}", bundle.graph.metadata.architecture);
    println!("target: {}", bundle.target_tag.trim());
    println!("nodes:  {}", bundle.graph.num_nodes());
    println!("stats:  {}", bundle.stats);
    if dump_ir {
        println!("\n{}", dump_graph(&bundle.graph));
    }
    Ok(())
}

fn cmd_list_architectures() -> anyhow::Result<()> {
    println!("Supported architectures:");
    for (hf_name, family) in [
        ("LlamaForCausalLM",   "LLaMA, LLaMA-2, LLaMA-3"),
        ("MistralForCausalLM", "Mistral, Mixtral"),
        ("Qwen2ForCausalLM",   "Qwen, Qwen-2"),
        ("PhiForCausalLM",     "Phi-2, Phi-3"),
        ("GemmaForCausalLM",   "Gemma, Gemma-2"),
    ] {
        println!("  {hf_name:<30} {family}");
    }
    Ok(())
}
