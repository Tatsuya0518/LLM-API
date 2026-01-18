use llm_api::{
    config::ModelConfig,
    data::SimpleTokenizer,
    inference::{InferenceConfig, InferenceEngine},
    init_device, is_cuda_available,
    model::LanguageModel,
    Backend,
};
use tracing_subscriber;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("=== LLM Inference Framework ===");
    tracing::info!("Optimized for Google Colab with NVIDIA GPUs (T4, H100, A100)");

    // Check CUDA availability
    if !is_cuda_available() {
        tracing::warn!("CUDA is not available. Please ensure NVIDIA drivers and CUDA toolkit are installed.");
        tracing::warn!("This program requires CUDA-capable GPU to run.");
        return Ok(());
    }

    // Initialize CUDA device
    let device = init_device();
    tracing::info!("Successfully initialized CUDA device");

    // Create model configuration (tiny model for demonstration)
    let model_config = ModelConfig::tiny();
    tracing::info!("Model Configuration:");
    tracing::info!("  - Vocabulary Size: {}", model_config.vocab_size);
    tracing::info!("  - Hidden Size: {}", model_config.hidden_size);
    tracing::info!("  - Number of Layers: {}", model_config.num_layers);
    tracing::info!("  - Number of Heads: {}", model_config.num_heads);
    tracing::info!("  - Max Sequence Length: {}", model_config.max_seq_len);

    // Validate configuration
    model_config.validate()?;

    // Create model (in production, you would load pre-trained weights here)
    tracing::info!("Initializing model...");
    let model = LanguageModel::<Backend>::new(&model_config, &device);
    tracing::info!("Model initialized with {} parameters", estimate_params(&model_config));

    // Create tokenizer
    let tokenizer = SimpleTokenizer::new(model_config.vocab_size);

    // Create inference configuration
    let inference_config = InferenceConfig::greedy();

    // Create inference engine
    let engine = InferenceEngine::new(model, tokenizer, device, inference_config);

    // Run inference examples
    tracing::info!("=== Running Inference Examples ===");

    let prompts = vec![
        "Hello, world!",
        "The quick brown fox",
        "Once upon a time",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        tracing::info!("Example {} / {}", i + 1, prompts.len());
        tracing::info!("  Prompt: {}", prompt);

        let generated = engine.generate(prompt);
        tracing::info!("  Generated: {}", generated);
        println!("\nPrompt: {}\nGenerated: {}\n{}", prompt, generated, "-".repeat(50));
    }

    tracing::info!("=== Inference completed successfully! ===");
    tracing::info!("To use this in Google Colab:");
    tracing::info!("1. Upload this project to Colab");
    tracing::info!("2. Install Rust: !curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y");
    tracing::info!("3. Run: !cargo run --release");

    Ok(())
}

/// Estimate number of parameters in the model
fn estimate_params(config: &ModelConfig) -> String {
    // Embedding: vocab_size * hidden_size
    let embedding_params = config.vocab_size * config.hidden_size;

    // Each transformer block:
    // - Attention: 4 * hidden_size^2
    // - FFN: 2 * hidden_size * ff_dim
    // - LayerNorm: 2 * hidden_size (gamma and beta) x 2
    let per_block = 4 * config.hidden_size * config.hidden_size
        + 2 * config.hidden_size * config.ff_dim
        + 4 * config.hidden_size;

    let transformer_params = config.num_layers * per_block;

    // Output head: hidden_size * vocab_size
    let head_params = config.hidden_size * config.vocab_size;

    let total = embedding_params + transformer_params + head_params;

    if total >= 1_000_000_000 {
        format!("{:.2}B", total as f64 / 1_000_000_000.0)
    } else if total >= 1_000_000 {
        format!("{:.2}M", total as f64 / 1_000_000.0)
    } else if total >= 1_000 {
        format!("{:.2}K", total as f64 / 1_000.0)
    } else {
        format!("{}", total)
    }
}
