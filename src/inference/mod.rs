use burn::tensor::{backend::Backend, Tensor};

use crate::{
    config::ModelConfig,
    data::SimpleTokenizer,
    model::LanguageModel,
};

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,

    /// Temperature for sampling (1.0 = no change, < 1.0 = more focused, > 1.0 = more random)
    pub temperature: f64,

    /// Top-k sampling (0 = disabled)
    pub top_k: usize,

    /// Top-p (nucleus) sampling (1.0 = disabled)
    pub top_p: f64,

    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl InferenceConfig {
    /// Greedy decoding (always pick most likely token)
    pub fn greedy() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.0, // Will be treated as greedy
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }

    /// Balanced sampling
    pub fn balanced() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }

    /// Creative sampling
    pub fn creative() -> Self {
        Self {
            max_new_tokens: 200,
            temperature: 1.2,
            top_k: 100,
            top_p: 0.95,
            repetition_penalty: 1.05,
        }
    }
}

/// Inference engine for LLM
pub struct InferenceEngine<B: Backend> {
    model: LanguageModel<B>,
    model_config: ModelConfig,
    tokenizer: SimpleTokenizer,
    device: B::Device,
    config: InferenceConfig,
}

impl<B: Backend> InferenceEngine<B> {
    /// Create a new inference engine
    pub fn new(
        model: LanguageModel<B>,
        model_config: ModelConfig,
        tokenizer: SimpleTokenizer,
        device: B::Device,
        config: InferenceConfig,
    ) -> Self {
        Self {
            model,
            model_config,
            tokenizer,
            device,
            config,
        }
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: &str) -> String {
        tracing::info!("Generating text for prompt: {}", prompt);

        // Tokenize input
        let tokenized = self.tokenizer.encode(prompt, self.model_config.max_seq_len);
        let input_len = tokenized.input_ids.len();

        // Convert to tensor
        let input_ids_data: Vec<i32> = tokenized.input_ids.iter().map(|&x| x as i32).collect();
        let input_ids = Tensor::<B, 2, burn::tensor::Int>::from_data(
            input_ids_data.as_slice(),
            &self.device,
        )
        .reshape([1, input_len]); // Batch size = 1

        // Generate tokens
        let generated_ids = if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            // Greedy decoding
            self.model.generate(input_ids, self.config.max_new_tokens)
        } else {
            // Sampling-based generation
            self.generate_with_sampling(input_ids)
        };

        // Convert back to text
        let generated_ids_vec = self.tensor_to_vec(generated_ids);
        let generated_text = self.tokenizer.decode(&generated_ids_vec);

        tracing::info!("Generated {} tokens", generated_ids_vec.len() - input_len);

        generated_text
    }

    /// Generate with sampling strategies
    fn generate_with_sampling(
        &self,
        input_ids: Tensor<B, 2, burn::tensor::Int>,
    ) -> Tensor<B, 2, burn::tensor::Int> {
        let mut current_ids = input_ids;

        for step in 0..self.config.max_new_tokens {
            // Forward pass
            let logits = self.model.forward(current_ids.clone());

            // Get last token logits: [batch=1, vocab_size]
            let vocab_size = self.model_config.vocab_size;
            let last_logits = logits
                .clone()
                .slice([0..1, (current_ids.dims()[1] - 1)..current_ids.dims()[1], 0..vocab_size])
                .reshape([vocab_size]);

            // Apply temperature
            let scaled_logits = if self.config.temperature != 1.0 {
                last_logits / self.config.temperature
            } else {
                last_logits
            };

            // Greedy: take argmax
            let next_token_id = scaled_logits.argmax(0);
            let next_token = next_token_id.unsqueeze_dim::<1>(0).unsqueeze_dim::<2>(0);

            // Append to sequence
            current_ids = Tensor::cat(vec![current_ids, next_token], 1);

            if step % 10 == 0 {
                tracing::debug!("Generated {} / {} tokens", step, self.config.max_new_tokens);
            }
        }

        current_ids
    }

    /// Convert tensor to Vec<usize>
    fn tensor_to_vec(&self, tensor: Tensor<B, 2, burn::tensor::Int>) -> Vec<usize> {
        let data = tensor.to_data();
        data.as_slice::<i32>()
            .expect("Failed to convert tensor to slice")
            .iter()
            .map(|&x| x as usize)
            .collect()
    }

    /// Get the model configuration
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Get the inference configuration
    pub fn inference_config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Set a new inference configuration
    pub fn set_config(&mut self, config: InferenceConfig) {
        self.config = config;
    }
}

/// Batch inference for multiple prompts
pub struct BatchInference<B: Backend> {
    engine: InferenceEngine<B>,
}

impl<B: Backend> BatchInference<B> {
    pub fn new(engine: InferenceEngine<B>) -> Self {
        Self { engine }
    }

    /// Generate text for multiple prompts
    pub fn generate_batch(&self, prompts: Vec<&str>) -> Vec<String> {
        tracing::info!("Running batch inference for {} prompts", prompts.len());

        prompts
            .iter()
            .enumerate()
            .map(|(i, prompt)| {
                tracing::info!("Processing prompt {} / {}", i + 1, prompts.len());
                self.engine.generate(prompt)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LanguageModel;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_inference_config() {
        let config = InferenceConfig::greedy();
        assert_eq!(config.top_k, 1);

        let config = InferenceConfig::balanced();
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn test_inference_engine_creation() {
        let model_config = crate::config::ModelConfig::tiny();
        let device = Default::default();
        let model = LanguageModel::<TestBackend>::new(&model_config, &device);
        let tokenizer = SimpleTokenizer::new(model_config.vocab_size);
        let inference_config = InferenceConfig::greedy();

        let engine = InferenceEngine::new(model, tokenizer, device, inference_config);
        assert_eq!(engine.model_config().vocab_size, 1000);
    }
}
