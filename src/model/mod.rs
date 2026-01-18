use burn::{
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

use crate::config::ModelConfig;

/// Transformer Block (Attention + FFN)
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ffn: FeedForward<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let attention_config = MultiHeadAttentionConfig::new(config.hidden_size, config.num_heads);
        let norm_config = LayerNormConfig::new(config.hidden_size);
        let dropout_config = DropoutConfig::new(config.dropout);

        Self {
            attention: attention_config.init(device),
            norm1: norm_config.init(device),
            ffn: FeedForward::new(config, device),
            norm2: norm_config.init(device),
            dropout: dropout_config.init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention with residual connection
        let mha_input = MhaInput::self_attn(x.clone());
        let attn_output = self.attention.forward(mha_input);
        let x = x + self.dropout.forward(attn_output.context);
        let x = self.norm1.forward(x);

        // Feed-forward with residual connection
        let ffn_output = self.ffn.forward(x.clone());
        let x = x + self.dropout.forward(ffn_output);
        let x = self.norm2.forward(x);

        x
    }
}

/// Feed-Forward Network
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let linear1_config = LinearConfig::new(config.hidden_size, config.ff_dim);
        let linear2_config = LinearConfig::new(config.ff_dim, config.hidden_size);
        let dropout_config = DropoutConfig::new(config.dropout);

        Self {
            linear1: linear1_config.init(device),
            linear2: linear2_config.init(device),
            dropout: dropout_config.init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

/// Main Language Model
#[derive(Module, Debug)]
pub struct LanguageModel<B: Backend> {
    embedding: Embedding<B>,
    blocks: Vec<TransformerBlock<B>>,
    norm: LayerNorm<B>,
    lm_head: Linear<B>,
    vocab_size: usize,
}

impl<B: Backend> LanguageModel<B> {
    /// Create a new language model
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let embedding_config = EmbeddingConfig::new(config.vocab_size, config.hidden_size);
        let norm_config = LayerNormConfig::new(config.hidden_size);
        let lm_head_config = LinearConfig::new(config.hidden_size, config.vocab_size);

        let blocks = (0..config.num_layers)
            .map(|_| TransformerBlock::new(config, device))
            .collect();

        Self {
            embedding: embedding_config.init(device),
            blocks,
            norm: norm_config.init(device),
            lm_head: lm_head_config.init(device),
            vocab_size: config.vocab_size,
        }
    }

    /// Forward pass
    /// Input shape: [batch_size, seq_len]
    /// Output shape: [batch_size, seq_len, vocab_size]
    pub fn forward(&self, input_ids: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        // Embedding: [batch, seq_len] -> [batch, seq_len, hidden]
        let mut x = self.embedding.forward(input_ids);

        // Apply transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Final layer norm
        let x = self.norm.forward(x);

        // Project to vocabulary
        self.lm_head.forward(x)
    }

    /// Generate text (simple greedy decoding)
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, burn::tensor::Int>,
        max_new_tokens: usize,
    ) -> Tensor<B, 2, burn::tensor::Int> {
        let mut current_ids = input_ids;

        for _ in 0..max_new_tokens {
            // Forward pass
            let logits = self.forward(current_ids.clone());

            // Get last token logits: [batch, 1, vocab_size]
            let [batch, seq_len, _vocab] = logits.dims();
            let last_logits = logits.clone().slice([0..batch, (seq_len - 1)..seq_len, 0..self.vocab_size]);

            // Greedy: take argmax on vocab dimension -> [batch, 1, 1], then squeeze to [batch, 1]
            let next_token = last_logits.argmax(2).squeeze::<2>(1);

            // Append to sequence
            current_ids = Tensor::cat(vec![current_ids, next_token], 1);
        }

        current_ids
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_model_creation() {
        let config = ModelConfig::tiny();
        let device = Default::default();
        let model = LanguageModel::<TestBackend>::new(&config, &device);

        assert_eq!(model.vocab_size(), 1000);
        assert_eq!(model.blocks.len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let config = ModelConfig::tiny();
        let device = Default::default();
        let model = LanguageModel::<TestBackend>::new(&config, &device);

        // Create dummy input: [batch_size=2, seq_len=10]
        let input_ids = Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([2, 10], &device);

        let output = model.forward(input_ids);

        // Output shape should be [2, 10, vocab_size]
        assert_eq!(output.dims(), [2, 10, config.vocab_size]);
    }
}
