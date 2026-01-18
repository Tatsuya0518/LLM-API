use serde::{Deserialize, Serialize};

/// Configuration for the LLM model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Hidden dimension size
    pub hidden_size: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Feed-forward network dimension
    pub ff_dim: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Dropout probability
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,      // Typical tokenizer vocab size
            num_layers: 6,          // Small model for testing
            hidden_size: 512,
            num_heads: 8,
            ff_dim: 2048,
            max_seq_len: 512,
            dropout: 0.1,
        }
    }
}

impl ModelConfig {
    /// Create a tiny model configuration for quick testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            num_layers: 2,
            hidden_size: 128,
            num_heads: 4,
            ff_dim: 512,
            max_seq_len: 128,
            dropout: 0.1,
        }
    }

    /// Create a small model configuration
    pub fn small() -> Self {
        Self::default()
    }

    /// Create a medium model configuration
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            num_layers: 12,
            hidden_size: 768,
            num_heads: 12,
            ff_dim: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size % self.num_heads != 0 {
            return Err(format!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            ));
        }
        Ok(())
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Number of epochs
    pub num_epochs: usize,

    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,

    /// Warmup steps
    pub warmup_steps: usize,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,

    /// Save checkpoint every N steps
    pub save_every_n_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            learning_rate: 3e-4,
            num_epochs: 10,
            gradient_accumulation_steps: 4,
            warmup_steps: 1000,
            max_grad_norm: 1.0,
            save_every_n_steps: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = ModelConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = ModelConfig {
            hidden_size: 513, // Not divisible by num_heads (8)
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }
}
