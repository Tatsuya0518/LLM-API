use burn::{
    optim::{AdamConfig, Optimizer},
    tensor::{backend::Backend, Tensor},
    train::{
        metric::{LossMetric, Adaptor},
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

use crate::{
    config::{ModelConfig, TrainingConfig},
    data::{SimpleTokenizer, TextDataset, TokenizedItem},
    model::LanguageModel,
};

/// Training batch
pub struct TrainingBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, burn::tensor::Int>,
    pub labels: Tensor<B, 2, burn::tensor::Int>,
}

/// Trainer for the language model
pub struct Trainer<B: Backend> {
    model: LanguageModel<B>,
    tokenizer: SimpleTokenizer,
    config: TrainingConfig,
    device: B::Device,
}

impl<B: Backend> Trainer<B> {
    /// Create a new trainer
    pub fn new(
        model_config: &ModelConfig,
        training_config: TrainingConfig,
        device: B::Device,
    ) -> Self {
        let model = LanguageModel::new(model_config, &device);
        let tokenizer = SimpleTokenizer::new(model_config.vocab_size);

        Self {
            model,
            tokenizer,
            config: training_config,
            device,
        }
    }

    /// Prepare a batch from text dataset
    pub fn prepare_batch(&self, texts: Vec<String>, max_length: usize) -> TrainingBatch<B> {
        let tokenized: Vec<TokenizedItem> = texts
            .iter()
            .map(|text| self.tokenizer.encode(text, max_length))
            .collect();

        let batch_size = tokenized.len();

        // Convert to tensors
        let input_ids_data: Vec<i32> = tokenized
            .iter()
            .flat_map(|item| item.input_ids.iter().map(|&x| x as i32))
            .collect();

        let input_ids =
            Tensor::from_data(input_ids_data.as_slice(), &self.device).reshape([batch_size, max_length]);

        // For language modeling, labels are shifted input_ids
        let labels = input_ids.clone();

        TrainingBatch { input_ids, labels }
    }

    /// Compute cross-entropy loss
    pub fn compute_loss(
        &self,
        logits: Tensor<B, 3>,
        labels: Tensor<B, 2, burn::tensor::Int>,
    ) -> Tensor<B, 1> {
        // logits: [batch, seq_len, vocab_size]
        // labels: [batch, seq_len]

        let [batch_size, seq_len, vocab_size] = logits.dims();

        // Reshape logits to [batch * seq_len, vocab_size]
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);

        // Reshape labels to [batch * seq_len]
        let labels_flat = labels.reshape([batch_size * seq_len]);

        // Compute cross-entropy loss
        let loss = burn::tensor::loss::cross_entropy_with_logits(logits_flat, labels_flat);

        loss
    }

    /// Single training step
    pub fn train_step(&mut self, batch: TrainingBatch<B>) -> f32 {
        // Forward pass
        let logits = self.model.forward(batch.input_ids.clone());

        // Compute loss
        let loss = self.compute_loss(logits, batch.labels);

        // Extract loss value for logging
        let loss_value = loss.clone().into_scalar().elem::<f32>();

        loss_value
    }

    /// Get the model reference
    pub fn model(&self) -> &LanguageModel<B> {
        &self.model
    }

    /// Get the model mutably
    pub fn model_mut(&mut self) -> &mut LanguageModel<B> {
        &mut self.model
    }
}

/// Simple training loop
pub fn train<B: Backend>(
    model_config: &ModelConfig,
    training_config: &TrainingConfig,
    dataset: TextDataset,
    device: B::Device,
) -> anyhow::Result<LanguageModel<B>> {
    tracing::info!("Starting training...");
    tracing::info!("Model config: {:?}", model_config);
    tracing::info!("Training config: {:?}", training_config);

    let mut trainer = Trainer::<B>::new(model_config, training_config.clone(), device);

    let num_batches = dataset.len() / training_config.batch_size;

    for epoch in 0..training_config.num_epochs {
        tracing::info!("Epoch {}/{}", epoch + 1, training_config.num_epochs);

        let mut total_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * training_config.batch_size;
            let end_idx = start_idx + training_config.batch_size;

            // Get batch texts
            let batch_texts: Vec<String> = (start_idx..end_idx)
                .filter_map(|i| dataset.get(i).map(|item| item.text))
                .collect();

            if batch_texts.is_empty() {
                continue;
            }

            // Prepare batch
            let batch = trainer.prepare_batch(batch_texts, model_config.max_seq_len);

            // Training step
            let loss = trainer.train_step(batch);
            total_loss += loss;

            if batch_idx % 10 == 0 {
                tracing::info!("  Batch {}/{}, Loss: {:.4}", batch_idx, num_batches, loss);
            }
        }

        let avg_loss = total_loss / num_batches as f32;
        tracing::info!("Epoch {} completed. Average loss: {:.4}", epoch + 1, avg_loss);
    }

    tracing::info!("Training completed!");

    Ok(trainer.model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_trainer_creation() {
        let model_config = ModelConfig::tiny();
        let training_config = TrainingConfig::default();
        let device = Default::default();

        let trainer = Trainer::<TestBackend>::new(&model_config, training_config, device);
        assert_eq!(trainer.model.config().vocab_size, 1000);
    }

    #[test]
    fn test_batch_preparation() {
        let model_config = ModelConfig::tiny();
        let training_config = TrainingConfig::default();
        let device = Default::default();

        let trainer = Trainer::<TestBackend>::new(&model_config, training_config, device);

        let texts = vec!["Hello world".to_string(), "Test text".to_string()];
        let batch = trainer.prepare_batch(texts, 20);

        assert_eq!(batch.input_ids.dims(), [2, 20]);
        assert_eq!(batch.labels.dims(), [2, 20]);
    }
}
