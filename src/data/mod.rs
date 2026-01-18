use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};

/// A simple text dataset item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextItem {
    pub text: String,
}

/// Tokenized dataset item
#[derive(Debug, Clone)]
pub struct TokenizedItem {
    pub input_ids: Vec<usize>,
    pub attention_mask: Vec<usize>,
}

/// Simple in-memory text dataset
pub struct TextDataset {
    items: Vec<TextItem>,
}

impl TextDataset {
    /// Create a new text dataset from a vector of strings
    pub fn new(texts: Vec<String>) -> Self {
        let items = texts.into_iter().map(|text| TextItem { text }).collect();
        Self { items }
    }

    /// Create a dummy dataset for testing
    pub fn dummy(size: usize) -> Self {
        let texts = (0..size)
            .map(|i| format!("This is sample text number {}.", i))
            .collect();
        Self::new(texts)
    }

    /// Load dataset from a text file (one sample per line)
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let texts: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

        Ok(Self::new(texts))
    }

    /// Get the number of items in the dataset
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Simple tokenizer wrapper
pub struct SimpleTokenizer {
    vocab_size: usize,
}

impl SimpleTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Tokenize text (dummy implementation for now)
    /// TODO: Integrate with actual tokenizer (e.g., sentencepiece, tokenizers crate)
    pub fn encode(&self, text: &str, max_length: usize) -> TokenizedItem {
        // Simple character-level tokenization for demonstration
        let chars: Vec<usize> = text
            .chars()
            .take(max_length)
            .map(|c| (c as usize) % self.vocab_size)
            .collect();

        let input_ids = if chars.len() < max_length {
            let mut padded = chars.clone();
            padded.resize(max_length, 0); // Pad with 0
            padded
        } else {
            chars
        };

        let attention_mask = vec![1; input_ids.len()];

        TokenizedItem {
            input_ids,
            attention_mask,
        }
    }

    /// Decode tokens back to text (dummy implementation)
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&t| char::from_u32(t as u32).unwrap_or('?'))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_dataset() {
        let dataset = TextDataset::dummy(10);
        assert_eq!(dataset.len(), 10);

        let item = dataset.get(0).unwrap();
        assert_eq!(item.text, "This is sample text number 0.");
    }

    #[test]
    fn test_tokenizer() {
        let tokenizer = SimpleTokenizer::new(1000);
        let item = tokenizer.encode("Hello, world!", 20);

        assert_eq!(item.input_ids.len(), 20);
        assert_eq!(item.attention_mask.len(), 20);
    }
}
