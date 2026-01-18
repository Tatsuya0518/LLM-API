// Increase recursion limit for type nesting
#![recursion_limit = "256"]

//! LLM Inference Framework using Burn and CUDA
//!
//! This library provides inference capabilities for Large Language Models
//! on NVIDIA GPUs (T4, H100, A100) using the CUDA backend.
//! Optimized for Google Colab environments.

pub mod config;
pub mod data;
pub mod inference;
pub mod model;
pub mod training;

// Re-export commonly used types
pub use burn::prelude::*;
pub use config::ModelConfig;

/// Type alias for CUDA backend with f32 precision
pub type Backend = burn::backend::cuda::Cuda<f32, i32>;

/// Type alias for CPU backend (fallback)
pub type CpuBackend = burn::backend::ndarray::NdArray<f32>;

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub device_id: usize,
}

/// Initialize the CUDA device for NVIDIA GPUs
/// Supports: T4, H100, A100, and other CUDA-capable GPUs
pub fn init_device() -> burn::backend::cuda::CudaDevice {
    tracing::info!("Initializing CUDA device for NVIDIA GPU");

    // Create default CUDA device (GPU 0)
    let device = burn::backend::cuda::CudaDevice::default();

    tracing::info!("CUDA device initialized successfully");
    tracing::info!("Device ID: {}", device.index);

    device
}

/// Initialize a specific CUDA device by ID
pub fn init_device_with_id(device_id: usize) -> burn::backend::cuda::CudaDevice {
    tracing::info!("Initializing CUDA device {}", device_id);

    let device = burn::backend::cuda::CudaDevice::new(device_id);

    tracing::info!("CUDA device {} initialized successfully", device_id);

    device
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    // Try to initialize a CUDA device
    std::panic::catch_unwind(|| {
        let _ = burn::backend::cuda::CudaDevice::default();
    })
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will pass on systems with CUDA, fail otherwise
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
    }
}
