#!/bin/bash

# Google Colab環境セットアップスクリプト
# LLM Inference with Burn + CUDA

set -e

echo "=== LLM Inference Setup for Google Colab ==="

# 1. GPU確認
echo ""
echo "Step 1: Checking GPU..."
nvidia-smi || { echo "Error: CUDA GPU not found. Please enable GPU in Colab runtime settings."; exit 1; }

# 2. Rustのインストール
echo ""
echo "Step 2: Installing Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "Rust is already installed."
fi

rustc --version
cargo --version

# 3. CUDA環境変数の設定
echo ""
echo "Step 3: Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

echo "CUDA_HOME: $CUDA_HOME"
nvcc --version || echo "Warning: nvcc not found"

# 4. ビルド
echo ""
echo "Step 4: Building project (this may take 10-20 minutes)..."
cargo build --release

echo ""
echo "=== Setup completed successfully! ==="
echo ""
echo "To run inference:"
echo "  cargo run --release"
echo ""
echo "Supported GPUs:"
echo "  - Tesla T4 (Colab Free)"
echo "  - A100 (Colab Pro+)"
echo "  - H100 (Colab Enterprise)"
