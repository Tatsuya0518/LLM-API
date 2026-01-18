# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Training Framework using Burn ML framework with WGPU backend, specifically optimized for Intel Arc A770 GPU. This is a Rust-based project for training Large Language Models.

## Architecture

### Core Components (defined in `src/lib.rs`)

- **Backend Type**: Uses WGPU backend with f32 precision (`burn::backend::wgpu::Wgpu<f32, i32>`)
- **Module Structure**: The library expects four main modules (currently defined but not yet implemented):
  - `config`: Model configuration (re-exports `ModelConfig`)
  - `data`: Data loading and preprocessing pipeline
  - `model`: LLM model architecture
  - `training`: Training loop and utilities
- **Device Initialization**: `init_device()` function initializes WGPU device for Intel Arc A770, uses `WGPU_BACKEND` environment variable for backend selection

### Dependencies

Key dependencies from `Cargo.toml`:
- **Burn**: v0.15 with `wgpu`, `train`, and `dataset` features for ML framework
- **burn-wgpu**: WGPU backend for Burn
- **burn-ndarray**: Tensor operations
- **tokenizers**: v0.20 with `onig` feature for text tokenization
- **tokio**: Async runtime with full features
- **tracing**: Logging infrastructure

## Common Commands

### Build
```bash
cargo build
cargo build --release
```

### Run
```bash
cargo run
cargo run --release
```

### Test
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_device_initialization

# Run tests with logging output
cargo test -- --nocapture
```

### Lint
```bash
cargo clippy
cargo clippy -- -D warnings
```

### Format
```bash
cargo fmt
cargo fmt -- --check
```

## Code Standards

### Rust Coding Conventions
**MUST strictly adhere to the Rust Style Guide**: https://doc.rust-lang.org/style-guide/#rust-style-guide

Key requirements:
- Follow official Rust formatting conventions
- Use `cargo fmt` to ensure consistent formatting
- Follow Rust naming conventions (snake_case for functions/variables, PascalCase for types)
- Write idiomatic Rust code

### Python Coding Conventions
**MUST strictly adhere to PEP 8**: https://peps.python.org/pep-0008/

(Note: Currently no Python files in the project, but this standard applies if Python is added)

### Communication Standards
- **思考は英語で会話は日本語で厳守すること** (Think in English, communicate in Japanese)
- Internal reasoning and code comments should be in English
- User-facing messages and documentation should be in Japanese

## GPU Configuration

The project is optimized for Intel Arc A770 GPU. Set the WGPU backend using:
```bash
set WGPU_BACKEND=vulkan  # Windows
export WGPU_BACKEND=vulkan  # Linux/macOS
```

## Build Profile

Release builds use aggressive optimization:
- `opt-level = 3`: Maximum optimization
- `lto = true`: Link-time optimization enabled
- `codegen-units = 1`: Single codegen unit for better optimization
