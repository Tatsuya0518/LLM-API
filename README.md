# LLM Inference Framework

Google Colab上でNVIDIA GPU（T4、H100、A100）を使用した、RustとBurnフレームワークによるLLM推論システム

## 特徴

- **Burn ML Framework**: RustネイティブのML/DLフレームワーク
- **CUDA Backend**: NVIDIA GPU（T4、H100、A100）で高速推論
- **型安全**: Rustの型システムによる安全な実装
- **Google Colab対応**: 無料GPUで簡単に推論を実行
- **高性能**: ゼロコスト抽象化と最適化されたコンパイル

## 対応GPU

| GPU | VRAM | Colab Tier | モデルサイズ目安 |
|-----|------|------------|-----------------|
| **Tesla T4** | 16GB | Free | Tiny〜Small |
| **A100** | 40GB | Pro+ | Medium〜Large |
| **H100** | 80GB | Enterprise | Large〜Very Large |

## プロジェクト構成

```
LLM-API/
├── src/
│   ├── main.rs          # 推論エントリポイント
│   ├── lib.rs           # ライブラリルート（CUDA対応）
│   ├── config/          # モデル・推論設定
│   │   └── mod.rs
│   ├── model/           # Transformerモデル実装
│   │   └── mod.rs
│   ├── data/            # データセット・トークナイザー
│   │   └── mod.rs
│   ├── inference/       # 推論エンジン
│   │   └── mod.rs
│   └── training/        # 学習ループ（将来用）
│       └── mod.rs
├── .cargo/
│   └── config.toml      # CUDA設定
├── Cargo.toml           # 依存関係（Burn + CUDA）
├── colab_setup.ipynb    # Google Colabノートブック
├── setup_colab.sh       # セットアップスクリプト
└── README.md
```

## Google Colabでの使い方（推奨）

### 方法1: Jupyter Notebookを使用

1. **Google Colabを開く**
   - [Google Colab](https://colab.research.google.com/)にアクセス

2. **ノートブックをアップロード**
   - `colab_setup.ipynb`をアップロード

3. **ランタイムの設定**
   - ランタイム → ランタイムのタイプを変更
   - ハードウェアアクセラレータ: **GPU**
   - GPU種類: T4（無料）、A100（Pro+）、H100（Enterprise）

4. **セルを順番に実行**
   - GPU確認 → Rustインストール → プロジェクトビルド → 推論実行

### 方法2: シェルスクリプトを使用

```bash
# 1. Colabでノートブックを開く
# 2. 以下を実行

# プロジェクトをアップロード/クローン
!git clone https://github.com/YOUR_USERNAME/LLM-API.git
%cd LLM-API

# セットアップスクリプト実行
!bash setup_colab.sh

# 推論実行
!cargo run --release
```

## ローカル環境での使い方（CUDA環境がある場合）

### 必要な環境

**ハードウェア:**
- NVIDIA GPU（CUDA対応）
- 十分なVRAM（モデルサイズに依存）

**ソフトウェア:**
- Rust 1.70以上
- CUDA Toolkit 11.8以上
- NVIDIA Drivers

### セットアップ

```bash
# 1. Rustのインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. CUDA環境変数の確認
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# 3. ビルド
cargo build --release

# 4. 推論実行
cargo run --release
```

## モデル設定

`src/config/mod.rs`で定義されたプリセット:

```rust
// 極小モデル（T4で動作確認用）
let config = ModelConfig::tiny();
// パラメータ数: ~1.5M

// 小規模モデル（T4推奨）
let config = ModelConfig::small();
// パラメータ数: ~50M

// 中規模モデル（A100推奨）
let config = ModelConfig::medium();
// パラメータ数: ~500M
```

## 推論設定

`src/inference/mod.rs`で定義されたプリセット:

```rust
// Greedy Decoding（最も確率が高いトークンを選択）
let config = InferenceConfig::greedy();

// バランス型（品質と多様性のバランス）
let config = InferenceConfig::balanced();

// クリエイティブ型（多様な出力）
let config = InferenceConfig::creative();
```

## カスタム推論設定

```rust
use llm_api::inference::InferenceConfig;

let config = InferenceConfig {
    max_new_tokens: 200,      // 生成する最大トークン数
    temperature: 0.8,         // 温度（高いほどランダム）
    top_k: 50,                // Top-Kサンプリング
    top_p: 0.9,               // Nucleus sampling
    repetition_penalty: 1.1,  // 繰り返しペナルティ
};
```

## トラブルシューティング

### CUDAが見つからない（Colab）

```bash
# 環境変数を設定
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### GPUが認識されない

```bash
# GPU確認
nvidia-smi

# CUDA確認
nvcc --version
```

### メモリ不足エラー

**解決策:**
1. より小さいモデルを使用（`ModelConfig::tiny()`）
2. `max_seq_len`を短くする
3. より大きなGPU（A100、H100）を使用

### ビルドエラー

```bash
# キャッシュクリア
cargo clean

# 再ビルド
cargo build --release
```

## 開発コマンド

```bash
# ビルド
cargo build                # デバッグビルド
cargo build --release      # リリースビルド（最適化）

# テスト
cargo test                 # 全テスト実行
cargo test --lib           # ライブラリテストのみ

# リント
cargo clippy               # 静的解析

# フォーマット
cargo fmt                  # コード整形

# GPU確認
nvidia-smi                 # GPU情報表示
```

## パフォーマンスベンチマーク（参考値）

| GPU | モデル | 推論速度 | メモリ使用量 |
|-----|--------|---------|-------------|
| T4 | Tiny | ~100 tokens/s | ~500MB |
| T4 | Small | ~50 tokens/s | ~2GB |
| A100 | Medium | ~200 tokens/s | ~8GB |
| H100 | Large | ~500 tokens/s | ~20GB |

*実際の性能はモデルアーキテクチャとシーケンス長に依存します*

## 次のステップ

**現在実装済み:**
- ✅ CUDAバックエンド対応
- ✅ 推論エンジン
- ✅ Google Colab対応
- ✅ 複数GPUサポート（T4/A100/H100）

**今後の実装予定:**
- [ ] 実トークナイザー統合（HuggingFace tokenizers）
- [ ] 事前学習済みモデルのロード（safetensors形式）
- [ ] バッチ推論の最適化
- [ ] ストリーミング生成
- [ ] REST API（axum/actix-web）
- [ ] モデル量子化（INT8、FP16）
- [ ] マルチGPU推論

## ライセンス

MIT License

## 参考資料

- [Burn Framework](https://burn.dev/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Google Colab](https://colab.research.google.com/)
- [NVIDIA GPU Cloud](https://www.nvidia.com/ja-jp/gpu-cloud/)
