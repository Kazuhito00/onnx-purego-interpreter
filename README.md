# ONNX Pure Go Interpreter

[![Go](https://img.shields.io/badge/Go-1.26+-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)<br>
CGo なし・アセンブリなし・ネイティブ依存なしの ピュアGo ONNX 推論ランタイムです。

> [!IMPORTANT]
> 本ライブラリはピュアGo実装であり、SIMD やネイティブライブラリによる最適化は行っていません。<br>
> そのため、推論性能は ONNX Runtime と比較しておおよそ 10〜50 倍程度遅くなります。
>
> 主な想定用途:
> - 小規模モデルの推論
> - クロスコンパイル前提の環境（cgo 非対応環境、WASM など）
> - ONNX グラフの解析・可視化・最適化検証
> - 教育・検証用途（ONNX の動作理解やデバッグ）
>
> 中〜大規模モデルの推論や、高スループット・低レイテンシが求められる実運用用途には適しておらず、<br>
> ONNX Runtime の代替として使用することは想定していません。

## Features

- **Pure Go** — `GOOS`/`GOARCH` を問わずクロスコンパイル可能
- **単一依存** — 外部依存は `google.golang.org/protobuf` のみ
- **主要オペレーター対応** — ONNX 標準約 200 個中、約 80% を実装
- **グラフ最適化** — Conv+BN 融合、GELU 融合、不要ノード除去など 11 パスを自動適用

## Installation

```bash
go get github.com/Kazuhito00/onnx-purego-interpreter
```

**動作要件:** Go 1.26+

## Quick Start

```go
package main

import (
	"fmt"
	"os"

	"github.com/Kazuhito00/onnx-purego-interpreter/onnx"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func main() {
	modelBytes, _ := os.ReadFile("model.onnx")

	sess, _ := onnx.NewSession(modelBytes)

	input := tensor.NewDense[float32](
		tensor.Shape{1, 3, 224, 224},
		make([]float32, 1*3*224*224),
	)

	outputs, _ := sess.Run(input)

	for name, t := range outputs {
		fmt.Printf("%s: shape=%v\n", name, t.Shape())
	}
}
```

## Usage

### 入力の渡し方

```go
// 位置指定: グラフ定義順に入力名を自動解決
outputs, err := sess.Run(inputTensor)
outputs, err := sess.Run(images, sizes)

// 名前指定: ヘルパー関数で簡潔に
outputs, err := sess.RunWithNames(onnx.Input("input_bgr", inputTensor))
outputs, err := sess.RunWithNames(onnx.Inputs("images", images, "sizes", sizes))
outputs, err := sess.RunWithNames(map[string]tensor.Tensor{"input": inputTensor})
```

### セッションオプション

```go
sess, err := onnx.NewSessionWithOptions(modelBytes,
	onnx.WithProgressLogger(os.Stdout),                    // ビルド/推論の進捗ログ
	onnx.WithDisabledOptimizationPasses("fuse_conv_silu"), // 特定パスを無効化
	onnx.WithKernelConfig(kc),                             // カーネル設定
)
```

## API Reference

| 関数 / メソッド | 説明 |
|---|---|
| `onnx.NewSession(onnxBytes)` | デフォルト設定でセッション作成 |
| `onnx.NewSessionWithOptions(onnxBytes, opts...)` | オプション付きセッション作成 |
| `Session.Run(tensors...)` | 推論実行（位置指定入力） |
| `Session.RunWithNames(inputs)` | 推論実行（名前指定入力） |
| `Session.RunDebug(inputs)` | デバッグ推論（verbose 出力） |
| `Session.InputNames()` | グラフ入力名の一覧を取得 |
| `Session.Warnings()` | opset 互換性の警告を取得 |
| `onnx.Input(name, tensor)` | 単一入力の map を作成 |
| `onnx.Inputs(name1, t1, ...)` | 複数入力の map を作成 |
| `tensor.NewDense[T](shape, data)` | 型付きテンソルを作成 |

### セッションオプション一覧

| オプション | 説明 |
|---|---|
| `WithProgressLogger(w)` | ビルド/推論の進捗ログを有効化 |
| `WithObserver(obs)` | ビルド/推論イベントの Observer を接続 |
| `WithNoOptimization()` | 全グラフ最適化を無効化 |
| `WithDisabledOptimizationPasses(names...)` | 指定パスのみ無効化 |
| `WithOnlyOptimizationPasses(names...)` | 指定パスのみ有効化（他は全無効） |
| `WithKernelConfig(kc)` | カーネルレベルの最適化を設定 |

## Architecture

`.onnx` バイト列から推論結果までの 7 段パイプライン:

```
.onnx bytes
  → Reader / Decoder          protobuf デシリアライズ
  → Frontend IR                protobuf 非依存のモデル表現
  → Canonical IR               正規化済み意味グラフ (Graph, Node, Initializer)
  → Analysis / Validation      グラフ整合性 + opset 互換性チェック
  → Optimization Passes        BN融合, Conv融合, GELU融合, 不要ノード除去
  → Execution Plan / Lowering  slot-based compiled plan + Arena 事前確保
  → Runtime / Kernels          Pure Go カーネルで推論実行
```

## Operator Coverage

ONNX 標準約 200 個中、約 80% を実装済み:

- **算術 / 線形代数** — Add, Sub, Mul, Div, MatMul, Gemm, ...
- **活性化** — Relu, Sigmoid, Softmax, GELU, ...
- **畳み込み / プーリング** — Conv, ConvTranspose, MaxPool, AveragePool, ...
- **正規化** — BatchNormalization, LayerNormalization, InstanceNormalization, ...
- **形状 / インデクシング** — Reshape, Transpose, Gather, Scatter, Slice, ...
- **RNN** — LSTM, GRU, RNN
- **制御フロー** — If, Loop, Scan
- **物体検出** — NonMaxSuppression, RoiAlign, DeformConv
- **Transformer** — Attention, RotaryEmbedding
- **量子化** — DequantizeLinear, QuantizeLinear, QLinearConv, ...
- **信号処理** — DFT, STFT, MelWeightMatrix

全一覧: [Operators.md](Operators.md)

## Graph Optimizations

全パスはデフォルトで有効です。デバッグやベンチマーク用にセッションオプションで制御できます。

| パス名 | 説明 |
|---|---|
| `materialize_constants` | Constant op を初期化子に変換 |
| `eliminate_dropout` | Dropout ノード除去 |
| `eliminate_identity` | Identity ノード除去 |
| `fuse_conv_batchnorm` | Conv + BatchNorm → Conv に融合 |
| `fuse_conv_add_bias` | Conv + Add(定数) → bias に折込み |
| `fuse_conv_activation` | Conv + ReLU/Clip/LeakyReLU → FusedConv |
| `fuse_conv_silu` | Conv + Sigmoid + Mul → Conv(SiLU) |
| `fuse_matmul_add_bias` | MatMul + Add → FusedMatMul |
| `fuse_mul_add_affine` | Mul + Add → FusedAffine |
| `fuse_gelu` | Div→Erf→Add→Mul→Mul → FastGELU |
| `eliminate_dead_nodes` | 未使用ノード除去 |

## Kernel Optimizations

`KernelConfig` で個々のカーネル最適化をオンオフできます:

```go
import "github.com/Kazuhito00/onnx-purego-interpreter/internal/ops"

kc := ops.DefaultKernelConfig()
kc.UseTiledGEMM = false  // tiled GEMM を無効化して比較
kc.MaxThreads = 4        // 並列度を制限

sess, _ := onnx.NewSessionWithOptions(modelBytes,
    onnx.WithKernelConfig(kc),
)
```

| フィールド | デフォルト | 説明 |
|---|---|---|
| `UseTiledGEMM` | true | Mc=128/Nc=192/Kc=128 tiled GEMM + microKernel4x8 |
| `UseDepthwiseKernel` | true | depthwise 3×3 特化カーネル |
| `Use1x1FastPath` | true | 1×1 Conv で im2col をスキップ |
| `UseConvTransposeGEMM` | true | ConvTranspose の GEMM 化 |
| `UsePoolFastPath` | true | MaxPool 2×2s2 / 3×3s2 特化 |
| `UseFastErf` | true | FastGELU 用の多項式近似 erf |
| `UseParallelConv` | true | 大きな Conv の goroutine 並列化 |
| `MaxThreads` | 0 | 最大 goroutine 数 (0 = `runtime.GOMAXPROCS`) |

## Development

### ビルド・テスト

```bash
go build ./...
go vet ./...
go test -v -timeout 0 ./...
```

### テストデータ生成

```bash
pip install -r requirements.txt
python testdata/generate_test_models.py
```

### HTML ベンチマークレポート

```bash
go test -v ./onnx -run TestGenerateReport
```

### 新しいオペレーターの追加

1. `internal/ops/` の適切なファイルに `opXxx` を実装
2. `internal/ops/opset.go` の `RegisterAll` に登録
3. `internal/ops/all_ops_test.go` の `unitTestedOps` に追加（`TestRegisterAllHasUnitCoverage` が網羅性を強制）
4. 同ファイルに unit test を追加
5. `testdata/generate_test_models.py` に Python ORT 比較テストを追加

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
ONNX Pure Go Interpreter is under [MIT license](LICENSE).<br>
