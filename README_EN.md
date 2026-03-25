# ONNX Pure Go Interpreter

**English** | [日本語](README.md)

[![Go](https://img.shields.io/badge/Go-1.26+-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)<br>
A pure Go ONNX inference package — no cgo, no assembly, no native dependencies.

> [!IMPORTANT]
> This library is a pure Go implementation without SIMD or native library optimizations.<br>
> Inference performance is roughly 10–50x slower than ONNX Runtime (varies by model and hardware).<br>
> Additionally, only a limited number of models have been verified, so many models may not work.
>
> Primary use cases:
> - Inference for small-scale models
> - Cross-compilation environments (no cgo, WASM, etc.)
> - ONNX graph analysis, visualization, and optimization verification
> - Educational and experimental use (understanding ONNX internals and debugging)
>
> Not suitable for medium/large model inference or high-throughput/low-latency production workloads.<br>
> Not intended as a replacement for ONNX Runtime.

## Features

- **Pure Go** — Cross-compile across any `GOOS`/`GOARCH`
- **Minimal dependencies** — Only `google.golang.org/protobuf`
- **Broad operator coverage** — ~80% of ~200 ONNX standard operators implemented (as of 2026/03/25)
- **Graph optimizations** — 11 passes including Conv+BN fusion, GELU fusion, dead node elimination

## Installation

```bash
go get github.com/Kazuhito00/onnx-purego-interpreter
```

**Requirements:** Go 1.26+

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
	modelBytes, err := os.ReadFile("model.onnx")
	if err != nil {
	    panic(err)
	}

	sess, err := onnx.NewSession(modelBytes)
	if err != nil {
	    panic(err)
	}

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

### Providing Inputs

```go
// Positional: input names resolved by graph definition order
outputs, err := sess.Run(inputTensor)
outputs, err := sess.Run(images, sizes)

// Named: explicit input name mapping
outputs, err := sess.RunWithNames(onnx.Input("input_bgr", inputTensor))
outputs, err := sess.RunWithNames(onnx.Inputs("images", images, "sizes", sizes))
outputs, err := sess.RunWithNames(map[string]tensor.Tensor{"input": inputTensor})
```

### Session Options

```go
sess, err := onnx.NewSessionWithOptions(modelBytes,
	onnx.WithProgressLogger(os.Stdout),                    // build/run progress logging
	onnx.WithDisabledOptimizationPasses("fuse_conv_silu"), // disable specific pass
	onnx.WithKernelConfig(kc),                             // kernel configuration
)
```

## API

| Function / Method | Description |
|---|---|
| `onnx.NewSession(onnxBytes)` | Create session with default settings |
| `onnx.NewSessionWithOptions(onnxBytes, opts...)` | Create session with options |
| `Session.Run(tensors...)` | Run inference (positional inputs) |
| `Session.RunWithNames(inputs)` | Run inference (named inputs) |
| `Session.RunDebug(inputs)` | Debug inference (verbose output) |
| `Session.InputNames()` | Get list of graph input names |
| `Session.Warnings()` | Get opset compatibility warnings |
| `onnx.Input(name, tensor)` | Create single-input map |
| `onnx.Inputs(name1, t1, ...)` | Create multi-input map |
| `onnx.DefaultKernelConfig()` | Create default kernel configuration |
| `onnx.OptimizationPassNames()` | Get list of available optimization pass names |
| `tensor.NewDense[T](shape, data)` | Create a typed tensor |

> [!CAUTION]
> `Session.Run` / `RunWithNames` is **not goroutine-safe**.<br>
> The session reuses an internal arena across runs. Create separate sessions for concurrent inference.

### Session Options

| Option | Description |
|---|---|
| `WithProgressLogger(w)` | Enable build/run progress logging |
| `WithObserver(obs)` | Attach build/run event observer |
| `WithNoOptimization()` | Disable all graph optimizations |
| `WithDisabledOptimizationPasses(names...)` | Disable specific passes |
| `WithOnlyOptimizationPasses(names...)` | Enable only specified passes (disable all others) |
| `WithKernelConfig(kc)` | Configure kernel-level optimizations |

## Architecture

7-stage pipeline from `.onnx` bytes to inference output:

```
.onnx bytes
  → Reader / Decoder           protobuf deserialization
  → Frontend IR                protobuf-free model representation
  → Canonical IR               normalized semantic graph (Graph, Node, Initializer)
  → Analysis / Validation      graph integrity + opset compatibility checks
  → Optimization Passes        BN fusion, Conv fusion, GELU fusion, dead node elimination
  → Execution Plan / Lowering  slot-based compiled plan + Arena pre-allocation
  → Runtime / Kernels          pure Go kernel execution
```

## Operator Coverage

~80% of ~200 ONNX standard operators implemented:

- **Arithmetic / Linear Algebra** — Add, Sub, Mul, Div, MatMul, Gemm, ...
- **Activation** — Relu, Sigmoid, Softmax, GELU, ...
- **Convolution / Pooling** — Conv, ConvTranspose, MaxPool, AveragePool, ...
- **Normalization** — BatchNormalization, LayerNormalization, InstanceNormalization, ...
- **Shape / Indexing** — Reshape, Transpose, Gather, Scatter, Slice, ...
- **RNN** — LSTM, GRU, RNN
- **Control Flow** — If, Loop, Scan
- **Object Detection** — NonMaxSuppression, RoiAlign, DeformConv
- **Transformer** — Attention, RotaryEmbedding
- **Quantization** — DequantizeLinear, QuantizeLinear, QLinearConv, ...
- **Signal Processing** — DFT, STFT, MelWeightMatrix

Full list: [Operators.md](Operators.md)

## Graph Optimizations

All passes are enabled by default. Control via session options for debugging or benchmarking.

| Pass Name | Description |
|---|---|
| `materialize_constants` | Convert Constant ops to initializers |
| `eliminate_dropout` | Remove Dropout nodes |
| `eliminate_identity` | Remove Identity nodes |
| `fuse_conv_batchnorm` | Fuse Conv + BatchNorm → Conv |
| `fuse_conv_add_bias` | Fold Conv + Add(const) → bias |
| `fuse_conv_activation` | Fuse Conv + ReLU/Clip/LeakyReLU → FusedConv |
| `fuse_conv_silu` | Fuse Conv + Sigmoid + Mul → Conv(SiLU) |
| `fuse_matmul_add_bias` | Fuse MatMul + Add → FusedMatMul |
| `fuse_mul_add_affine` | Fuse Mul + Add → FusedAffine |
| `fuse_gelu` | Fuse Div→Erf→Add→Mul→Mul → FastGELU |
| `eliminate_dead_nodes` | Remove unused nodes |

## Kernel Optimizations

Control individual kernel optimizations via `KernelConfig`:

```go
kc := onnx.DefaultKernelConfig()
kc.UseTiledGEMM = false  // disable tiled GEMM for comparison
kc.MaxThreads = 4        // limit parallelism

sess, _ := onnx.NewSessionWithOptions(modelBytes,
    onnx.WithKernelConfig(kc),
)
```

| Field | Default | Description |
|---|---|---|
| `UseTiledGEMM` | true | Mc=128/Nc=192/Kc=128 tiled GEMM + microKernel4x8 |
| `UseDepthwiseKernel` | true | Depthwise 3×3 specialized kernel |
| `Use1x1FastPath` | true | Skip im2col for 1×1 Conv |
| `UseConvTransposeGEMM` | true | GEMM-based ConvTranspose |
| `UsePoolFastPath` | true | MaxPool 2×2s2 / 3×3s2 specialization |
| `UseFastErf` | true | Polynomial erf approximation for FastGELU |
| `UseParallelConv` | true | Goroutine parallelism for large Conv |
| `MaxThreads` | 0 | Max goroutine count (0 = `runtime.GOMAXPROCS`) |

## Profiling

The built-in profiler measures per-op and per-node execution time and memory allocation.

```go
prof := onnx.NewProfiler()
prof.Enable()

sess, _ := onnx.NewSessionWithOptions(modelBytes,
    onnx.WithObserver(prof),
)

outputs, _ := sess.Run(input)

// Print per-op summary
fmt.Println(prof.Summary())
```

Example output:

```
Op Profiling Summary (total: 45.2ms)
Op                           Calls   Total(ms)    Avg(ms)   Alloc(KB)
─────────────────────────────────────────────────────────────────────────
Conv                            26      32.1      1.235      1024.0  (71.0%)
MatMul                           2       5.3      2.650       256.0  (11.7%)
Relu                            24       3.1      0.129         0.0  (6.9%)
...
```

### Profiler API

| Method | Description |
|---|---|
| `NewProfiler()` | Create a profiler (initially disabled) |
| `Enable()` / `Disable()` | Turn profiling on / off |
| `Reset()` | Clear all collected data |
| `Summary()` | Return a summary string |
| `Results()` | Get per-op results (sorted by total time, descending) |
| `NodeResults()` | Get per-node results (sorted by total time, descending) |
| `TotalNs()` | Return total profiled time in nanoseconds |

## Development

### Build & Test

```bash
go build ./...
go vet ./...
go test -v -timeout 0 ./...
```

### Generate Test Data

```bash
pip install -r requirements.txt
python testdata/generate_test_models.py
```

### HTML Benchmark Report

```bash
go test -v ./onnx -run TestGenerateReport
```

### Adding a New Operator

1. Implement `opXxx` in the appropriate file under `internal/ops/`
2. Register in `internal/ops/opset.go` → `RegisterAll`
3. Add to `unitTestedOps` in `internal/ops/all_ops_test.go` (enforced by `TestRegisterAllHasUnitCoverage`)
4. Add unit test in the same file
5. Add Python ORT comparison test in `testdata/generate_test_models.py`

# Author
Kazuhito Takahashi (https://x.com/KzhtTkhs)

# License
ONNX Pure Go Interpreter is under [MIT license](LICENSE).
