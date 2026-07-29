package ops

import (
	"runtime"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// KernelConfig controls which kernel optimizations are active.
// All fields default to true (all optimizations enabled).
type KernelConfig struct {
	UseTiledGEMM         bool // microKernel4x8 tiled GEMM (vs simple ikj loop)
	UseDepthwiseKernel   bool // depthwise direct kernel (3x3 specialized + generic KxK)
	Use1x1FastPath       bool // 1x1 Conv direct GEMM (skip im2col)
	UseConvTransposeGEMM bool // GEMM-based ConvTranspose (vs naive 7-nested loop)
	UsePoolFastPath      bool // MaxPool 2x2s1/s2 / 3x3s2 specialization
	UseFastErf           bool // polynomial erf approximation in FastGELU
	UseParallelConv      bool // goroutine parallelism for large Conv
	UseParallelOps       bool // goroutine parallelism for non-Conv ops (pool/matmul/activation/resize/reduce)
	UseReduceFastPath    bool // ReduceMean trailing-axes fast path (GAP/LN shapes)
	MaxThreads           int  // max goroutines for parallel ops (0 = runtime.GOMAXPROCS)
}

// DefaultKernelConfig returns a config with all optimizations enabled.
func DefaultKernelConfig() *KernelConfig {
	return &KernelConfig{
		UseTiledGEMM:         true,
		UseDepthwiseKernel:   true,
		Use1x1FastPath:       true,
		UseConvTransposeGEMM: true,
		UsePoolFastPath:      true,
		UseFastErf:           true,
		UseParallelConv:      true,
		UseParallelOps:       true,
		UseReduceFastPath:    true,
		MaxThreads:           0, // 0 = use runtime.GOMAXPROCS
	}
}

// ParallelOpsWorkers は Conv 以外の演算の並列度を返す(無効時は 1)。
// nil レシーバはデフォルト設定(有効)として扱う。
func (kc *KernelConfig) ParallelOpsWorkers() int {
	if kc != nil && !kc.UseParallelOps {
		return 1
	}
	return kc.Workers()
}

// ReduceFastPathEnabled は ReduceMean 末尾軸 fast path の有効判定。
func (kc *KernelConfig) ReduceFastPathEnabled() bool {
	return kc == nil || kc.UseReduceFastPath
}

// TiledGEMMEnabled は microKernel 系 tiled GEMM の有効判定(nil = 有効)。
// 無効時は単純な ikj ループ(gemmF32Simple)へフォールバックする。
func (kc *KernelConfig) TiledGEMMEnabled() bool {
	return kc == nil || kc.UseTiledGEMM
}

// Workers returns the effective number of worker goroutines.
// If MaxThreads > 0, it is used; otherwise runtime.GOMAXPROCS(0).
func (kc *KernelConfig) Workers() int {
	if kc != nil && kc.MaxThreads > 0 {
		return kc.MaxThreads
	}
	return runtime.GOMAXPROCS(0)
}

// OpFunc is the signature for all operator implementations.
type OpFunc func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error)

// Registry maps operator names to their implementations.
type Registry struct {
	ops map[string]OpFunc
}

// NewRegistry creates an empty op registry.
func NewRegistry() *Registry {
	return &Registry{ops: make(map[string]OpFunc)}
}

// Register adds an operator implementation.
func (r *Registry) Register(name string, fn OpFunc) {
	r.ops[name] = fn
}

// Lookup finds an operator by name. Returns nil if not found.
func (r *Registry) Lookup(name string) OpFunc {
	return r.ops[name]
}
