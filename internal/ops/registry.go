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
	UseDepthwiseKernel   bool // depthwise 3x3 specialized kernel
	Use1x1FastPath       bool // 1x1 Conv direct GEMM (skip im2col)
	UseConvTransposeGEMM bool // GEMM-based ConvTranspose (vs naive 7-nested loop)
	UsePoolFastPath      bool // MaxPool 2x2s2 / 3x3s2 specialization
	UseFastErf           bool // polynomial erf approximation in FastGELU
	UseParallelConv      bool // goroutine parallelism for large Conv
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
		MaxThreads:           0, // 0 = use runtime.GOMAXPROCS
	}
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
