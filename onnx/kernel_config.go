package onnx

import "github.com/Kazuhito00/onnx-purego-interpreter/internal/ops"

// KernelConfig controls which kernel optimizations are active.
// All fields default to true (all optimizations enabled) when created via DefaultKernelConfig.
// カーネル最適化の有効/無効を制御する。
// DefaultKernelConfig で作成した場合、全フィールドが true (全最適化有効)。
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
	UseWinograd          bool // Winograd F(2x2,3x3) for 3x3 stride-1 dense conv
	MaxThreads           int  // max goroutines for parallel ops (0 = runtime.GOMAXPROCS)
}

// DefaultKernelConfig returns a config with all optimizations enabled.
// 全最適化が有効な KernelConfig を返す。
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
		UseWinograd:          true,
		MaxThreads:           0,
	}
}

// toInternal converts to the internal ops.KernelConfig.
func (kc *KernelConfig) toInternal() *ops.KernelConfig {
	if kc == nil {
		return nil
	}
	return &ops.KernelConfig{
		UseTiledGEMM:         kc.UseTiledGEMM,
		UseDepthwiseKernel:   kc.UseDepthwiseKernel,
		Use1x1FastPath:       kc.Use1x1FastPath,
		UseConvTransposeGEMM: kc.UseConvTransposeGEMM,
		UsePoolFastPath:      kc.UsePoolFastPath,
		UseFastErf:           kc.UseFastErf,
		UseParallelConv:      kc.UseParallelConv,
		UseParallelOps:       kc.UseParallelOps,
		UseReduceFastPath:    kc.UseReduceFastPath,
		UseWinograd:          kc.UseWinograd,
		MaxThreads:           kc.MaxThreads,
	}
}
