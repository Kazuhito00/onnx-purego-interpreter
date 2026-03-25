package onnx

import (
	"io"
	"strings"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/optimize"
)

// SessionOption mutates session construction behavior.
// セッション構築の挙動を変更するオプション関数。
type SessionOption func(*sessionOptions)

type sessionOptions struct {
	observers      []Observer
	disabledPasses map[string]bool
	noOptimize     bool
	onlyPasses     map[string]bool // if non-nil, only these passes run
	kernelConfig   *KernelConfig
}

// WithObserver registers an observer from session construction onward.
// セッション構築時から Observer を登録する。
func WithObserver(observer Observer) SessionOption {
	return func(opts *sessionOptions) {
		if observer == nil {
			return
		}
		opts.observers = append(opts.observers, observer)
	}
}

// WithProgressLogger registers a simple progress logger from session construction onward.
// セッション構築時から簡易プログレスロガーを登録する。
func WithProgressLogger(w io.Writer) SessionOption {
	return func(opts *sessionOptions) {
		if w == nil {
			return
		}
		opts.observers = append(opts.observers, &ProgressLogger{Writer: w})
	}
}

// WithKernelConfig sets the kernel optimization configuration.
// Controls which specialized kernels (tiled GEMM, depthwise, pool fast paths, etc.) are active.
// Default: all optimizations enabled.
// カーネル最適化の設定を行う。特化カーネル (tiled GEMM, depthwise, pool 高速パス等) の有効/無効を制御する。
// デフォルト: 全最適化有効。
//
// Example:
//
//	kc := onnx.DefaultKernelConfig()
//	kc.UseTiledGEMM = false  // disable tiled GEMM, use simple loop
//	onnx.NewSessionWithOptions(modelBytes, onnx.WithKernelConfig(kc))
func WithKernelConfig(config *KernelConfig) SessionOption {
	return func(opts *sessionOptions) {
		opts.kernelConfig = config
	}
}

// WithNoOptimization disables all graph optimization passes.
// The graph is still validated and lowered, but no fusion or elimination is applied.
// 全グラフ最適化パスを無効化する。グラフの検証と lowering は行われるが、融合や除去は適用されない。
func WithNoOptimization() SessionOption {
	return func(opts *sessionOptions) {
		opts.noOptimize = true
	}
}

// WithOnlyOptimizationPasses enables ONLY the specified passes and disables all others.
// Pass names can be found via optimize.PassNames().
// 指定パスのみ有効化し、他は全て無効化する。パス名は OptimizationPassNames() で取得可能。
//
// Example:
//
//	onnx.WithOnlyOptimizationPasses("fuse_conv_batchnorm", "eliminate_dead_nodes")
func WithOnlyOptimizationPasses(names ...string) SessionOption {
	return func(opts *sessionOptions) {
		if opts.onlyPasses == nil {
			opts.onlyPasses = make(map[string]bool)
		}
		for _, name := range names {
			name = strings.TrimSpace(name)
			if name != "" {
				opts.onlyPasses[name] = true
			}
		}
	}
}

// OptimizationPassNames returns the names of all available optimization passes.
// 利用可能な全最適化パスの名前を返す。
func OptimizationPassNames() []string {
	return optimize.PassNames()
}

// WithDisabledOptimizationPasses disables selected graph optimization passes by name.
// 指定した名前のグラフ最適化パスを無効化する。
func WithDisabledOptimizationPasses(names ...string) SessionOption {
	return func(opts *sessionOptions) {
		if opts.disabledPasses == nil {
			opts.disabledPasses = make(map[string]bool)
		}
		for _, name := range names {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			opts.disabledPasses[name] = true
		}
	}
}
