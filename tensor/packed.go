package tensor

import "unsafe"

// PackedF32 holds a pre-packed float32 matrix for GEMM-friendly access.
// It implements the Tensor interface so it can be stored in the runtime execution context.
// GEMM に適したアクセスパターン用にパック済みの float32 行列を保持する。
// ランタイム実行コンテキストに格納できるよう Tensor インターフェースを実装する。
type PackedF32 struct {
	originalShape Shape
	K, N          int
	Packed        []float32
	Original      []float32 // keep original for non-packed fallback
}

// NewPackedF32 creates a packed float32 matrix for optimized GEMM access.
// GEMM 最適化用のパック済み float32 行列を作成する。
func NewPackedF32(shape Shape, k, n int, packed, original []float32) *PackedF32 {
	return &PackedF32{originalShape: shape, K: k, N: n, Packed: packed, Original: original}
}

func (p *PackedF32) DType() DType  { return DTypeFloat32 }
func (p *PackedF32) Shape() Shape  { return p.originalShape }
func (p *PackedF32) Len() int      { return p.K * p.N }
func (p *PackedF32) Clone() Tensor { return p } // immutable
func (p *PackedF32) DataPtr() unsafe.Pointer {
	if len(p.Original) == 0 {
		return nil
	}
	return unsafe.Pointer(&p.Original[0])
}
