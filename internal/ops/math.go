package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// binaryOp implements element-wise binary operations with broadcasting.
func binaryOp[T tensor.Numeric](a, b *tensor.Dense[T], fn func(T, T) T) (*tensor.Dense[T], error) {
	outShape, err := tensor.BroadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}

	// Fast path: same shape
	if a.Shape().Equal(b.Shape()) {
		data := make([]T, a.Len())
		ad, bd := a.Data(), b.Data()
		for i := range data {
			data[i] = fn(ad[i], bd[i])
		}
		return tensor.NewDense[T](outShape, data), nil
	}

	// Fast path: one operand is scalar (0-D or single element)
	if b.Len() == 1 {
		bv := b.Data()[0]
		data := make([]T, a.Len())
		for i, v := range a.Data() {
			data[i] = fn(v, bv)
		}
		return tensor.NewDense[T](outShape, data), nil
	}
	if a.Len() == 1 {
		av := a.Data()[0]
		data := make([]T, b.Len())
		for i, v := range b.Data() {
			data[i] = fn(av, v)
		}
		return tensor.NewDense[T](outShape, data), nil
	}

	// Fast path: same ndim, broadcast only on dims of size 1
	// Common pattern: [N,C,H,W] + [1,C,1,1] (per-channel bias)
	as, bs := a.Shape(), b.Shape()
	if as.NDim() == bs.NDim() && as.NDim() == 4 {
		if bs[0] == 1 && bs[2] == 1 && bs[3] == 1 && bs[1] == as[1] {
			// [N,C,H,W] + [1,C,1,1]
			N, C, HW := as[0], as[1], as[2]*as[3]
			data := make([]T, outShape.Size())
			ad, bd := a.Data(), b.Data()
			for n := 0; n < N; n++ {
				for c := 0; c < C; c++ {
					bv := bd[c]
					off := n*C*HW + c*HW
					for i := 0; i < HW; i++ {
						data[off+i] = fn(ad[off+i], bv)
					}
				}
			}
			return tensor.NewDense[T](outShape, data), nil
		}
		if as[0] == 1 && as[2] == 1 && as[3] == 1 && as[1] == bs[1] {
			// [1,C,1,1] + [N,C,H,W]
			N, C, HW := bs[0], bs[1], bs[2]*bs[3]
			data := make([]T, outShape.Size())
			ad, bd := a.Data(), b.Data()
			for n := 0; n < N; n++ {
				for c := 0; c < C; c++ {
					av := ad[c]
					off := n*C*HW + c*HW
					for i := 0; i < HW; i++ {
						data[off+i] = fn(av, bd[off+i])
					}
				}
			}
			return tensor.NewDense[T](outShape, data), nil
		}
	}

	// Fast path: [C,1,1] op [N,C,H,W] or [N,C,H,W] op [C,1,1] (DenseNet BN pattern)
	if as.NDim() == 3 && bs.NDim() == 4 && as[0] == bs[1] && as[1] == 1 && as[2] == 1 {
		N, C, HW := bs[0], bs[1], bs[2]*bs[3]
		data := make([]T, outShape.Size())
		ad, bd := a.Data(), b.Data()
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				av := ad[c]
				off := n*C*HW + c*HW
				for i := 0; i < HW; i++ {
					data[off+i] = fn(av, bd[off+i])
				}
			}
		}
		return tensor.NewDense[T](outShape, data), nil
	}
	if bs.NDim() == 3 && as.NDim() == 4 && bs[0] == as[1] && bs[1] == 1 && bs[2] == 1 {
		N, C, HW := as[0], as[1], as[2]*as[3]
		data := make([]T, outShape.Size())
		ad, bd := a.Data(), b.Data()
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				bv := bd[c]
				off := n*C*HW + c*HW
				for i := 0; i < HW; i++ {
					data[off+i] = fn(ad[off+i], bv)
				}
			}
		}
		return tensor.NewDense[T](outShape, data), nil
	}

	// Fast path: [C] op [N,C,H,W]  E1D per-channel broadcast
	if as.NDim() == 1 && bs.NDim() == 4 && as[0] == bs[1] {
		N, C, HW := bs[0], bs[1], bs[2]*bs[3]
		data := make([]T, outShape.Size())
		ad, bd := a.Data(), b.Data()
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				av := ad[c]
				off := n*C*HW + c*HW
				for i := 0; i < HW; i++ {
					data[off+i] = fn(av, bd[off+i])
				}
			}
		}
		return tensor.NewDense[T](outShape, data), nil
	}
	if bs.NDim() == 1 && as.NDim() == 4 && bs[0] == as[1] {
		N, C, HW := as[0], as[1], as[2]*as[3]
		data := make([]T, outShape.Size())
		ad, bd := a.Data(), b.Data()
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				bv := bd[c]
				off := n*C*HW + c*HW
				for i := 0; i < HW; i++ {
					data[off+i] = fn(ad[off+i], bv)
				}
			}
		}
		return tensor.NewDense[T](outShape, data), nil
	}

	// Fast path: one operand all-ones-prefix broadcast (e.g. [1,1,S,D] op [B,H,S,D])
	// Uses modulo tiling  Eonly correct when trailing dims are contiguous match
	if as.NDim() == bs.NDim() && as.NDim() >= 2 && canTileBroadcast(as, bs) && as.Size() > 0 {
		return tileBroadcastOp(a, b, outShape, fn, false)
	}
	if as.NDim() == bs.NDim() && bs.NDim() >= 2 && canTileBroadcast(bs, as) && bs.Size() > 0 {
		return tileBroadcastOp(b, a, outShape, fn, true)
	}

	// General broadcast path — coordinate-based with inner loop optimization
	size := outShape.Size()
	data := make([]T, size)
	ad, bd := a.Data(), b.Data()
	ndim := outShape.NDim()

	if ndim >= 2 {
		// Precompute broadcast strides: for each dim, if source dim==1, stride=0 (broadcast)
		aBS := make([]int, ndim)
		bBS := make([]int, ndim)
		aShape := padShapeLeft(as, ndim)
		bShape := padShapeLeft(bs, ndim)
		aStr := tensor.Strides(aShape)
		bStr := tensor.Strides(bShape)
		for d := 0; d < ndim; d++ {
			if aShape[d] == 1 { aBS[d] = 0 } else { aBS[d] = aStr[d] }
			if bShape[d] == 1 { bBS[d] = 0 } else { bBS[d] = bStr[d] }
		}
		innerN := outShape[ndim-1]
		outerN := size / innerN
		coords := make([]int, ndim)
		for outer := 0; outer < outerN; outer++ {
			// Decode outer index into coords[0..ndim-2]
			rem := outer
			for d := ndim - 2; d >= 0; d-- {
				coords[d] = rem % outShape[d]
				rem /= outShape[d]
			}
			// Compute base offsets for A and B
			aBase, bBase := 0, 0
			for d := 0; d < ndim-1; d++ {
				aBase += coords[d] * aBS[d]
				bBase += coords[d] * bBS[d]
			}
			oBase := outer * innerN
			aInner, bInner := aBS[ndim-1], bBS[ndim-1]
			if aInner != 0 && bInner != 0 {
				// Both vary along inner dim
				for j := 0; j < innerN; j++ {
					data[oBase+j] = fn(ad[aBase+j*aInner], bd[bBase+j*bInner])
				}
			} else if aInner != 0 {
				// B is broadcast along inner dim
				bv := bd[bBase]
				for j := 0; j < innerN; j++ {
					data[oBase+j] = fn(ad[aBase+j*aInner], bv)
				}
			} else if bInner != 0 {
				// A is broadcast along inner dim
				av := ad[aBase]
				for j := 0; j < innerN; j++ {
					data[oBase+j] = fn(av, bd[bBase+j*bInner])
				}
			} else {
				// Both broadcast along inner dim
				v := fn(ad[aBase], bd[bBase])
				for j := 0; j < innerN; j++ {
					data[oBase+j] = v
				}
			}
		}
		return tensor.NewDense[T](outShape, data), nil
	}

	// 1D fallback
	outStrides := tensor.Strides(outShape)
	aStrides := tensor.Strides(a.Shape())
	bStrides := tensor.Strides(b.Shape())
	for i := 0; i < size; i++ {
		ai := tensor.BroadcastIndex(i, outShape, a.Shape(), outStrides, aStrides)
		bi := tensor.BroadcastIndex(i, outShape, b.Shape(), outStrides, bStrides)
		data[i] = fn(ad[ai], bd[bi])
	}
	return tensor.NewDense[T](outShape, data), nil
}

// padShapeLeft pads a shape with 1s on the left to match target ndim.
func padShapeLeft(s tensor.Shape, ndim int) tensor.Shape {
	if len(s) >= ndim {
		return s
	}
	out := make(tensor.Shape, ndim)
	off := ndim - len(s)
	for i := 0; i < off; i++ {
		out[i] = 1
	}
	copy(out[off:], s)
	return out
}

func dispatchBinaryOp(inputs []tensor.Tensor, fn32 func(float32, float32) float32, fn64 func(float64, float64) float64, fnI32 func(int32, int32) int32, fnI64 func(int64, int64) int64) (tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	switch at := a.(type) {
	case *tensor.Dense[float32]:
		bt, ok := b.(*tensor.Dense[float32])
		if !ok {
			return nil, fmt.Errorf("ops: type mismatch: %T vs %T", a, b)
		}
		return binaryOp(at, bt, fn32)
	case *tensor.Dense[float64]:
		bt, ok := b.(*tensor.Dense[float64])
		if !ok {
			return nil, fmt.Errorf("ops: type mismatch: %T vs %T", a, b)
		}
		return binaryOp(at, bt, fn64)
	case *tensor.Dense[int32]:
		bt, ok := b.(*tensor.Dense[int32])
		if !ok {
			return nil, fmt.Errorf("ops: type mismatch: %T vs %T", a, b)
		}
		return binaryOp(at, bt, fnI32)
	case *tensor.Dense[int64]:
		bt, ok := b.(*tensor.Dense[int64])
		if !ok {
			return nil, fmt.Errorf("ops: type mismatch: %T vs %T", a, b)
		}
		return binaryOp(at, bt, fnI64)
	default:
		return nil, fmt.Errorf("ops: unsupported tensor type %T for binary op", a)
	}
}

func opAdd(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 { return a + b },
		func(a, b float64) float64 { return a + b },
		func(a, b int32) int32 { return a + b },
		func(a, b int64) int64 { return a + b },
	)
	if err != nil {
		return nil, fmt.Errorf("Add: %w", err)
	}
	return []tensor.Tensor{out}, nil
}

func opSub(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 { return a - b },
		func(a, b float64) float64 { return a - b },
		func(a, b int32) int32 { return a - b },
		func(a, b int64) int64 { return a - b },
	)
	if err != nil {
		return nil, fmt.Errorf("Sub: %w", err)
	}
	return []tensor.Tensor{out}, nil
}

func opMul(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 { return a * b },
		func(a, b float64) float64 { return a * b },
		func(a, b int32) int32 { return a * b },
		func(a, b int64) int64 { return a * b },
	)
	if err != nil {
		return nil, fmt.Errorf("Mul: %w", err)
	}
	return []tensor.Tensor{out}, nil
}

func opDiv(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 { return a / b },
		func(a, b float64) float64 { return a / b },
		func(a, b int32) int32 { return a / b },
		func(a, b int64) int64 { return a / b },
	)
	if err != nil {
		return nil, fmt.Errorf("Div: %w", err)
	}
	return []tensor.Tensor{out}, nil
}

// canTileBroadcast checks if 'small' can be broadcast to 'big' by simple tiling.
// small must have same ndim, leading dims all 1, trailing dims matching big.
func canTileBroadcast(small, big tensor.Shape) bool {
	if small.NDim() != big.NDim() || small.NDim() == 0 {
		return false
	}
	if small.Equal(big) {
		return false
	}
	ndim := small.NDim()
	for d := 0; d < ndim; d++ {
		if small[d] == 1 && big[d] > 1 {
			continue
		}
		if small[d] == big[d] {
			for d2 := d; d2 < ndim; d2++ {
				if small[d2] != big[d2] {
					return false
				}
			}
			return true
		}
		return false
	}
	return false
}

// tileBroadcastOp performs element-wise op where 'small' tiles into 'big'.
func tileBroadcastOp[T tensor.Numeric](small, big *tensor.Dense[T], outShape tensor.Shape, fn func(T, T) T, swapped bool) (*tensor.Dense[T], error) {
	sd, bd := small.Data(), big.Data()
	smallLen := small.Len()
	bigLen := big.Len()
	data := make([]T, outShape.Size())
	if smallLen == 0 || bigLen == 0 {
		return tensor.NewDense[T](outShape, data), nil
	}
	if swapped {
		for i := 0; i < bigLen; i++ {
			data[i] = fn(bd[i], sd[i%smallLen])
		}
	} else {
		for i := 0; i < bigLen; i++ {
			data[i] = fn(sd[i%smallLen], bd[i])
		}
	}
	return tensor.NewDense[T](outShape, data), nil
}

func opMod(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	fmodAttr := node.GetAttrInt("fmod", 0)
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 {
			if fmodAttr != 0 {
				return float32(math.Mod(float64(a), float64(b)))
			}
			return a - b*float32(math.Floor(float64(a)/float64(b)))
		},
		func(a, b float64) float64 {
			if fmodAttr != 0 {
				return math.Mod(a, b)
			}
			return a - b*math.Floor(a/b)
		},
		func(a, b int32) int32 {
			return a % b
		},
		func(a, b int64) int64 {
			return a % b
		},
	)
	if err != nil {
		return nil, fmt.Errorf("Mod: %w", err)
	}
	return []tensor.Tensor{out}, nil
}



