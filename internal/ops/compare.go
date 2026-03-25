package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func opGreater(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch a := inputs[0].(type) {
	case *tensor.Dense[float32]:
		b, ok := inputs[1].(*tensor.Dense[float32])
		if !ok {
			return nil, fmt.Errorf("Greater: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y float32) bool { return x > y })
	case *tensor.Dense[int32]:
		b, ok := inputs[1].(*tensor.Dense[int32])
		if !ok {
			return nil, fmt.Errorf("Greater: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y int32) bool { return x > y })
	case *tensor.Dense[int64]:
		b, ok := inputs[1].(*tensor.Dense[int64])
		if !ok {
			return nil, fmt.Errorf("Greater: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y int64) bool { return x > y })
	case *tensor.Dense[float64]:
		b, ok := inputs[1].(*tensor.Dense[float64])
		if !ok {
			return nil, fmt.Errorf("Greater: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y float64) bool { return x > y })
	case *tensor.Dense[uint8]:
		b, ok := inputs[1].(*tensor.Dense[uint8])
		if !ok {
			return nil, fmt.Errorf("Greater: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y uint8) bool { return x > y })
	default:
		return nil, fmt.Errorf("Greater: unsupported type %T", inputs[0])
	}
}

func opLess(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch a := inputs[0].(type) {
	case *tensor.Dense[float32]:
		b, ok := inputs[1].(*tensor.Dense[float32])
		if !ok {
			return nil, fmt.Errorf("Less: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y float32) bool { return x < y })
	case *tensor.Dense[int32]:
		b, ok := inputs[1].(*tensor.Dense[int32])
		if !ok {
			return nil, fmt.Errorf("Less: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y int32) bool { return x < y })
	case *tensor.Dense[int64]:
		b, ok := inputs[1].(*tensor.Dense[int64])
		if !ok {
			return nil, fmt.Errorf("Less: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y int64) bool { return x < y })
	case *tensor.Dense[float64]:
		b, ok := inputs[1].(*tensor.Dense[float64])
		if !ok {
			return nil, fmt.Errorf("Less: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y float64) bool { return x < y })
	case *tensor.Dense[uint8]:
		b, ok := inputs[1].(*tensor.Dense[uint8])
		if !ok {
			return nil, fmt.Errorf("Less: type mismatch %T vs %T", inputs[0], inputs[1])
		}
		return compareDense(a, b, func(x, y uint8) bool { return x < y })
	default:
		return nil, fmt.Errorf("Less: unsupported type %T", inputs[0])
	}
}

func compareDense[T tensor.Numeric](a, b *tensor.Dense[T], pred func(T, T) bool) ([]tensor.Tensor, error) {
	outShape, err := tensor.BroadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	size := outShape.Size()
	if size == 0 {
		size = 1
	}
	out := make([]uint8, size)
	outStrides := tensor.Strides(outShape)
	aStrides := tensor.Strides(a.Shape())
	bStrides := tensor.Strides(b.Shape())
	ad, bd := a.Data(), b.Data()
	for i := 0; i < len(out); i++ {
		ai := tensor.BroadcastIndex(i, outShape, a.Shape(), outStrides, aStrides)
		bi := tensor.BroadcastIndex(i, outShape, b.Shape(), outStrides, bStrides)
		if pred(ad[ai], bd[bi]) {
			out[i] = 1
		}
	}
	return []tensor.Tensor{tensor.NewDense[uint8](outShape, out)}, nil
}

func opAnd(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	a, ok := inputs[0].(*tensor.Dense[uint8])
	if !ok {
		return nil, fmt.Errorf("And: unsupported type %T", inputs[0])
	}
	b, ok := inputs[1].(*tensor.Dense[uint8])
	if !ok {
		return nil, fmt.Errorf("And: unsupported type %T", inputs[1])
	}
	outShape, err := tensor.BroadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	size := outShape.Size()
	if size == 0 {
		size = 1
	}
	out := make([]uint8, size)
	outStrides := tensor.Strides(outShape)
	aStrides := tensor.Strides(a.Shape())
	bStrides := tensor.Strides(b.Shape())
	ad, bd := a.Data(), b.Data()
	for i := 0; i < len(out); i++ {
		ai := tensor.BroadcastIndex(i, outShape, a.Shape(), outStrides, aStrides)
		bi := tensor.BroadcastIndex(i, outShape, b.Shape(), outStrides, bStrides)
		if ad[ai] != 0 && bd[bi] != 0 {
			out[i] = 1
		}
	}
	return []tensor.Tensor{tensor.NewDense[uint8](outShape, out)}, nil
}

func opMin(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 { if a < b { return a }; return b },
		func(a, b float64) float64 { if a < b { return a }; return b },
		func(a, b int32) int32 { if a < b { return a }; return b },
		func(a, b int64) int64 { if a < b { return a }; return b },
	)
	if err != nil {
		return nil, fmt.Errorf("Min: %w", err)
	}
	return []tensor.Tensor{out}, nil
}

func opMax(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 { if a > b { return a }; return b },
		func(a, b float64) float64 { if a > b { return a }; return b },
		func(a, b int32) int32 { if a > b { return a }; return b },
		func(a, b int64) int64 { if a > b { return a }; return b },
	)
	if err != nil {
		return nil, fmt.Errorf("Max: %w", err)
	}
	return []tensor.Tensor{out}, nil
}
