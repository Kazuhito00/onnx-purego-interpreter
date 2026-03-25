package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Xor ──

func opXor(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	a := inputs[0].(*tensor.Dense[uint8])
	b := inputs[1].(*tensor.Dense[uint8])
	ad, bd := a.Data(), b.Data()
	out := make([]uint8, len(ad))
	for i := range ad {
		if (ad[i] != 0) != (bd[i] != 0) { out[i] = 1 }
	}
	return []tensor.Tensor{tensor.NewDense[uint8](a.Shape().Clone(), out)}, nil
}

// ── BitShift ──

func opBitShift(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	direction := node.GetAttrString("direction", "LEFT")
	switch a := inputs[0].(type) {
	case *tensor.Dense[uint8]:
		b := inputs[1].(*tensor.Dense[uint8])
		ad, bd := a.Data(), b.Data()
		out := make([]uint8, len(ad))
		for i := range ad {
			if direction == "LEFT" { out[i] = ad[i] << bd[i] } else { out[i] = ad[i] >> bd[i] }
		}
		return []tensor.Tensor{tensor.NewDense[uint8](a.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("BitShift: unsupported type %T", inputs[0])
	}
}

// ── BitwiseAnd / BitwiseNot / BitwiseOr / BitwiseXor ──

func opBitwiseAnd(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch a := inputs[0].(type) {
	case *tensor.Dense[uint8]:
		b := inputs[1].(*tensor.Dense[uint8])
		out := make([]uint8, a.Len())
		for i, v := range a.Data() { out[i] = v & b.Data()[i] }
		return []tensor.Tensor{tensor.NewDense[uint8](a.Shape().Clone(), out)}, nil
	case *tensor.Dense[int32]:
		b := inputs[1].(*tensor.Dense[int32])
		out := make([]int32, a.Len())
		for i, v := range a.Data() { out[i] = v & b.Data()[i] }
		return []tensor.Tensor{tensor.NewDense[int32](a.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("BitwiseAnd: unsupported type %T", inputs[0])
	}
}

func opBitwiseNot(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch a := inputs[0].(type) {
	case *tensor.Dense[uint8]:
		out := make([]uint8, a.Len())
		for i, v := range a.Data() { out[i] = ^v }
		return []tensor.Tensor{tensor.NewDense[uint8](a.Shape().Clone(), out)}, nil
	case *tensor.Dense[int32]:
		out := make([]int32, a.Len())
		for i, v := range a.Data() { out[i] = ^v }
		return []tensor.Tensor{tensor.NewDense[int32](a.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("BitwiseNot: unsupported type %T", inputs[0])
	}
}

func opBitwiseOr(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch a := inputs[0].(type) {
	case *tensor.Dense[uint8]:
		b := inputs[1].(*tensor.Dense[uint8])
		out := make([]uint8, a.Len())
		for i, v := range a.Data() { out[i] = v | b.Data()[i] }
		return []tensor.Tensor{tensor.NewDense[uint8](a.Shape().Clone(), out)}, nil
	case *tensor.Dense[int32]:
		b := inputs[1].(*tensor.Dense[int32])
		out := make([]int32, a.Len())
		for i, v := range a.Data() { out[i] = v | b.Data()[i] }
		return []tensor.Tensor{tensor.NewDense[int32](a.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("BitwiseOr: unsupported type %T", inputs[0])
	}
}

func opBitwiseXor(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch a := inputs[0].(type) {
	case *tensor.Dense[uint8]:
		b := inputs[1].(*tensor.Dense[uint8])
		out := make([]uint8, a.Len())
		for i, v := range a.Data() { out[i] = v ^ b.Data()[i] }
		return []tensor.Tensor{tensor.NewDense[uint8](a.Shape().Clone(), out)}, nil
	case *tensor.Dense[int32]:
		b := inputs[1].(*tensor.Dense[int32])
		out := make([]int32, a.Len())
		for i, v := range a.Data() { out[i] = v ^ b.Data()[i] }
		return []tensor.Tensor{tensor.NewDense[int32](a.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("BitwiseXor: unsupported type %T", inputs[0])
	}
}
