package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// If is handled by the engine runtime because it needs access to outer scope values.
func opIf(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return nil, fmt.Errorf("If: handled by engine runtime")
}

// Not - element-wise logical NOT
func opNot(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[uint8]:
		data := make([]uint8, t.Len())
		for i, v := range t.Data() {
			if v == 0 {
				data[i] = 1
			}
		}
		return []tensor.Tensor{tensor.NewDense[uint8](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Not: unsupported type %T", inputs[0])
	}
}

// Size - returns the total number of elements as int64 scalar
func opSize(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	n := int64(inputs[0].Len())
	return []tensor.Tensor{tensor.NewDenseScalar(n)}, nil
}

// ConstantOfShape - creates a tensor of the given shape filled with a constant value
func opConstantOfShape(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	shapeTensor := inputs[0].(*tensor.Dense[int64])
	shape := make(tensor.Shape, shapeTensor.Len())
	for i, v := range shapeTensor.Data() {
		shape[i] = int(v)
	}

	if t := node.GetAttrTensor("value"); t != nil {
		switch t.DType {
		case ir.DataTypeFloat:
			data, _ := materialize.Float32(t)
			val := float32(0)
			if len(data) > 0 {
				val = data[0]
			}
			out := make([]float32, shape.Size())
			if val != 0 {
				for i := range out {
					out[i] = val
				}
			}
			return []tensor.Tensor{tensor.NewDense[float32](shape, out)}, nil
		case ir.DataTypeInt64:
			data, _ := materialize.Int64(t)
			val := int64(0)
			if len(data) > 0 {
				val = data[0]
			}
			out := make([]int64, shape.Size())
			if val != 0 {
				for i := range out { out[i] = val }
			}
			return []tensor.Tensor{tensor.NewDense[int64](shape, out)}, nil
		case ir.DataTypeInt32:
			raw := t.RawData
			val := int32(0)
			if len(raw) >= 4 {
				val = int32(raw[0]) | int32(raw[1])<<8 | int32(raw[2])<<16 | int32(raw[3])<<24
			}
			out := make([]int32, shape.Size())
			if val != 0 {
				for i := range out { out[i] = val }
			}
			return []tensor.Tensor{tensor.NewDense[int32](shape, out)}, nil
		case ir.DataTypeUint8, ir.DataTypeBool:
			val := uint8(0)
			if len(t.RawData) > 0 {
				val = t.RawData[0]
			}
			out := make([]uint8, shape.Size())
			if val != 0 {
				for i := range out { out[i] = val }
			}
			return []tensor.Tensor{tensor.NewDense[uint8](shape, out)}, nil
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](shape, make([]float32, shape.Size()))}, nil
}

func opRange(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch start := inputs[0].(type) {
	case *tensor.Dense[int64]:
		limit := inputs[1].(*tensor.Dense[int64]).At(0)
		delta := inputs[2].(*tensor.Dense[int64]).At(0)
		var data []int64
		for v := start.At(0); (delta > 0 && v < limit) || (delta < 0 && v > limit); v += delta {
			data = append(data, v)
		}
		return []tensor.Tensor{tensor.NewDense[int64](tensor.Shape{len(data)}, data)}, nil
	case *tensor.Dense[float32]:
		limit := inputs[1].(*tensor.Dense[float32]).At(0)
		delta := inputs[2].(*tensor.Dense[float32]).At(0)
		var data []float32
		for v := start.At(0); (delta > 0 && v < limit) || (delta < 0 && v > limit); v += delta {
			data = append(data, v)
		}
		return []tensor.Tensor{tensor.NewDense[float32](tensor.Shape{len(data)}, data)}, nil
	default:
		return nil, fmt.Errorf("Range: unsupported type %T", inputs[0])
	}
}


func opCumSum(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := 0
	switch axisT := inputs[1].(type) {
	case *tensor.Dense[int64]:
		if axisT.Len() == 0 {
			return nil, fmt.Errorf("CumSum: axis must be scalar")
		}
		axis = int(axisT.At(0))
	case *tensor.Dense[int32]:
		if axisT.Len() == 0 {
			return nil, fmt.Errorf("CumSum: axis must be scalar")
		}
		axis = int(axisT.At(0))
	default:
		return nil, fmt.Errorf("CumSum: axis must be int32/int64 scalar, got %T", inputs[1])
	}
	exclusive := node.GetAttrInt("exclusive", 0) != 0
	reverse := node.GetAttrInt("reverse", 0) != 0
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{cumSumDense(x, axis, exclusive, reverse)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{cumSumDense(x, axis, exclusive, reverse)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{cumSumDense(x, axis, exclusive, reverse)}, nil
	default:
		return nil, fmt.Errorf("CumSum: unsupported type %T", inputs[0])
	}
}

func cumSumDense[T tensor.Numeric](x *tensor.Dense[T], axis int, exclusive, reverse bool) *tensor.Dense[T] {
	shape := x.Shape()
	ndim := shape.NDim()
	if axis < 0 {
		axis += ndim
	}
	out := make([]T, x.Len())
	copy(out, x.Data())
	axisSize := shape[axis]
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}
	inner := 1
	for i := axis + 1; i < ndim; i++ {
		inner *= shape[i]
	}
	for o := 0; o < outer; o++ {
		for inr := 0; inr < inner; inr++ {
			running := T(0)
			if reverse {
				for a := axisSize - 1; a >= 0; a-- {
					idx := (o*axisSize+a)*inner + inr
					v := out[idx]
					if exclusive {
						out[idx] = running
						running += v
					} else {
						running += v
						out[idx] = running
					}
				}
			} else {
				for a := 0; a < axisSize; a++ {
					idx := (o*axisSize+a)*inner + inr
					v := out[idx]
					if exclusive {
						out[idx] = running
						running += v
					} else {
						running += v
						out[idx] = running
					}
				}
			}
		}
	}
	return tensor.NewDense[T](shape.Clone(), out)
}
