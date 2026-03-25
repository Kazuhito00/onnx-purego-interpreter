package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// Dropout in inference mode simply passes through the input.
func opDropout(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if len(inputs) == 0 || inputs[0] == nil {
		return nil, fmt.Errorf("Dropout: no input")
	}
	return []tensor.Tensor{inputs[0]}, nil
}

// Clip clamps values to [min, max].
func opClip(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		minVal := float32(-math.MaxFloat32)
		maxVal := float32(math.MaxFloat32)
		if len(inputs) > 1 && inputs[1] != nil {
			if m, ok := inputs[1].(*tensor.Dense[float32]); ok && m.Len() > 0 {
				minVal = m.At(0)
			}
		}
		if len(inputs) > 2 && inputs[2] != nil {
			if m, ok := inputs[2].(*tensor.Dense[float32]); ok && m.Len() > 0 {
				maxVal = m.At(0)
			}
		}
		return []tensor.Tensor{clipDense(x, minVal, maxVal)}, nil
	case *tensor.Dense[float64]:
		minVal := -math.MaxFloat64
		maxVal := math.MaxFloat64
		if len(inputs) > 1 && inputs[1] != nil {
			if m, ok := inputs[1].(*tensor.Dense[float64]); ok && m.Len() > 0 {
				minVal = m.At(0)
			}
		}
		if len(inputs) > 2 && inputs[2] != nil {
			if m, ok := inputs[2].(*tensor.Dense[float64]); ok && m.Len() > 0 {
				maxVal = m.At(0)
			}
		}
		return []tensor.Tensor{clipDense(x, minVal, maxVal)}, nil
	default:
		return nil, fmt.Errorf("Clip: unsupported type %T", inputs[0])
	}
}

func clipDense[T tensor.Numeric](x *tensor.Dense[T], minVal, maxVal T) *tensor.Dense[T] {
	data := make([]T, x.Len())
	for i, v := range x.Data() {
		if v < minVal {
			data[i] = minVal
		} else if v > maxVal {
			data[i] = maxVal
		} else {
			data[i] = v
		}
	}
	return tensor.NewDense[T](x.Shape().Clone(), data)
}
