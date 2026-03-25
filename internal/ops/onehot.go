package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func opOneHot(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if len(inputs) < 3 {
		return nil, fmt.Errorf("OneHot: expected 3 inputs")
	}
	axis := int(node.GetAttrInt("axis", -1))

	depth, err := scalarToInt(inputs[1])
	if err != nil {
		return nil, fmt.Errorf("OneHot depth: %w", err)
	}
	if depth <= 0 {
		return nil, fmt.Errorf("OneHot: depth must be > 0")
	}

	switch values := inputs[2].(type) {
	case *tensor.Dense[float32]:
		if values.Len() != 2 {
			return nil, fmt.Errorf("OneHot: values must have len 2")
		}
		return oneHotTyped[float32](inputs[0], depth, axis, values.Data()[0], values.Data()[1])
	case *tensor.Dense[int64]:
		if values.Len() != 2 {
			return nil, fmt.Errorf("OneHot: values must have len 2")
		}
		return oneHotTyped[int64](inputs[0], depth, axis, values.Data()[0], values.Data()[1])
	case *tensor.Dense[int32]:
		if values.Len() != 2 {
			return nil, fmt.Errorf("OneHot: values must have len 2")
		}
		return oneHotTyped[int32](inputs[0], depth, axis, values.Data()[0], values.Data()[1])
	default:
		return nil, fmt.Errorf("OneHot: unsupported values type %T", inputs[2])
	}
}

func scalarToInt(t tensor.Tensor) (int, error) {
	switch dt := t.(type) {
	case *tensor.Dense[int64]:
		if dt.Len() == 0 {
			return 0, fmt.Errorf("empty scalar")
		}
		return int(dt.At(0)), nil
	case *tensor.Dense[int32]:
		if dt.Len() == 0 {
			return 0, fmt.Errorf("empty scalar")
		}
		return int(dt.At(0)), nil
	default:
		return 0, fmt.Errorf("unsupported scalar type %T", t)
	}
}

func indicesToInts(t tensor.Tensor) ([]int, tensor.Shape, error) {
	switch dt := t.(type) {
	case *tensor.Dense[int64]:
		out := make([]int, dt.Len())
		for i, v := range dt.Data() {
			out[i] = int(v)
		}
		return out, dt.Shape().Clone(), nil
	case *tensor.Dense[int32]:
		out := make([]int, dt.Len())
		for i, v := range dt.Data() {
			out[i] = int(v)
		}
		return out, dt.Shape().Clone(), nil
	case *tensor.Dense[float32]:
		out := make([]int, dt.Len())
		for i, v := range dt.Data() {
			out[i] = int(v)
		}
		return out, dt.Shape().Clone(), nil
	default:
		return nil, nil, fmt.Errorf("unsupported indices type %T", t)
	}
}

func oneHotTyped[T tensor.Numeric](indicesT tensor.Tensor, depth, axis int, offValue, onValue T) ([]tensor.Tensor, error) {
	indices, idxShape, err := indicesToInts(indicesT)
	if err != nil {
		return nil, err
	}
	rank := idxShape.NDim()
	if axis < 0 {
		axis += rank + 1
	}
	if axis < 0 || axis > rank {
		return nil, fmt.Errorf("OneHot: axis %d out of range for rank %d", axis, rank)
	}

	outShape := make(tensor.Shape, 0, rank+1)
	outShape = append(outShape, idxShape[:axis]...)
	outShape = append(outShape, depth)
	outShape = append(outShape, idxShape[axis:]...)

	outData := make([]T, outShape.Size())
	if offValue != 0 {
		for i := range outData {
			outData[i] = offValue
		}
	}

	idxStrides := tensor.Strides(idxShape)
	outStrides := tensor.Strides(outShape)
	for flat, cls := range indices {
		if cls < 0 {
			cls += depth
		}
		if cls < 0 || cls >= depth {
			continue
		}
		outIdx := 0
		rem := flat
		for d := 0; d < rank; d++ {
			coord := rem / idxStrides[d]
			rem %= idxStrides[d]
			if d < axis {
				outIdx += coord * outStrides[d]
			} else {
				outIdx += coord * outStrides[d+1]
			}
		}
		outIdx += cls * outStrides[axis]
		outData[outIdx] = onValue
	}

	return []tensor.Tensor{tensor.NewDense[T](outShape, outData)}, nil
}
