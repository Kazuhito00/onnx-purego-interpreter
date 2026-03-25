package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func opConstant(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Constant op stores value in attributes
	if t := node.GetAttrTensor("value"); t != nil {
		out, err := materialize.Tensor(t)
		if err != nil {
			return nil, fmt.Errorf("Constant: %w", err)
		}
		return []tensor.Tensor{out}, nil
	}

	// value_float
	if v, ok := node.Attrs["value_float"]; ok {
		if af, ok := v.(ir.AttrFloat); ok {
			return []tensor.Tensor{tensor.NewDenseScalar(af.Value)}, nil
		}
	}

	// value_int
	if v, ok := node.Attrs["value_int"]; ok {
		if ai, ok := v.(ir.AttrInt); ok {
			return []tensor.Tensor{tensor.NewDenseScalar(ai.Value)}, nil
		}
	}

	// value_floats
	if v, ok := node.Attrs["value_floats"]; ok {
		if af, ok := v.(ir.AttrFloats); ok {
			data := make([]float32, len(af.Value))
			copy(data, af.Value)
			return []tensor.Tensor{tensor.NewDense[float32](tensor.Shape{len(data)}, data)}, nil
		}
	}

	// value_ints
	if v, ok := node.Attrs["value_ints"]; ok {
		if ai, ok := v.(ir.AttrInts); ok {
			data := make([]int64, len(ai.Value))
			copy(data, ai.Value)
			return []tensor.Tensor{tensor.NewDense[int64](tensor.Shape{len(data)}, data)}, nil
		}
	}

	return nil, fmt.Errorf("Constant: no value attribute found")
}

func opIdentity(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if len(inputs) == 0 || inputs[0] == nil {
		return nil, fmt.Errorf("Identity: no input")
	}
	return []tensor.Tensor{inputs[0].Clone()}, nil
}

func opGather(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", 0))

	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return gatherTyped(dt, inputs[1], axis)
	case *tensor.Dense[float64]:
		return gatherTyped(dt, inputs[1], axis)
	case *tensor.Dense[int32]:
		return gatherTyped(dt, inputs[1], axis)
	case *tensor.Dense[int64]:
		return gatherTyped(dt, inputs[1], axis)
	case *tensor.Dense[uint8]:
		return gatherTyped(dt, inputs[1], axis)
	case *tensor.Dense[int8]:
		return gatherTyped(dt, inputs[1], axis)
	default:
		return nil, fmt.Errorf("Gather: unsupported data type %T", inputs[0])
	}
}

func gatherTyped[T tensor.Numeric](data *tensor.Dense[T], indices tensor.Tensor, axis int) ([]tensor.Tensor, error) {
	dataShape := data.Shape()
	ndim := dataShape.NDim()

	if axis < 0 {
		axis += ndim
	}

	// Get indices as int64
	idxValues, idxShape, err := getIndices(indices)
	if err != nil {
		return nil, fmt.Errorf("Gather: %w", err)
	}

	// Handle negative indices
	axisSize := dataShape[axis]
	for i, v := range idxValues {
		if v < 0 {
			idxValues[i] = v + int64(axisSize)
		}
	}

	// Output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
	var outShape tensor.Shape
	outShape = append(outShape, dataShape[:axis]...)
	outShape = append(outShape, idxShape...)
	outShape = append(outShape, dataShape[axis+1:]...)

	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}

	outSize := outShape.Size()
	outData := make([]T, outSize)
	srcData := data.Data()

	// Compute strides for iteration
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= dataShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < ndim; i++ {
		innerSize *= dataShape[i]
	}

	numIndices := len(idxValues)
	if numIndices == 0 {
		numIndices = 1
	}

	outIdx := 0
	for outer := 0; outer < outerSize; outer++ {
		for _, idx := range idxValues {
			srcBase := outer*axisSize*innerSize + int(idx)*innerSize
			copy(outData[outIdx:outIdx+innerSize], srcData[srcBase:srcBase+innerSize])
			outIdx += innerSize
		}
	}

	return []tensor.Tensor{tensor.NewDense[T](outShape, outData)}, nil
}

func getIndices(t tensor.Tensor) ([]int64, tensor.Shape, error) {
	switch dt := t.(type) {
	case *tensor.Dense[int64]:
		return dt.Data(), dt.Shape(), nil
	case *tensor.Dense[int32]:
		data := make([]int64, dt.Len())
		for i, v := range dt.Data() {
			data[i] = int64(v)
		}
		return data, dt.Shape(), nil
	default:
		return nil, nil, fmt.Errorf("indices must be int32 or int64, got %T", t)
	}
}
