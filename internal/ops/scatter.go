package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func opScatterND(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if len(inputs) < 3 {
		return nil, fmt.Errorf("ScatterND: expected 3 inputs")
	}
	reduction := node.GetAttrString("reduction", "none")
	switch data := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return scatterND(data, inputs[1], inputs[2], reduction)
	case *tensor.Dense[float64]:
		return scatterND(data, inputs[1], inputs[2], reduction)
	case *tensor.Dense[int64]:
		return scatterND(data, inputs[1], inputs[2], reduction)
	case *tensor.Dense[int32]:
		return scatterND(data, inputs[1], inputs[2], reduction)
	case *tensor.Dense[uint8]:
		return scatterND(data, inputs[1], inputs[2], reduction)
	case *tensor.Dense[int8]:
		return scatterND(data, inputs[1], inputs[2], reduction)
	default:
		return nil, fmt.Errorf("ScatterND: unsupported data type %T", inputs[0])
	}
}

func scatterND[T tensor.Numeric](data *tensor.Dense[T], indicesT tensor.Tensor, updatesT tensor.Tensor, reduction string) ([]tensor.Tensor, error) {
	var indices []int64
	switch it := indicesT.(type) {
	case *tensor.Dense[int64]:
		indices = it.Data()
	case *tensor.Dense[int32]:
		indices = make([]int64, len(it.Data()))
		for i, v := range it.Data() {
			indices[i] = int64(v)
		}
	default:
		return nil, fmt.Errorf("ScatterND: unsupported indices type %T", indicesT)
	}
	updates, ok := updatesT.(*tensor.Dense[T])
	if !ok {
		return nil, fmt.Errorf("ScatterND: update type mismatch %T vs %T", data, updatesT)
	}

	dataShape := data.Shape()
	indicesShape := indicesT.Shape()
	if len(indicesShape) == 0 {
		return nil, fmt.Errorf("ScatterND: indices must have rank >= 1")
	}
	k := indicesShape[len(indicesShape)-1]
	if k < 1 || k > len(dataShape) {
		return nil, fmt.Errorf("ScatterND: invalid index depth %d for data rank %d", k, len(dataShape))
	}

	prefixCount := 1
	for _, d := range indicesShape[:len(indicesShape)-1] {
		prefixCount *= d
	}
	sliceSize := 1
	for _, d := range dataShape[k:] {
		sliceSize *= d
	}
	if updates.Len() != prefixCount*sliceSize {
		return nil, fmt.Errorf("ScatterND: updates length mismatch: got %d want %d", updates.Len(), prefixCount*sliceSize)
	}

	dataStrides := tensor.Strides(dataShape)
	out := make([]T, len(data.Data()))
	copy(out, data.Data())
	updatesData := updates.Data()

	for p := 0; p < prefixCount; p++ {
		base := 0
		for j := 0; j < k; j++ {
			idx := int(indices[p*k+j])
			dim := dataShape[j]
			if idx < 0 {
				idx += dim
			}
			if idx < 0 || idx >= dim {
				return nil, fmt.Errorf("ScatterND: index %d out of range for axis %d with dim %d", idx, j, dim)
			}
			base += idx * dataStrides[j]
		}
		uOff := p * sliceSize
		switch reduction {
		case "add":
			for i := 0; i < sliceSize; i++ { out[base+i] += updatesData[uOff+i] }
		case "mul":
			for i := 0; i < sliceSize; i++ { out[base+i] *= updatesData[uOff+i] }
		case "min":
			for i := 0; i < sliceSize; i++ { if updatesData[uOff+i] < out[base+i] { out[base+i] = updatesData[uOff+i] } }
		case "max":
			for i := 0; i < sliceSize; i++ { if updatesData[uOff+i] > out[base+i] { out[base+i] = updatesData[uOff+i] } }
		default: // "none"
			copy(out[base:base+sliceSize], updatesData[uOff:uOff+sliceSize])
		}
	}

	return []tensor.Tensor{tensor.NewDense[T](dataShape, out)}, nil
}
