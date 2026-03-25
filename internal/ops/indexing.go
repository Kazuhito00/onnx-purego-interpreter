package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func opGatherND(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	data := inputs[0]
	indices, ok := inputs[1].(*tensor.Dense[int64])
	if !ok {
		return nil, fmt.Errorf("GatherND: indices must be int64, got %T", inputs[1])
	}
	batchDims := int(node.GetAttrInt("batch_dims", 0))
	if batchDims != 0 {
		return nil, fmt.Errorf("GatherND: batch_dims=%d unsupported", batchDims)
	}

	switch x := data.(type) {
	case *tensor.Dense[float32]:
		return gatherNDDense(x, indices)
	case *tensor.Dense[int32]:
		return gatherNDDense(x, indices)
	case *tensor.Dense[int64]:
		return gatherNDDense(x, indices)
	case *tensor.Dense[float64]:
		return gatherNDDense(x, indices)
	case *tensor.Dense[uint8]:
		return gatherNDDense(x, indices)
	case *tensor.Dense[int8]:
		return gatherNDDense(x, indices)
	default:
		return nil, fmt.Errorf("GatherND: unsupported data type %T", data)
	}
}

func gatherNDDense[T tensor.Numeric](data *tensor.Dense[T], indices *tensor.Dense[int64]) ([]tensor.Tensor, error) {
	ds := data.Shape()
	is := indices.Shape()
	if is.NDim() == 0 {
		return nil, fmt.Errorf("GatherND: indices rank must be >= 1")
	}
	k := is[is.NDim()-1]
	if k > ds.NDim() {
		return nil, fmt.Errorf("GatherND: last index dim %d exceeds data rank %d", k, ds.NDim())
	}
	innerSize := 1
	for i := k; i < ds.NDim(); i++ {
		innerSize *= ds[i]
	}
	outShape := make(tensor.Shape, 0, is.NDim()-1+ds.NDim()-k)
	outShape = append(outShape, is[:is.NDim()-1]...)
	outShape = append(outShape, ds[k:]...)
	outData := make([]T, outShape.Size())
	dataStrides := tensor.Strides(ds)
	numSlices := 1
	for i := 0; i < is.NDim()-1; i++ {
		numSlices *= is[i]
	}
	idxData := indices.Data()
	src := data.Data()
	for s := 0; s < numSlices; s++ {
		base := 0
		for j := 0; j < k; j++ {
			coord := int(idxData[s*k+j])
			if coord < 0 {
				coord += ds[j]
			}
			base += coord * dataStrides[j]
		}
		copy(outData[s*innerSize:(s+1)*innerSize], src[base:base+innerSize])
	}
	return []tensor.Tensor{tensor.NewDense[T](outShape, outData)}, nil
}
