package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func opReshape(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	data := inputs[0]
	shapeT := inputs[1]

	// Shape tensor must be int64
	shapeDense, ok := shapeT.(*tensor.Dense[int64])
	if !ok {
		return nil, fmt.Errorf("Reshape: shape input must be int64, got %T", shapeT)
	}

	newShape := make(tensor.Shape, shapeDense.Len())
	totalSize := data.Shape().Size()
	unknownIdx := -1

	for i, v := range shapeDense.Data() {
		if v == 0 {
			// 0 means keep original dimension
			if i < data.Shape().NDim() {
				newShape[i] = data.Shape()[i]
			} else {
				newShape[i] = 1
			}
		} else if v == -1 {
			unknownIdx = i
			newShape[i] = -1
		} else {
			newShape[i] = int(v)
		}
	}

	if unknownIdx >= 0 {
		known := 1
		for i, v := range newShape {
			if i != unknownIdx {
				known *= v
			}
		}
		if known == 0 {
			return nil, fmt.Errorf("Reshape: cannot infer dimension with zero-size known dimensions")
		}
		newShape[unknownIdx] = totalSize / known
	}

	// Create new tensor with copied data
	out := reshapeTensor(data, newShape)
	if out == nil {
		return nil, fmt.Errorf("Reshape: unsupported type %T", data)
	}
	return []tensor.Tensor{out}, nil
}

func opShape(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if len(inputs) == 0 || inputs[0] == nil {
		return nil, fmt.Errorf("Shape: no input")
	}
	s := inputs[0].Shape()
	data := make([]int64, len(s))
	for i, d := range s {
		data[i] = int64(d)
	}
	return []tensor.Tensor{tensor.NewDense[int64](tensor.Shape{len(s)}, data)}, nil
}

func reshapeTensor(t tensor.Tensor, newShape tensor.Shape) tensor.Tensor {
	switch dt := t.(type) {
	case *tensor.Dense[float32]:
		return dt.Reshape(newShape)
	case *tensor.Dense[float64]:
		return dt.Reshape(newShape)
	case *tensor.Dense[int32]:
		return dt.Reshape(newShape)
	case *tensor.Dense[int64]:
		return dt.Reshape(newShape)
	case *tensor.Dense[uint8]:
		return dt.Reshape(newShape)
	case *tensor.Dense[int8]:
		return dt.Reshape(newShape)
	default:
		return nil
	}
}

func opTranspose(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	t := inputs[0]
	perm := node.GetAttrInts("perm", nil)

	switch dt := t.(type) {
	case *tensor.Dense[float32]:
		out, err := transposeDense(dt, perm)
		if err != nil {
			return nil, fmt.Errorf("Transpose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		out, err := transposeDense(dt, perm)
		if err != nil {
			return nil, fmt.Errorf("Transpose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[int32]:
		out, err := transposeDense(dt, perm)
		if err != nil {
			return nil, fmt.Errorf("Transpose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[int64]:
		out, err := transposeDense(dt, perm)
		if err != nil {
			return nil, fmt.Errorf("Transpose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[uint8]:
		out, err := transposeDense(dt, perm)
		if err != nil {
			return nil, fmt.Errorf("Transpose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[int8]:
		out, err := transposeDense(dt, perm)
		if err != nil {
			return nil, fmt.Errorf("Transpose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("Transpose: unsupported type %T", t)
	}
}

func transposeDense[T tensor.Numeric](t *tensor.Dense[T], perm []int64) (*tensor.Dense[T], error) {
	shape := t.Shape()
	ndim := shape.NDim()

	// Default perm: reverse dimensions
	if perm == nil {
		perm = make([]int64, ndim)
		for i := range perm {
			perm[i] = int64(ndim - 1 - i)
		}
	}

	newShape := make(tensor.Shape, ndim)
	for i, p := range perm {
		newShape[i] = shape[int(p)]
	}

	srcStrides := tensor.Strides(shape)
	dstStrides := tensor.Strides(newShape)
	data := make([]T, t.Len())
	src := t.Data()

	for i := 0; i < t.Len(); i++ {
		// Decompose dst flat index into dst coords
		remaining := i
		srcIdx := 0
		for d := 0; d < ndim; d++ {
			coord := remaining / dstStrides[d]
			remaining %= dstStrides[d]
			srcIdx += coord * srcStrides[int(perm[d])]
		}
		data[i] = src[srcIdx]
	}

	return tensor.NewDense[T](newShape, data), nil
}

func opSqueeze(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	t := inputs[0]
	shape := t.Shape()

	// Axes from second input (opset 13+) or attribute (opset < 13)
	var axes []int
	if len(inputs) > 1 && inputs[1] != nil {
		axesT := inputs[1].(*tensor.Dense[int64])
		for _, v := range axesT.Data() {
			a := int(v)
			if a < 0 {
				a += shape.NDim()
			}
			axes = append(axes, a)
		}
	} else {
		attrAxes := node.GetAttrInts("axes", nil)
		if attrAxes != nil {
			for _, v := range attrAxes {
				a := int(v)
				if a < 0 {
					a += shape.NDim()
				}
				axes = append(axes, a)
			}
		} else {
			// Squeeze all dims of size 1
			for i, d := range shape {
				if d == 1 {
					axes = append(axes, i)
				}
			}
		}
	}

	axesSet := make(map[int]bool)
	for _, a := range axes {
		axesSet[a] = true
	}

	var newShape tensor.Shape
	for i, d := range shape {
		if !axesSet[i] {
			newShape = append(newShape, d)
		}
	}

	if len(newShape) == 0 {
		newShape = tensor.Shape{}
	}

	out := reshapeTensor(t, newShape)
	if out == nil {
		return nil, fmt.Errorf("Squeeze: unsupported type %T", t)
	}
	return []tensor.Tensor{out}, nil
}

func opUnsqueeze(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	t := inputs[0]
	shape := t.Shape()

	// Axes from second input (opset 13+) or attribute
	var axes []int
	if len(inputs) > 1 && inputs[1] != nil {
		axesT := inputs[1].(*tensor.Dense[int64])
		for _, v := range axesT.Data() {
			axes = append(axes, int(v))
		}
	} else {
		attrAxes := node.GetAttrInts("axes", nil)
		for _, v := range attrAxes {
			axes = append(axes, int(v))
		}
	}

	newNdim := shape.NDim() + len(axes)

	// Normalize negative axes
	for i, a := range axes {
		if a < 0 {
			axes[i] = a + newNdim
		}
	}

	axesSet := make(map[int]bool)
	for _, a := range axes {
		axesSet[a] = true
	}

	newShape := make(tensor.Shape, newNdim)
	srcIdx := 0
	for i := 0; i < newNdim; i++ {
		if axesSet[i] {
			newShape[i] = 1
		} else {
			newShape[i] = shape[srcIdx]
			srcIdx++
		}
	}

	out := reshapeTensor(t, newShape)
	if out == nil {
		return nil, fmt.Errorf("Unsqueeze: unsupported type %T", t)
	}
	return []tensor.Tensor{out}, nil
}

func opFlatten(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	t := inputs[0]
	shape := t.Shape()
	axis := int(node.GetAttrInt("axis", 1))

	if axis < 0 {
		axis += shape.NDim()
	}

	dim0 := 1
	for i := 0; i < axis; i++ {
		dim0 *= shape[i]
	}
	dim1 := 1
	for i := axis; i < shape.NDim(); i++ {
		dim1 *= shape[i]
	}

	newShape := tensor.Shape{dim0, dim1}
	out := reshapeTensor(t, newShape)
	if out == nil {
		return nil, fmt.Errorf("Flatten: unsupported type %T", t)
	}
	return []tensor.Tensor{out}, nil
}

func opConcat(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", 0))

	filtered := make([]tensor.Tensor, 0, len(inputs))
	for _, input := range inputs {
		if input != nil {
			filtered = append(filtered, input)
		}
	}
	if len(filtered) == 0 {
		return nil, fmt.Errorf("Concat: no inputs")
	}
	if len(filtered) == 1 {
		return []tensor.Tensor{filtered[0]}, nil
	}

	switch filtered[0].(type) {
	case *tensor.Dense[float32]:
		return concatTyped[float32](filtered, axis)
	case *tensor.Dense[float64]:
		return concatTyped[float64](filtered, axis)
	case *tensor.Dense[int32]:
		return concatTyped[int32](filtered, axis)
	case *tensor.Dense[int64]:
		return concatTyped[int64](filtered, axis)
	case *tensor.Dense[uint8]:
		return concatTyped[uint8](filtered, axis)
	case *tensor.Dense[int8]:
		return concatTyped[int8](filtered, axis)
	default:
		return nil, fmt.Errorf("Concat: unsupported type %T", filtered[0])
	}
}

func concatTyped[T tensor.Numeric](inputs []tensor.Tensor, axis int) ([]tensor.Tensor, error) {
	tensors := make([]*tensor.Dense[T], len(inputs))
	for i, inp := range inputs {
		t, ok := inp.(*tensor.Dense[T])
		if !ok {
			return nil, fmt.Errorf("Concat: type mismatch at input %d", i)
		}
		tensors[i] = t
	}

	shape := tensors[0].Shape()
	ndim := shape.NDim()
	if axis < 0 {
		axis += ndim
	}

	// Compute output shape
	outShape := shape.Clone()
	totalAxis := shape[axis]
	for i := 1; i < len(tensors); i++ {
		totalAxis += tensors[i].Shape()[axis]
	}
	outShape[axis] = totalAxis

	outData := make([]T, outShape.Size())

	// Copy data
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= outShape[i]
	}
	innerSize := 1
	for i := axis + 1; i < ndim; i++ {
		innerSize *= outShape[i]
	}

	outOffset := 0
	for outer := 0; outer < outerSize; outer++ {
		for _, t := range tensors {
			chunkSize := t.Shape()[axis] * innerSize
			srcOffset := outer * chunkSize
			copy(outData[outOffset:outOffset+chunkSize], t.Data()[srcOffset:srcOffset+chunkSize])
			outOffset += chunkSize
		}
	}

	return []tensor.Tensor{tensor.NewDense[T](outShape, outData)}, nil
}

func opSplit(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", 0))

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return splitDense(x, node, inputs, axis)
	case *tensor.Dense[float64]:
		return splitDense(x, node, inputs, axis)
	case *tensor.Dense[int32]:
		return splitDense(x, node, inputs, axis)
	case *tensor.Dense[int64]:
		return splitDense(x, node, inputs, axis)
	case *tensor.Dense[uint8]:
		return splitDense(x, node, inputs, axis)
	case *tensor.Dense[int8]:
		return splitDense(x, node, inputs, axis)
	default:
		return nil, fmt.Errorf("Split: unsupported type %T", inputs[0])
	}
}

func splitDense[T tensor.Numeric](x *tensor.Dense[T], node *ir.Node, inputs []tensor.Tensor, axis int) ([]tensor.Tensor, error) {
	shape := x.Shape()
	ndim := shape.NDim()
	if axis < 0 {
		axis += ndim
	}
	axisSize := shape[axis]

	var splits []int
	if len(inputs) > 1 && inputs[1] != nil {
		if sp, ok := inputs[1].(*tensor.Dense[int64]); ok {
			for _, v := range sp.Data() {
				splits = append(splits, int(v))
			}
		}
	}
	if len(splits) == 0 {
		attrSplits := node.GetAttrInts("split", nil)
		if attrSplits != nil {
			for _, v := range attrSplits {
				splits = append(splits, int(v))
			}
		}
	}
	if len(splits) == 0 {
		numOutputs := len(node.Outputs)
		if numOutputs == 0 {
			numOutputs = 2
		}
		splitSize := axisSize / numOutputs
		for i := 0; i < numOutputs; i++ {
			splits = append(splits, splitSize)
		}
		rem := axisSize - splitSize*numOutputs
		if rem > 0 {
			splits[len(splits)-1] += rem
		}
	}

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	innerSize := 1
	for i := axis + 1; i < ndim; i++ {
		innerSize *= shape[i]
	}

	xData := x.Data()
	var outputs []tensor.Tensor
	axisOffset := 0

	for _, splitSize := range splits {
		outShape := shape.Clone()
		outShape[axis] = splitSize
		outData := make([]T, outShape.Size())

		for outer := 0; outer < outerSize; outer++ {
			srcBase := outer*axisSize*innerSize + axisOffset*innerSize
			dstBase := outer * splitSize * innerSize
			for s := 0; s < splitSize; s++ {
				copy(outData[dstBase+s*innerSize:dstBase+(s+1)*innerSize],
					xData[srcBase+s*innerSize:srcBase+(s+1)*innerSize])
			}
		}

		outputs = append(outputs, tensor.NewDense[T](outShape, outData))
		axisOffset += splitSize
	}

	return outputs, nil
}
