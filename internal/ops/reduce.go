package ops

import (
	"fmt"
	"sort"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// TopK
func opTopK(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", -1))
	largest := node.GetAttrInt("largest", 1) != 0

	k := 0
	// opset >= 10: k from input[1]; opset < 10: k from attribute
	if len(inputs) > 1 && inputs[1] != nil {
		kT := inputs[1].(*tensor.Dense[int64])
		k = int(kT.At(0))
	}
	if k == 0 {
		k = int(node.GetAttrInt("k", 0))
	}
	if k == 0 {
		return nil, fmt.Errorf("TopK: k=0")
	}

	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return topKDense(dt, k, axis, largest)
	default:
		return nil, fmt.Errorf("TopK: unsupported type %T", inputs[0])
	}
}

func topKDense[T tensor.Numeric](t *tensor.Dense[T], k, axis int, largest bool) ([]tensor.Tensor, error) {
	shape := t.Shape()
	ndim := shape.NDim()
	if axis < 0 {
		axis += ndim
	}
	axisSize := shape[axis]
	if k > axisSize {
		k = axisSize
	}

	outShape := shape.Clone()
	outShape[axis] = k

	outerSize := 1
	for d := 0; d < axis; d++ {
		outerSize *= shape[d]
	}
	innerSize := 1
	for d := axis + 1; d < ndim; d++ {
		innerSize *= shape[d]
	}

	src := t.Data()
	valOut := make([]T, outShape.Size())
	idxOut := make([]int64, outShape.Size())

	type vi struct {
		val T
		idx int
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			items := make([]vi, axisSize)
			for a := 0; a < axisSize; a++ {
				srcIdx := outer*axisSize*innerSize + a*innerSize + inner
				items[a] = vi{src[srcIdx], a}
			}
			if largest {
				sort.Slice(items, func(i, j int) bool {
					if items[i].val == items[j].val {
						return items[i].idx < items[j].idx
					}
					return items[i].val > items[j].val
				})
			} else {
				sort.Slice(items, func(i, j int) bool {
					if items[i].val == items[j].val {
						return items[i].idx < items[j].idx
					}
					return items[i].val < items[j].val
				})
			}
			for a := 0; a < k; a++ {
				outIdx := outer*k*innerSize + a*innerSize + inner
				valOut[outIdx] = items[a].val
				idxOut[outIdx] = int64(items[a].idx)
			}
		}
	}

	return []tensor.Tensor{
		tensor.NewDense[T](outShape.Clone(), valOut),
		tensor.NewDense[int64](outShape, idxOut),
	}, nil
}

// resolveReduceAxes resolves axes for Reduce ops.
// ReduceSum: axes moved to input at opset 13.
// Other Reduce ops: axes moved to input at opset 18.
// For compatibility, both sources are checked with fallback.
func resolveReduceAxes(node *ir.Node, inputs []tensor.Tensor) []int {
	// ReduceSum switched at opset 13; all others at opset 18
	threshold := int64(18)
	if node.OpType == "ReduceSum" {
		threshold = 13
	}
	var axes []int
	if node.OpsetVersion >= threshold || (node.OpsetVersion == 0 && threshold == 18) {
		// opset 18+: axes from input first
		if len(inputs) > 1 && inputs[1] != nil {
			if axT, ok := inputs[1].(*tensor.Dense[int64]); ok {
				for _, a := range axT.Data() {
					axes = append(axes, int(a))
				}
			}
		}
		if len(axes) == 0 {
			// fallback to attribute for compatibility
			for _, a := range node.GetAttrInts("axes", nil) {
				axes = append(axes, int(a))
			}
		}
	} else {
		// opset < 18: axes from attribute first
		for _, a := range node.GetAttrInts("axes", nil) {
			axes = append(axes, int(a))
		}
		if len(axes) == 0 && len(inputs) > 1 && inputs[1] != nil {
			if axT, ok := inputs[1].(*tensor.Dense[int64]); ok {
				for _, a := range axT.Data() {
					axes = append(axes, int(a))
				}
			}
		}
	}
	return axes
}

// ReduceMax
func opReduceMax(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axes := resolveReduceAxes(node, inputs)

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{reduceMaxDense(x, axes, keepDims)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{reduceMaxDense(x, axes, keepDims)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{reduceMaxDense(x, axes, keepDims)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{reduceMaxDense(x, axes, keepDims)}, nil
	default:
		return nil, fmt.Errorf("ReduceMax: unsupported type %T", inputs[0])
	}
}

// reduceAxes delegates to resolveReduceAxes for opset-aware axis resolution.
func reduceAxes(node *ir.Node, inputs []tensor.Tensor) []int {
	return resolveReduceAxes(node, inputs)
}

func reduceMeta(shape tensor.Shape, axes []int, keepDims bool) (tensor.Shape, map[int]bool) {
	ndim := shape.NDim()
	for i, a := range axes {
		if a < 0 {
			axes[i] = a + ndim
		}
	}
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = i
		}
	}
	axesSet := make(map[int]bool)
	for _, a := range axes {
		axesSet[a] = true
	}
	var outShape tensor.Shape
	for i, d := range shape {
		if axesSet[i] {
			if keepDims {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}
	return outShape, axesSet
}

func opReduceProd(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axes := reduceAxes(node, inputs)
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{reduceProdDense(x, axes, keepDims)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{reduceProdDense(x, axes, keepDims)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{reduceProdDense(x, axes, keepDims)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{reduceProdDense(x, axes, keepDims)}, nil
	default:
		return nil, fmt.Errorf("ReduceProd: unsupported type %T", inputs[0])
	}
}

func reduceProdDense[T tensor.Numeric](x *tensor.Dense[T], axes []int, keepDims bool) *tensor.Dense[T] {
	shape := x.Shape()
	ndim := shape.NDim()
	outShape, axesSet := reduceMeta(shape, axes, keepDims)
	outData := make([]T, outShape.Size())
	for i := range outData {
		outData[i] = T(1)
	}
	xData := x.Data()
	strides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)
	for i := 0; i < x.Len(); i++ {
		outIdx := 0
		remaining := i
		outDim := 0
		for d := 0; d < ndim; d++ {
			coord := remaining / strides[d]
			remaining %= strides[d]
			if !axesSet[d] {
				if outDim < len(outStrides) {
					outIdx += coord * outStrides[outDim]
				}
				outDim++
			} else if keepDims {
				outDim++
			}
		}
		outData[outIdx] *= xData[i]
	}
	return tensor.NewDense[T](outShape, outData)
}

func opReduceSumSquare(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axes := reduceAxes(node, inputs)
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{reduceSumSquareDense(x, axes, keepDims)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{reduceSumSquareDense(x, axes, keepDims)}, nil
	default:
		return nil, fmt.Errorf("ReduceSumSquare: unsupported type %T", inputs[0])
	}
}

func reduceSumSquareDense[T tensor.Numeric](x *tensor.Dense[T], axes []int, keepDims bool) *tensor.Dense[T] {
	shape := x.Shape()
	ndim := shape.NDim()
	outShape, axesSet := reduceMeta(shape, axes, keepDims)
	outData := make([]T, outShape.Size())
	xData := x.Data()
	strides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)
	for i := 0; i < x.Len(); i++ {
		outIdx := 0
		remaining := i
		outDim := 0
		for d := 0; d < ndim; d++ {
			coord := remaining / strides[d]
			remaining %= strides[d]
			if !axesSet[d] {
				if outDim < len(outStrides) {
					outIdx += coord * outStrides[outDim]
				}
				outDim++
			} else if keepDims {
				outDim++
			}
		}
		outData[outIdx] += xData[i] * xData[i]
	}
	return tensor.NewDense[T](outShape, outData)
}

func reduceMaxDense[T tensor.Numeric](x *tensor.Dense[T], axes []int, keepDims bool) *tensor.Dense[T] {
	shape := x.Shape()
	ndim := shape.NDim()
	for i, a := range axes {
		if a < 0 {
			axes[i] = a + ndim
		}
	}
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = i
		}
	}
	axesSet := make(map[int]bool)
	for _, a := range axes {
		axesSet[a] = true
	}

	var outShape tensor.Shape
	for i, d := range shape {
		if axesSet[i] {
			if keepDims {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}

	xData := x.Data()
	outData := make([]T, outShape.Size())
	// Init with first element value (safe for all numeric types)
	if x.Len() > 0 {
		for i := range outData {
			outData[i] = xData[0]
		}
	}
	strides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)

	for i := 0; i < x.Len(); i++ {
		outIdx := 0
		remaining := i
		outDim := 0
		for d := 0; d < ndim; d++ {
			coord := remaining / strides[d]
			remaining %= strides[d]
			if !axesSet[d] {
				if outDim < len(outStrides) {
					outIdx += coord * outStrides[outDim]
				}
				outDim++
			} else if keepDims {
				outDim++
			}
		}
		if xData[i] > outData[outIdx] {
			outData[outIdx] = xData[i]
		}
	}
	return tensor.NewDense[T](outShape, outData)
}

// ReduceSum
func opReduceSum(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	noopEmpty := node.GetAttrInt("noop_with_empty_axes", 0) != 0
	axes := resolveReduceAxes(node, inputs)

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{reduceSumDense(x, axes, keepDims, noopEmpty)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{reduceSumDense(x, axes, keepDims, noopEmpty)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{reduceSumDense(x, axes, keepDims, noopEmpty)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{reduceSumDense(x, axes, keepDims, noopEmpty)}, nil
	default:
		return nil, fmt.Errorf("ReduceSum: unsupported type %T", inputs[0])
	}
}

func opReduceMean(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axes := resolveReduceAxes(node, inputs)

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out, err := reduceMeanDenseFloat32(x, axes, keepDims)
		if err != nil {
			return nil, fmt.Errorf("ReduceMean: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		out, err := reduceMeanDenseFloat64(x, axes, keepDims)
		if err != nil {
			return nil, fmt.Errorf("ReduceMean: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("ReduceMean: unsupported type %T", inputs[0])
	}
}

func reduceMeanDense[T tensor.Numeric](x *tensor.Dense[T], axes []int, keepDims bool) (*tensor.Dense[T], error) {
	shape := x.Shape()
	ndim := shape.NDim()
	for i, a := range axes {
		if a < 0 {
			axes[i] = a + ndim
		}
	}
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = i
		}
	}

	axesSet := make(map[int]bool)
	for _, a := range axes {
		axesSet[a] = true
	}

	var outShape tensor.Shape
	reduceCount := 1
	for i, d := range shape {
		if axesSet[i] {
			reduceCount *= d
			if keepDims {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}

	outData := make([]T, outShape.Size())
	xData := x.Data()
	strides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)
	for i := 0; i < x.Len(); i++ {
		outIdx := 0
		remaining := i
		outDim := 0
		for d := 0; d < ndim; d++ {
			coord := remaining / strides[d]
			remaining %= strides[d]
			if !axesSet[d] {
				if outDim < len(outStrides) {
					outIdx += coord * outStrides[outDim]
				}
				outDim++
			} else if keepDims {
				outDim++
			}
		}
		outData[outIdx] += xData[i]
	}
	for i := range outData {
		outData[i] = T(float64(outData[i]) / float64(reduceCount))
	}
	return tensor.NewDense[T](outShape, outData), nil
}

func reduceMeanDenseFloat32(x *tensor.Dense[float32], axes []int, keepDims bool) (*tensor.Dense[float32], error) {
	shape, outShape, axesSet, err := reduceMeanMeta(x.Shape(), axes, keepDims)
	if err != nil {
		return nil, err
	}

	outData := make([]float64, outShape.Size())
	xData := x.Data()
	strides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)
	ndim := shape.NDim()
	reduceCount := reduceMeanCount(shape, axesSet)

	for i := 0; i < x.Len(); i++ {
		outIdx := 0
		remaining := i
		outDim := 0
		for d := 0; d < ndim; d++ {
			coord := remaining / strides[d]
			remaining %= strides[d]
			if !axesSet[d] {
				if outDim < len(outStrides) {
					outIdx += coord * outStrides[outDim]
				}
				outDim++
			} else if keepDims {
				outDim++
			}
		}
		outData[outIdx] += float64(xData[i])
	}

	scale := 1.0 / float64(reduceCount)
	data := make([]float32, len(outData))
	for i := range outData {
		data[i] = float32(outData[i] * scale)
	}
	return tensor.NewDense[float32](outShape, data), nil
}

func reduceMeanDenseFloat64(x *tensor.Dense[float64], axes []int, keepDims bool) (*tensor.Dense[float64], error) {
	return reduceMeanDense(x, axes, keepDims)
}

func reduceMeanMeta(shape tensor.Shape, axes []int, keepDims bool) (tensor.Shape, tensor.Shape, map[int]bool, error) {
	ndim := shape.NDim()
	normAxes := append([]int(nil), axes...)
	for i, a := range normAxes {
		if a < 0 {
			normAxes[i] = a + ndim
		}
	}
	if len(normAxes) == 0 {
		normAxes = make([]int, ndim)
		for i := range normAxes {
			normAxes[i] = i
		}
	}

	axesSet := make(map[int]bool, len(normAxes))
	for _, a := range normAxes {
		if a < 0 || a >= ndim {
			return nil, nil, nil, fmt.Errorf("axis %d out of range for rank %d", a, ndim)
		}
		axesSet[a] = true
	}

	var outShape tensor.Shape
	for i, d := range shape {
		if axesSet[i] {
			if keepDims {
				outShape = append(outShape, 1)
			}
			continue
		}
		outShape = append(outShape, d)
	}
	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}
	return shape, outShape, axesSet, nil
}

func reduceMeanCount(shape tensor.Shape, axesSet map[int]bool) int {
	reduceCount := 1
	for i, d := range shape {
		if axesSet[i] {
			reduceCount *= d
		}
	}
	return reduceCount
}

func opArgMax(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", 0))
	keepDims := node.GetAttrInt("keepdims", 1) != 0

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return argMaxDense(x, axis, keepDims)
	case *tensor.Dense[float64]:
		return argMaxDense(x, axis, keepDims)
	default:
		return nil, fmt.Errorf("ArgMax: unsupported type %T", inputs[0])
	}
}

func argMaxDense[T tensor.Numeric](x *tensor.Dense[T], axis int, keepDims bool) ([]tensor.Tensor, error) {
	shape := x.Shape()
	ndim := shape.NDim()
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("ArgMax: axis %d out of range for rank %d", axis, ndim)
	}

	outShape := make(tensor.Shape, 0, ndim)
	for i, d := range shape {
		if i == axis {
			if keepDims {
				outShape = append(outShape, 1)
			}
			continue
		}
		outShape = append(outShape, d)
	}
	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	innerSize := 1
	for i := axis + 1; i < ndim; i++ {
		innerSize *= shape[i]
	}
	axisSize := shape[axis]

	data := x.Data()
	outLen := outShape.Size()
	if outLen == 0 {
		outLen = 1
	}
	out := make([]int64, outLen)
	outIdx := 0
	for outer := 0; outer < outerSize; outer++ {
		base := outer * axisSize * innerSize
		for inner := 0; inner < innerSize; inner++ {
			bestIdx := 0
			bestVal := data[base+inner]
			for a := 1; a < axisSize; a++ {
				v := data[base+a*innerSize+inner]
				if v > bestVal {
					bestVal = v
					bestIdx = a
				}
			}
			out[outIdx] = int64(bestIdx)
			outIdx++
		}
	}
	return []tensor.Tensor{tensor.NewDense[int64](outShape, out)}, nil
}

func opArgMin(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", 0))
	keepDims := node.GetAttrInt("keepdims", 1) != 0

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return argMinDense(x, axis, keepDims)
	case *tensor.Dense[float64]:
		return argMinDense(x, axis, keepDims)
	default:
		return nil, fmt.Errorf("ArgMin: unsupported type %T", inputs[0])
	}
}

func argMinDense[T tensor.Numeric](x *tensor.Dense[T], axis int, keepDims bool) ([]tensor.Tensor, error) {
	shape := x.Shape()
	ndim := shape.NDim()
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("ArgMin: axis %d out of range for rank %d", axis, ndim)
	}

	outShape := make(tensor.Shape, 0, ndim)
	for i, d := range shape {
		if i == axis {
			if keepDims {
				outShape = append(outShape, 1)
			}
			continue
		}
		outShape = append(outShape, d)
	}
	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	innerSize := 1
	for i := axis + 1; i < ndim; i++ {
		innerSize *= shape[i]
	}
	axisSize := shape[axis]

	data := x.Data()
	outLen := outShape.Size()
	if outLen == 0 {
		outLen = 1
	}
	out := make([]int64, outLen)
	outIdx := 0
	for outer := 0; outer < outerSize; outer++ {
		base := outer * axisSize * innerSize
		for inner := 0; inner < innerSize; inner++ {
			bestIdx := 0
			bestVal := data[base+inner]
			for a := 1; a < axisSize; a++ {
				v := data[base+a*innerSize+inner]
				if v < bestVal {
					bestVal = v
					bestIdx = a
				}
			}
			out[outIdx] = int64(bestIdx)
			outIdx++
		}
	}
	return []tensor.Tensor{tensor.NewDense[int64](outShape, out)}, nil
}

func reduceSumDense[T tensor.Numeric](x *tensor.Dense[T], axes []int, keepDims, noopEmpty bool) *tensor.Dense[T] {
	shape := x.Shape()
	ndim := shape.NDim()
	if len(axes) == 0 && noopEmpty {
		return x
	}
	for i, a := range axes {
		if a < 0 {
			axes[i] = a + ndim
		}
	}
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = i
		}
	}
	axesSet := make(map[int]bool)
	for _, a := range axes {
		axesSet[a] = true
	}

	var outShape tensor.Shape
	for i, d := range shape {
		if axesSet[i] {
			if keepDims {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = tensor.Shape{}
	}

	outData := make([]T, outShape.Size())
	xData := x.Data()
	strides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)

	for i := 0; i < x.Len(); i++ {
		outIdx := 0
		remaining := i
		outDim := 0
		for d := 0; d < ndim; d++ {
			coord := remaining / strides[d]
			remaining %= strides[d]
			if !axesSet[d] {
				if outDim < len(outStrides) {
					outIdx += coord * outStrides[outDim]
				}
				outDim++
			} else if keepDims {
				outDim++
			}
		}
		outData[outIdx] += xData[i]
	}
	return tensor.NewDense[T](outShape, outData)
}
