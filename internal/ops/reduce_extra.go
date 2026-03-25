package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Reduction extras ──

func opReduceMin(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axes := resolveReduceAxes(node, inputs)
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{reduceMinDense(x, axes, keepDims)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{reduceMinDense(x, axes, keepDims)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{reduceMinDense(x, axes, keepDims)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{reduceMinDense(x, axes, keepDims)}, nil
	default:
		return nil, fmt.Errorf("ReduceMin: unsupported type %T", inputs[0])
	}
}

func reduceMinDense[T tensor.Numeric](x *tensor.Dense[T], axes []int, keepDims bool) *tensor.Dense[T] {
	shape := x.Shape(); ndim := shape.NDim()
	for i, a := range axes { if a < 0 { axes[i] = a + ndim } }
	if len(axes) == 0 { axes = make([]int, ndim); for i := range axes { axes[i] = i } }
	axesSet := make(map[int]bool); for _, a := range axes { axesSet[a] = true }
	var outShape tensor.Shape
	for i, d := range shape {
		if axesSet[i] { if keepDims { outShape = append(outShape, 1) } } else { outShape = append(outShape, d) }
	}
	if len(outShape) == 0 { outShape = tensor.Shape{} }
	xData := x.Data()
	outData := make([]T, outShape.Size())
	// Init with first element
	if x.Len() > 0 { for i := range outData { outData[i] = xData[0] } }
	first := make([]bool, outShape.Size())
	strides := tensor.Strides(shape); outStrides := tensor.Strides(outShape)
	for i := 0; i < x.Len(); i++ {
		outIdx := 0; rem := i; outDim := 0
		for d := 0; d < ndim; d++ {
			coord := rem / strides[d]; rem %= strides[d]
			if !axesSet[d] { if outDim < len(outStrides) { outIdx += coord * outStrides[outDim] }; outDim++ } else if keepDims { outDim++ }
		}
		if !first[outIdx] { outData[outIdx] = xData[i]; first[outIdx] = true } else if xData[i] < outData[outIdx] { outData[outIdx] = xData[i] }
	}
	return tensor.NewDense[T](outShape, outData)
}

func opReduceL1(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axesInt := resolveReduceAxes(node, inputs)
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		// ReduceL1 = ReduceSum(Abs(x))
		absData := make([]float32, x.Len())
		for i, v := range x.Data() { if v < 0 { absData[i] = -v } else { absData[i] = v } }
		absT := tensor.NewDense[float32](x.Shape().Clone(), absData)
		return []tensor.Tensor{reduceSumDense(absT, axesInt, keepDims, false)}, nil
	default:
		return nil, fmt.Errorf("ReduceL1: unsupported type %T", inputs[0])
	}
}

func opReduceL2(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axesInt := resolveReduceAxes(node, inputs)
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		sqData := make([]float32, x.Len())
		for i, v := range x.Data() { sqData[i] = v * v }
		sqT := tensor.NewDense[float32](x.Shape().Clone(), sqData)
		result := reduceSumDense(sqT, axesInt, keepDims, false)
		rd := result.Data()
		for i, v := range rd { rd[i] = float32(math.Sqrt(float64(v))) }
		return []tensor.Tensor{result}, nil
	default:
		return nil, fmt.Errorf("ReduceL2: unsupported type %T", inputs[0])
	}
}

func opSum(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if len(inputs) == 0 { return nil, fmt.Errorf("Sum: no inputs") }
	result := inputs[0].Clone()
	for i := 1; i < len(inputs); i++ {
		r, err := dispatchBinaryOp([]tensor.Tensor{result, inputs[i]},
			func(a, b float32) float32 { return a + b }, func(a, b float64) float64 { return a + b },
			func(a, b int32) int32 { return a + b }, func(a, b int64) int64 { return a + b })
		if err != nil { return nil, err }
		result = r
	}
	return []tensor.Tensor{result}, nil
}

func opMean(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	sumOut, err := opSum(node, inputs)
	if err != nil { return nil, err }
	n := float32(len(inputs))
	switch t := sumOut[0].(type) {
	case *tensor.Dense[float32]:
		d := t.Data()
		for i := range d { d[i] /= n }
		return sumOut, nil
	default:
		return nil, fmt.Errorf("Mean: unsupported type %T", sumOut[0])
	}
}

// ── ReduceLogSum / ReduceLogSumExp ──

func opReduceLogSum(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axes := resolveReduceAxes(node, inputs)
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		result := reduceSumDense(x, axes, keepDims, false)
		rd := result.Data()
		for i, v := range rd { rd[i] = float32(math.Log(float64(v))) }
		return []tensor.Tensor{result}, nil
	default:
		return nil, fmt.Errorf("ReduceLogSum: unsupported type %T", inputs[0])
	}
}

func opReduceLogSumExp(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	keepDims := node.GetAttrInt("keepdims", 1) != 0
	axes := resolveReduceAxes(node, inputs)
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		// exp(x) then sum then log
		expData := make([]float32, x.Len())
		for i, v := range x.Data() { expData[i] = float32(math.Exp(float64(v))) }
		expT := tensor.NewDense[float32](x.Shape().Clone(), expData)
		result := reduceSumDense(expT, axes, keepDims, false)
		rd := result.Data()
		for i, v := range rd { rd[i] = float32(math.Log(float64(v))) }
		return []tensor.Tensor{result}, nil
	default:
		return nil, fmt.Errorf("ReduceLogSumExp: unsupported type %T", inputs[0])
	}
}
