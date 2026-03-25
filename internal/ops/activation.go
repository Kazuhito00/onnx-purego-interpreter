package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func opRelu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{reluDense(t)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{reluDense(t)}, nil
	default:
		return nil, fmt.Errorf("Relu: unsupported type %T", inputs[0])
	}
}

func opLeakyRelu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	alpha := node.GetAttrFloat("alpha", 0.01)
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			if v >= 0 {
				data[i] = v
			} else {
				data[i] = v * alpha
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			if v >= 0 {
				data[i] = v
			} else {
				data[i] = v * float64(alpha)
			}
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("LeakyRelu: unsupported type %T", inputs[0])
	}
}

// PRelu: Y = X if X >= 0, slope * X otherwise. slope is per-channel.
func opPRelu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		slope := inputs[1].(*tensor.Dense[float32])
		return []tensor.Tensor{preluF32(x, slope)}, nil
	default:
		return nil, fmt.Errorf("PRelu: unsupported type %T", inputs[0])
	}
}

func reluDense[T tensor.Numeric](t *tensor.Dense[T]) *tensor.Dense[T] {
	data := make([]T, t.Len())
	src := t.Data()
	var zero T
	for i, v := range src {
		if v > zero {
			data[i] = v
		}
	}
	return tensor.NewDense[T](t.Shape().Clone(), data)
}

func opSigmoid(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = 1.0 / (1.0 + math.Exp(-v))
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Sigmoid: unsupported type %T", inputs[0])
	}
}

func opHardSigmoid(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	alpha := float64(node.GetAttrFloat("alpha", 0.2))
	beta := float64(node.GetAttrFloat("beta", 0.5))
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			y := alpha*float64(v) + beta
			if y < 0 {
				y = 0
			} else if y > 1 {
				y = 1
			}
			data[i] = float32(y)
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			y := alpha*v + beta
			if y < 0 {
				y = 0
			} else if y > 1 {
				y = 1
			}
			data[i] = y
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("HardSigmoid: unsupported type %T", inputs[0])
	}
}

func opHardSwish(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			hsig := float32(math.Min(math.Max(float64(v+3), 0), 6) / 6.0)
			data[i] = v * hsig
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			hsig := math.Min(math.Max(v+3, 0), 6) / 6.0
			data[i] = v * hsig
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("HardSwish: unsupported type %T", inputs[0])
	}
}

func opTanh(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Tanh(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Tanh(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Tanh: unsupported type %T", inputs[0])
	}
}

func opSoftmax(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Default axis: -1 for opset >= 13, 1 for opset < 13
	legacyMode := node.OpsetVersion > 0 && node.OpsetVersion < 13
	defaultAxis := int64(-1)
	if legacyMode {
		defaultAxis = 1
	}
	axis := int(node.GetAttrInt("axis", defaultAxis))

	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out, err := softmaxDispatch(t, axis, legacyMode)
		if err != nil {
			return nil, fmt.Errorf("Softmax: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		out, err := softmaxDispatch(t, axis, legacyMode)
		if err != nil {
			return nil, fmt.Errorf("Softmax: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("Softmax: unsupported type %T", inputs[0])
	}
}

// softmaxDispatch handles opset < 13 (flatten from axis) vs >= 13 (single axis).
func softmaxDispatch[T tensor.Numeric](t *tensor.Dense[T], axis int, legacy bool) (*tensor.Dense[T], error) {
	shape := t.Shape()
	ndim := shape.NDim()
	if axis < 0 {
		axis += ndim
	}

	if !legacy || ndim <= 2 {
		return softmaxDense(t, axis)
	}

	// opset < 13: flatten [d0..d_{axis-1}] and [d_axis..d_{ndim-1}], softmax, reshape back
	outerDim := 1
	for i := 0; i < axis; i++ {
		outerDim *= shape[i]
	}
	innerDim := 1
	for i := axis; i < ndim; i++ {
		innerDim *= shape[i]
	}
	flat := t.Reshape(tensor.Shape{outerDim, innerDim})
	result, err := softmaxDense(flat, 1)
	if err != nil {
		return nil, err
	}
	return result.Reshape(shape.Clone()), nil
}

func softmaxDense[T tensor.Numeric](t *tensor.Dense[T], axis int) (*tensor.Dense[T], error) {
	shape := t.Shape()
	ndim := shape.NDim()

	if ndim == 0 {
		return tensor.NewDense[T](shape.Clone(), []T{1}), nil
	}

	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("axis %d out of range for shape %v", axis, shape)
	}

	// Float32 specialized 2-pass kernel for axis=-1 (contiguous reduction)
	if ft, ok := any(t).(*tensor.Dense[float32]); ok && axis == ndim-1 {
		out := softmaxF32LastAxis(ft)
		return any(out).(*tensor.Dense[T]), nil
	}

	// Generic path
	data := make([]T, t.Len())
	copy(data, t.Data())

	axisSize := shape[axis]
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	innerSize := 1
	for i := axis + 1; i < ndim; i++ {
		innerSize *= shape[i]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			base := outer*axisSize*innerSize + inner
			maxVal := data[base]
			for a := 1; a < axisSize; a++ {
				if v := data[base+a*innerSize]; v > maxVal {
					maxVal = v
				}
			}
			var sum float64
			for a := 0; a < axisSize; a++ {
				idx := base + a*innerSize
				v := math.Exp(float64(data[idx] - maxVal))
				data[idx] = T(v)
				sum += v
			}
			invSum := 1.0 / sum
			for a := 0; a < axisSize; a++ {
				idx := base + a*innerSize
				data[idx] = T(float64(data[idx]) * invSum)
			}
		}
	}

	return tensor.NewDense[T](shape.Clone(), data), nil
}

// softmaxF32LastAxis is a 2-pass float32 kernel for softmax along the last axis.
// Pass 1: find max + compute exp and sum. Pass 2: normalize.
func softmaxF32LastAxis(t *tensor.Dense[float32]) *tensor.Dense[float32] {
	shape := t.Shape()
	axisSize := shape[shape.NDim()-1]
	outerSize := t.Len() / axisSize
	src := t.Data()
	out := make([]float32, t.Len())

	for o := 0; o < outerSize; o++ {
		base := o * axisSize
		row := src[base : base+axisSize]
		oRow := out[base : base+axisSize]

		// Pass 1: max + exp + sum (fused)
		maxVal := row[0]
		for i := 1; i < axisSize; i++ {
			if row[i] > maxVal {
				maxVal = row[i]
			}
		}
		var sum float64
		for i := 0; i < axisSize; i++ {
			v := math.Exp(float64(row[i] - maxVal))
			oRow[i] = float32(v)
			sum += v
		}

		// Pass 2: normalize
		invSum := 1.0 / sum
		for i := 0; i < axisSize; i++ {
			oRow[i] = float32(float64(oRow[i]) * invSum)
		}
	}

	return tensor.NewDense[float32](shape.Clone(), out)
}

func opElu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	alpha := float64(node.GetAttrFloat("alpha", 1.0))
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			if v >= 0 {
				data[i] = v
			} else {
				data[i] = float32(alpha * (math.Exp(float64(v)) - 1.0))
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			if v >= 0 {
				data[i] = v
			} else {
				data[i] = alpha * (math.Exp(v) - 1.0)
			}
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Elu: unsupported type %T", inputs[0])
	}
}

func preluF32(x, slope *tensor.Dense[float32]) *tensor.Dense[float32] {
	xData := x.Data()
	sData := slope.Data()
	out := make([]float32, len(xData))
	sLen := len(sData)
	xShape := x.Shape()

	if sLen == 1 {
		// Scalar slope
		s := sData[0]
		for i, v := range xData {
			if v >= 0 {
				out[i] = v
			} else {
				out[i] = v * s
			}
		}
	} else if xShape.NDim() >= 2 {
		// Per-channel slope: broadcast over spatial dims
		C := xShape[1]
		spatialSize := 1
		for d := 2; d < xShape.NDim(); d++ {
			spatialSize *= xShape[d]
		}
		for i, v := range xData {
			c := (i / spatialSize) % C
			if v >= 0 {
				out[i] = v
			} else {
				out[i] = v * sData[c%sLen]
			}
		}
	} else {
		// Element-wise
		for i, v := range xData {
			if v >= 0 {
				out[i] = v
			} else {
				out[i] = v * sData[i%sLen]
			}
		}
	}
	return tensor.NewDense[float32](xShape.Clone(), out)
}

// ── Swish (SiLU) ──

func opSwish(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = v * float32(1.0/(1.0+math.Exp(-float64(v))))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = v / (1.0 + math.Exp(-v))
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Swish: unsupported type %T", inputs[0])
	}
}
