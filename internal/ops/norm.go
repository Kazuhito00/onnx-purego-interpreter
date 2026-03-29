package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// opLRN implements Local Response Normalization (across channels).
func opLRN(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	alpha := float64(node.GetAttrFloat("alpha", 0.0001))
	beta := float64(node.GetAttrFloat("beta", 0.75))
	bias := float64(node.GetAttrFloat("bias", 1.0))
	size := int(node.GetAttrInt("size", 5))

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{lrnF32(x, size, alpha, beta, bias)}, nil
	default:
		return nil, fmt.Errorf("LRN: unsupported type %T", inputs[0])
	}
}

func lrnF32(x *tensor.Dense[float32], size int, alpha, beta, bias float64) *tensor.Dense[float32] {
	s := x.Shape() // [N, C, H, W]
	N, C, H, W := s[0], s[1], s[2], s[3]
	HW := H * W
	data := x.Data()
	out := make([]float32, len(data))
	half := size / 2

	for n := 0; n < N; n++ {
		nBase := n * C * HW
		for c := 0; c < C; c++ {
			cStart := c - half
			if cStart < 0 {
				cStart = 0
			}
			cEnd := c + half + 1
			if cEnd > C {
				cEnd = C
			}
			for hw := 0; hw < HW; hw++ {
				idx := nBase + c*HW + hw
				var sqSum float64
				for cc := cStart; cc < cEnd; cc++ {
					v := float64(data[nBase+cc*HW+hw])
					sqSum += v * v
				}
				norm := math.Pow(bias+alpha/float64(size)*sqSum, beta)
				out[idx] = float32(float64(data[idx]) / norm)
			}
		}
	}
	return tensor.NewDense[float32](s.Clone(), out)
}

func opBatchNormalization(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// inputs: X, scale, B, input_mean, input_var
	epsilon := float64(node.GetAttrFloat("epsilon", 1e-5))

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		scale := inputs[1].(*tensor.Dense[float32])
		bias := inputs[2].(*tensor.Dense[float32])
		mean := inputs[3].(*tensor.Dense[float32])
		variance := inputs[4].(*tensor.Dense[float32])
		out := batchNorm(x, scale, bias, mean, variance, epsilon)
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		scale := inputs[1].(*tensor.Dense[float64])
		bias := inputs[2].(*tensor.Dense[float64])
		mean := inputs[3].(*tensor.Dense[float64])
		variance := inputs[4].(*tensor.Dense[float64])
		out := batchNorm(x, scale, bias, mean, variance, epsilon)
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("BatchNormalization: unsupported type %T", inputs[0])
	}
}

func batchNorm[T tensor.Numeric](x, scale, bias, mean, variance *tensor.Dense[T], epsilon float64) *tensor.Dense[T] {
	xShape := x.Shape() // [N, C, ...spatial...]
	N, C := xShape[0], xShape[1]
	spatialSize := 1
	for i := 2; i < xShape.NDim(); i++ {
		spatialSize *= xShape[i]
	}

	outData := make([]T, x.Len())
	xData := x.Data()
	scaleData := scale.Data()
	biasData := bias.Data()
	meanData := mean.Data()
	varData := variance.Data()

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			s := float64(scaleData[c])
			b := float64(biasData[c])
			m := float64(meanData[c])
			v := float64(varData[c])
			invStd := s / math.Sqrt(v+epsilon)

			base := n*C*spatialSize + c*spatialSize
			xSlice := xData[base : base+spatialSize] // BCE
			oSlice := outData[base : base+spatialSize] // BCE
			for i := 0; i < spatialSize; i++ {
				oSlice[i] = T((float64(xSlice[i])-m)*invStd + b)
			}
		}
	}

	return tensor.NewDense[T](xShape.Clone(), outData)
}
// LayerNormalization
func opLayerNormalization(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", -1))
	epsilon := float64(node.GetAttrFloat("epsilon", 1e-5))

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		scale := inputs[1].(*tensor.Dense[float32])
		var bias *tensor.Dense[float32]
		if len(inputs) > 2 && inputs[2] != nil {
			bias = inputs[2].(*tensor.Dense[float32])
		}
		return []tensor.Tensor{layerNormF32(x, scale, bias, axis, epsilon)}, nil
	default:
		return nil, fmt.Errorf("LayerNormalization: unsupported type %T", inputs[0])
	}
}

func layerNormF32(x, scale, bias *tensor.Dense[float32], axis int, epsilon float64) *tensor.Dense[float32] {
	shape := x.Shape()
	ndim := shape.NDim()
	if axis < 0 { axis += ndim }

	outerSize := 1
	for d := 0; d < axis; d++ { outerSize *= shape[d] }
	innerSize := 1
	for d := axis; d < ndim; d++ { innerSize *= shape[d] }

	xData := x.Data()
	out := make([]float32, len(xData))
	scaleData := scale.Data()
	var biasData []float32
	if bias != nil { biasData = bias.Data() }

	for o := 0; o < outerSize; o++ {
		base := o * innerSize
		xSlice := xData[base : base+innerSize] // BCE
		oSlice := out[base : base+innerSize]    // BCE
		// Compute mean
		var sum float64
		for i := 0; i < innerSize; i++ { sum += float64(xSlice[i]) }
		mean := sum / float64(innerSize)
		// Compute variance
		var varSum float64
		for i := 0; i < innerSize; i++ {
			d := float64(xSlice[i]) - mean
			varSum += d * d
		}
		variance := varSum / float64(innerSize)
		invStd := 1.0 / math.Sqrt(variance+epsilon)

		for i := 0; i < innerSize; i++ {
			v := (float64(xSlice[i]) - mean) * invStd
			v *= float64(scaleData[i])
			if biasData != nil { v += float64(biasData[i]) }
			oSlice[i] = float32(v)
		}
	}
	return tensor.NewDense[float32](shape.Clone(), out)
}

// InstanceNormalization: Y = scale * (X - mean) / sqrt(var + epsilon) + bias
// Per-instance, per-channel normalization.
func opInstanceNormalization(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	epsilon := float64(node.GetAttrFloat("epsilon", 1e-5))

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		scale := inputs[1].(*tensor.Dense[float32])
		bias := inputs[2].(*tensor.Dense[float32])
		return []tensor.Tensor{instanceNormF32(x, scale, bias, epsilon)}, nil
	default:
		return nil, fmt.Errorf("InstanceNormalization: unsupported type %T", inputs[0])
	}
}

func instanceNormF32(x, scale, bias *tensor.Dense[float32], epsilon float64) *tensor.Dense[float32] {
	s := x.Shape() // [N, C, H, W, ...]
	N, C := s[0], s[1]
	spatialSize := 1
	for i := 2; i < s.NDim(); i++ {
		spatialSize *= s[i]
	}

	xData := x.Data()
	scaleData := scale.Data()
	biasData := bias.Data()
	out := make([]float32, len(xData))

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			base := n*C*spatialSize + c*spatialSize
			xSlice := xData[base : base+spatialSize] // BCE
			oSlice := out[base : base+spatialSize]    // BCE
			// Compute mean
			var sum float64
			for i := 0; i < spatialSize; i++ {
				sum += float64(xSlice[i])
			}
			mean := sum / float64(spatialSize)
			// Compute variance
			var varSum float64
			for i := 0; i < spatialSize; i++ {
				d := float64(xSlice[i]) - mean
				varSum += d * d
			}
			variance := varSum / float64(spatialSize)
			invStd := 1.0 / math.Sqrt(variance+epsilon)

			sc := float64(scaleData[c])
			b := float64(biasData[c])
			for i := 0; i < spatialSize; i++ {
				oSlice[i] = float32((float64(xSlice[i])-mean)*invStd*sc + b)
			}
		}
	}
	return tensor.NewDense[float32](s.Clone(), out)
}
