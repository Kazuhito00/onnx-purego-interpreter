package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Normalization extras ──

func opGroupNormalization(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	epsilon := float64(node.GetAttrFloat("epsilon", 1e-5))
	numGroups := int(node.GetAttrInt("num_groups", 1))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		scale := inputs[1].(*tensor.Dense[float32])
		bias := inputs[2].(*tensor.Dense[float32])
		return []tensor.Tensor{groupNormF32(x, scale, bias, numGroups, epsilon)}, nil
	default:
		return nil, fmt.Errorf("GroupNormalization: unsupported type %T", inputs[0])
	}
}

func groupNormF32(x, scale, bias *tensor.Dense[float32], numGroups int, epsilon float64) *tensor.Dense[float32] {
	s := x.Shape(); N, C := s[0], s[1]
	spatialSize := 1; for i := 2; i < s.NDim(); i++ { spatialSize *= s[i] }
	chPerGroup := C / numGroups
	groupSize := chPerGroup * spatialSize
	xData := x.Data(); scaleData := scale.Data(); biasData := bias.Data()
	out := make([]float32, len(xData))
	for n := 0; n < N; n++ {
		for g := 0; g < numGroups; g++ {
			base := n*C*spatialSize + g*groupSize
			var sum float64
			for i := 0; i < groupSize; i++ { sum += float64(xData[base+i]) }
			mean := sum / float64(groupSize)
			var varSum float64
			for i := 0; i < groupSize; i++ { d := float64(xData[base+i]) - mean; varSum += d * d }
			invStd := 1.0 / math.Sqrt(varSum/float64(groupSize)+epsilon)
			for ch := 0; ch < chPerGroup; ch++ {
				c := g*chPerGroup + ch
				// scale/bias may be [num_groups] or [C]
				si := c; if si >= len(scaleData) { si = g }
				bi2 := c; if bi2 >= len(biasData) { bi2 = g }
				sc := float64(scaleData[si]); bi := float64(biasData[bi2])
				off := base + ch*spatialSize
				for i := 0; i < spatialSize; i++ {
					out[off+i] = float32((float64(xData[off+i])-mean)*invStd*sc + bi)
				}
			}
		}
	}
	return tensor.NewDense[float32](s.Clone(), out)
}

func opRMSNormalization(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	epsilon := float64(node.GetAttrFloat("epsilon", 1e-5))
	axis := int(node.GetAttrInt("axis", -1))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		scale := inputs[1].(*tensor.Dense[float32])
		return []tensor.Tensor{rmsNormF32(x, scale, axis, epsilon)}, nil
	default:
		return nil, fmt.Errorf("RMSNormalization: unsupported type %T", inputs[0])
	}
}

func rmsNormF32(x, scale *tensor.Dense[float32], axis int, epsilon float64) *tensor.Dense[float32] {
	shape := x.Shape(); ndim := shape.NDim()
	if axis < 0 { axis += ndim }
	outerSize := 1; for d := 0; d < axis; d++ { outerSize *= shape[d] }
	innerSize := 1; for d := axis; d < ndim; d++ { innerSize *= shape[d] }
	xData := x.Data(); scaleData := scale.Data()
	out := make([]float32, len(xData))
	for o := 0; o < outerSize; o++ {
		base := o * innerSize
		var sumSq float64
		for i := 0; i < innerSize; i++ { v := float64(xData[base+i]); sumSq += v * v }
		rms := math.Sqrt(sumSq/float64(innerSize) + epsilon)
		invRms := 1.0 / rms
		for i := 0; i < innerSize; i++ {
			out[base+i] = float32(float64(xData[base+i]) * invRms * float64(scaleData[i]))
		}
	}
	return tensor.NewDense[float32](shape.Clone(), out)
}

// ── LpNormalization ──

func opLpNormalization(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", -1))
	p := int(node.GetAttrInt("p", 2))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		shape := x.Shape(); ndim := shape.NDim()
		if axis < 0 { axis += ndim }
		outerSize := 1; for d := 0; d < axis; d++ { outerSize *= shape[d] }
		axisSize := shape[axis]
		innerSize := 1; for d := axis + 1; d < ndim; d++ { innerSize *= shape[d] }
		data := x.Data()
		out := make([]float32, len(data))
		copy(out, data)
		for o := 0; o < outerSize; o++ {
			for i := 0; i < innerSize; i++ {
				var norm float64
				for a := 0; a < axisSize; a++ {
					idx := (o*axisSize+a)*innerSize + i
					v := math.Abs(float64(data[idx]))
					if p == 1 { norm += v } else { norm += v * v }
				}
				if p != 1 { norm = math.Sqrt(norm) }
				if norm == 0 { norm = 1 }
				for a := 0; a < axisSize; a++ {
					idx := (o*axisSize+a)*innerSize + i
					out[idx] = float32(float64(data[idx]) / norm)
				}
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](shape.Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("LpNormalization: unsupported type %T", inputs[0])
	}
}

// ── MeanVarianceNormalization ──

func opMeanVarianceNormalization(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axes := node.GetAttrInts("axes", []int64{0, 2, 3})
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		// Compute mean, then variance, then normalize
		meanT := reduceSumDense(x, intsToSlice(axes), true, false)
		shape := x.Shape()
		totalReduced := 1
		for _, a := range axes { totalReduced *= shape[int(a)] }
		md := meanT.Data()
		for i := range md { md[i] /= float32(totalReduced) }

		data := x.Data()
		out := make([]float32, len(data))
		// Compute variance
		varT := make([]float32, len(md))
		strides := tensor.Strides(shape)
		meanShape := meanT.Shape()
		meanStrides := tensor.Strides(meanShape)
		for i := 0; i < len(data); i++ {
			// Find mean index
			mIdx := 0; rem := i
			for d := 0; d < shape.NDim(); d++ {
				coord := rem / strides[d]; rem %= strides[d]
				if meanShape[d] == 1 { continue }
				mIdx += coord * meanStrides[d]
			}
			diff := data[i] - md[mIdx]
			varT[mIdx] += diff * diff
		}
		for i := range varT { varT[i] = float32(math.Sqrt(float64(varT[i]) / float64(totalReduced))) }
		// Normalize
		for i := 0; i < len(data); i++ {
			mIdx := 0; rem := i
			for d := 0; d < shape.NDim(); d++ {
				coord := rem / strides[d]; rem %= strides[d]
				if meanShape[d] == 1 { continue }
				mIdx += coord * meanStrides[d]
			}
			std := varT[mIdx]; if std == 0 { std = 1 }
			out[i] = (data[i] - md[mIdx]) / std
		}
		return []tensor.Tensor{tensor.NewDense[float32](shape.Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("MeanVarianceNormalization: unsupported type %T", inputs[0])
	}
}

func intsToSlice(a []int64) []int { s := make([]int, len(a)); for i, v := range a { s[i] = int(v) }; return s }
