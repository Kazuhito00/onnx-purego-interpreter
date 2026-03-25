package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// Slice
func opSlice(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	data := inputs[0]

	// opset >= 10: starts/ends/axes/steps from inputs[1..4]
	// opset <= 9: starts/ends/axes from attributes
	var starts, ends, axes, steps []int64
	if len(inputs) > 1 && inputs[1] != nil {
		starts = toInt64Slice(inputs[1])
		ends = toInt64Slice(inputs[2])
		if len(inputs) > 3 && inputs[3] != nil {
			axes = toInt64Slice(inputs[3])
		}
		if len(inputs) > 4 && inputs[4] != nil {
			steps = toInt64Slice(inputs[4])
		}
	} else {
		starts = node.GetAttrInts("starts", nil)
		ends = node.GetAttrInts("ends", nil)
		axes = node.GetAttrInts("axes", nil)
	}

	shape := data.Shape()
	ndim := shape.NDim()

	if axes == nil {
		axes = make([]int64, len(starts))
		for i := range axes {
			axes[i] = int64(i)
		}
	}
	if steps == nil {
		steps = make([]int64, len(starts))
		for i := range steps {
			steps[i] = 1
		}
	}

	// Normalize and compute output shape
	outShape := shape.Clone()
	sliceParams := make([]struct{ start, end, step int }, ndim)
	for d := 0; d < ndim; d++ {
		sliceParams[d] = struct{ start, end, step int }{0, shape[d], 1}
	}
	for i, ax := range axes {
		d := int(ax)
		if d < 0 {
			d += ndim
		}
		rawStart := starts[i]
		rawEnd := ends[i]
		s := int(rawStart)
		e := int(rawEnd)
		st := int(steps[i])
		dim := shape[d]
		if st == 0 {
			st = 1
		}
		if st > 0 {
			if s < 0 {
				s += dim
			}
			if e < 0 {
				e += dim
			}
			if rawEnd > 2000000000 {
				e = dim
			}
			if s < 0 {
				s = 0
			}
			if s > dim {
				s = dim
			}
			if e < 0 {
				e = 0
			}
			if e > dim {
				e = dim
			}
		} else {
			if rawStart < 0 {
				s += dim
			}
			if rawEnd < 0 {
				e += dim
			}
			if rawEnd < -2000000000 {
				e = -1
			}
			if s < 0 {
				s = dim - 1
			}
			if s >= dim {
				s = dim - 1
			}
			if e < -1 {
				e = -1
			}
			if e >= dim {
				e = dim - 1
			}
		}
		n := 0
		if st > 0 {
			for idx := s; idx < e; idx += st {
				n++
			}
		} else {
			for idx := s; idx > e; idx += st {
				n++
			}
		}
		outShape[d] = n
		sliceParams[d] = struct{ start, end, step int }{s, e, st}
	}

	switch dt := data.(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{sliceDense(dt, outShape, sliceParams)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{sliceDense(dt, outShape, sliceParams)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{sliceDense(dt, outShape, sliceParams)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{sliceDense(dt, outShape, sliceParams)}, nil
	case *tensor.Dense[uint8]:
		return []tensor.Tensor{sliceDense(dt, outShape, sliceParams)}, nil
	case *tensor.Dense[int8]:
		return []tensor.Tensor{sliceDense(dt, outShape, sliceParams)}, nil
	default:
		return nil, fmt.Errorf("Slice: unsupported type %T", data)
	}
}

func sliceDense[T tensor.Numeric](t *tensor.Dense[T], outShape tensor.Shape, params []struct{ start, end, step int }) *tensor.Dense[T] {
	src := t.Data()
	srcShape := t.Shape()
	ndim := srcShape.NDim()
	size := outShape.Size()
	out := make([]T, size)
	srcStrides := tensor.Strides(srcShape)
	outStrides := tensor.Strides(outShape)

	for i := 0; i < size; i++ {
		srcIdx := 0
		rem := i
		for d := 0; d < ndim; d++ {
			coord := rem / outStrides[d]
			rem %= outStrides[d]
			srcCoord := params[d].start + coord*params[d].step
			srcIdx += srcCoord * srcStrides[d]
		}
		out[i] = src[srcIdx]
	}
	return tensor.NewDense[T](outShape, out)
}

// Pad
func opPad(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var padsData []int64
	var constVal float32

	// opset >= 11: pads from input[1]
	if len(inputs) > 1 && inputs[1] != nil {
		if pt, ok := inputs[1].(*tensor.Dense[int64]); ok {
			padsData = pt.Data()
		}
	}
	// opset <= 10: pads from attribute
	if padsData == nil {
		padsData = node.GetAttrInts("pads", nil)
	}
	if padsData == nil {
		return nil, fmt.Errorf("Pad: no pads provided")
	}

	// constant value from input[2] (opset >= 11) or attribute (opset <= 10)
	if len(inputs) > 2 && inputs[2] != nil {
		switch cv := inputs[2].(type) {
		case *tensor.Dense[float32]:
			if cv.Len() > 0 {
				constVal = cv.At(0)
			}
		case *tensor.Dense[float64]:
			if cv.Len() > 0 {
				constVal = float32(cv.At(0))
			}
		}
	} else {
		constVal = node.GetAttrFloat("value", 0)
	}

	mode := node.GetAttrString("mode", "constant")

	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{padDenseWithMode(dt, padsData, constVal, mode)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{padDenseWithMode(dt, padsData, float64(constVal), mode)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{padDenseWithMode(dt, padsData, int32(constVal), mode)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{padDenseWithMode(dt, padsData, int64(constVal), mode)}, nil
	case *tensor.Dense[uint8]:
		return []tensor.Tensor{padDenseWithMode(dt, padsData, uint8(constVal), mode)}, nil
	case *tensor.Dense[int8]:
		return []tensor.Tensor{padDenseWithMode(dt, padsData, int8(constVal), mode)}, nil
	default:
		return nil, fmt.Errorf("Pad: unsupported type %T", inputs[0])
	}
}

func padDenseWithMode[T tensor.Numeric](t *tensor.Dense[T], pads []int64, constVal T, mode string) *tensor.Dense[T] {
	shape := t.Shape()
	ndim := shape.NDim()
	pads = normalizePads(pads, ndim)
	outShape := make(tensor.Shape, ndim)
	for d := 0; d < ndim; d++ {
		outShape[d] = shape[d] + int(pads[d]) + int(pads[ndim+d])
	}
	src := t.Data()
	out := make([]T, outShape.Size())
	srcStrides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)

	if mode == "reflect" {
		// Reflect padding: mirror at boundaries
		for i := 0; i < len(out); i++ {
			srcIdx := 0
			rem := i
			for d := 0; d < ndim; d++ {
				outCoord := rem / outStrides[d]
				rem %= outStrides[d]
				// Map output coord to source coord with reflect
				srcCoord := outCoord - int(pads[d])
				dim := shape[d]
				// Reflect: if out of bounds, bounce back
				if srcCoord < 0 {
					srcCoord = -srcCoord
				}
				if srcCoord >= dim {
					srcCoord = 2*(dim-1) - srcCoord
				}
				// Clamp for safety
				if srcCoord < 0 {
					srcCoord = 0
				}
				if srcCoord >= dim {
					srcCoord = dim - 1
				}
				srcIdx += srcCoord * srcStrides[d]
			}
			out[i] = src[srcIdx]
		}
	} else if mode == "edge" {
		// Edge padding: repeat boundary values
		for i := 0; i < len(out); i++ {
			srcIdx := 0
			rem := i
			for d := 0; d < ndim; d++ {
				outCoord := rem / outStrides[d]
				rem %= outStrides[d]
				srcCoord := outCoord - int(pads[d])
				if srcCoord < 0 {
					srcCoord = 0
				}
				if srcCoord >= shape[d] {
					srcCoord = shape[d] - 1
				}
				srcIdx += srcCoord * srcStrides[d]
			}
			out[i] = src[srcIdx]
		}
	} else {
		// Constant padding
		if constVal != 0 {
			for i := range out {
				out[i] = constVal
			}
		}
		for i := 0; i < t.Len(); i++ {
			srcIdx := 0
			outIdx := 0
			rem := i
			for d := 0; d < ndim; d++ {
				coord := rem / srcStrides[d]
				rem %= srcStrides[d]
				srcIdx += coord * srcStrides[d]
				outIdx += (coord + int(pads[d])) * outStrides[d]
			}
			out[outIdx] = src[srcIdx]
		}
	}
	return tensor.NewDense[T](outShape, out)
}

func normalizePads(pads []int64, ndim int) []int64 {
	if len(pads) == 2*ndim {
		return pads
	}
	if len(pads)%2 != 0 {
		out := make([]int64, 2*ndim)
		copy(out, pads)
		return out
	}

	// Accept trailing-dimension padding as produced by some PyTorch-exported models:
	// [last_begin, last_end, prev_begin, prev_end, ...]
	out := make([]int64, 2*ndim)
	pairs := len(pads) / 2
	for i := 0; i < pairs && i < ndim; i++ {
		dstDim := ndim - 1 - i
		out[dstDim] = pads[2*i]
		out[ndim+dstDim] = pads[2*i+1]
	}
	return out
}

// Tile
func opTile(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	repeats := toInt64Slice(inputs[1])
	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{tileDense(dt, repeats)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{tileDense(dt, repeats)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{tileDense(dt, repeats)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{tileDense(dt, repeats)}, nil
	case *tensor.Dense[uint8]:
		return []tensor.Tensor{tileDense(dt, repeats)}, nil
	case *tensor.Dense[int8]:
		return []tensor.Tensor{tileDense(dt, repeats)}, nil
	default:
		return nil, fmt.Errorf("Tile: unsupported type %T", inputs[0])
	}
}

func tileDense[T tensor.Numeric](t *tensor.Dense[T], repeats []int64) *tensor.Dense[T] {
	shape := t.Shape()
	ndim := shape.NDim()
	outShape := make(tensor.Shape, ndim)
	for d := 0; d < ndim; d++ {
		outShape[d] = shape[d] * int(repeats[d])
	}
	src := t.Data()
	out := make([]T, outShape.Size())
	srcStrides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)

	for i := 0; i < len(out); i++ {
		srcIdx := 0
		rem := i
		for d := 0; d < ndim; d++ {
			coord := rem / outStrides[d]
			rem %= outStrides[d]
			srcIdx += (coord % shape[d]) * srcStrides[d]
		}
		out[i] = src[srcIdx]
	}
	return tensor.NewDense[T](outShape, out)
}

// Helper: extract int64 slice from tensor
func toInt64Slice(t tensor.Tensor) []int64 {
	switch dt := t.(type) {
	case *tensor.Dense[int64]:
		return dt.Data()
	case *tensor.Dense[int32]:
		data := make([]int64, dt.Len())
		for i, v := range dt.Data() {
			data[i] = int64(v)
		}
		return data
	default:
		return nil
	}
}
