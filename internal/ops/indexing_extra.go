package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Indexing ──

func opScatterElements(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", 0))
	reduction := node.GetAttrString("reduction", "none")
	indices := inputs[1].(*tensor.Dense[int64])
	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{scatterElementsDense(dt, indices, inputs[2].(*tensor.Dense[float32]), axis, reduction)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{scatterElementsDense(dt, indices, inputs[2].(*tensor.Dense[float64]), axis, reduction)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{scatterElementsDense(dt, indices, inputs[2].(*tensor.Dense[int32]), axis, reduction)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{scatterElementsDense(dt, indices, inputs[2].(*tensor.Dense[int64]), axis, reduction)}, nil
	case *tensor.Dense[uint8]:
		return []tensor.Tensor{scatterElementsDense(dt, indices, inputs[2].(*tensor.Dense[uint8]), axis, reduction)}, nil
	case *tensor.Dense[int8]:
		return []tensor.Tensor{scatterElementsDense(dt, indices, inputs[2].(*tensor.Dense[int8]), axis, reduction)}, nil
	default:
		return nil, fmt.Errorf("ScatterElements: unsupported type %T", inputs[0])
	}
}

func scatterElementsDense[T tensor.Numeric](data *tensor.Dense[T], indices *tensor.Dense[int64], updates *tensor.Dense[T], axis int, reduction string) *tensor.Dense[T] {
	shape := data.Shape(); ndim := shape.NDim()
	if axis < 0 { axis += ndim }
	out := make([]T, data.Len())
	copy(out, data.Data())
	idxData := indices.Data(); updData := updates.Data()
	idxShape := indices.Shape()
	strides := tensor.Strides(shape); idxStrides := tensor.Strides(idxShape)
	for i := 0; i < len(idxData); i++ {
		outIdx := 0; rem := i
		for d := 0; d < ndim; d++ {
			coord := rem / idxStrides[d]; rem %= idxStrides[d]
			if d == axis {
				idx := int(idxData[i]); if idx < 0 { idx += shape[d] }
				outIdx += idx * strides[d]
			} else {
				outIdx += coord * strides[d]
			}
		}
		switch reduction {
		case "add":
			out[outIdx] += updData[i]
		case "mul":
			out[outIdx] *= updData[i]
		case "min":
			if updData[i] < out[outIdx] { out[outIdx] = updData[i] }
		case "max":
			if updData[i] > out[outIdx] { out[outIdx] = updData[i] }
		default: // "none"
			out[outIdx] = updData[i]
		}
	}
	return tensor.NewDense[T](shape.Clone(), out)
}

// ── Matrix ──

func opTrilu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	upper := node.GetAttrInt("upper", 1) != 0
	k := 0
	if len(inputs) > 1 && inputs[1] != nil {
		if kt, ok := inputs[1].(*tensor.Dense[int64]); ok && kt.Len() > 0 { k = int(kt.Data()[0]) }
	}
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{triluDense(x, upper, k)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{triluDense(x, upper, k)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{triluDense(x, upper, k)}, nil
	default:
		return nil, fmt.Errorf("Trilu: unsupported type %T", inputs[0])
	}
}

func triluDense[T tensor.Numeric](x *tensor.Dense[T], upper bool, k int) *tensor.Dense[T] {
	shape := x.Shape()
	ndim := shape.NDim()
	H, W := shape[ndim-2], shape[ndim-1]
	batchSize := x.Len() / (H * W)
	data := x.Data()
	out := make([]T, len(data))
	for b := 0; b < batchSize; b++ {
		base := b * H * W
		for i := 0; i < H; i++ {
			for j := 0; j < W; j++ {
				if upper {
					if j >= i+k { out[base+i*W+j] = data[base+i*W+j] }
				} else {
					if j <= i+k { out[base+i*W+j] = data[base+i*W+j] }
				}
			}
		}
	}
	return tensor.NewDense[T](shape.Clone(), out)
}

// ── EyeLike ──

func opEyeLike(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	k := int(node.GetAttrInt("k", 0))
	shape := inputs[0].Shape()
	if shape.NDim() != 2 { return nil, fmt.Errorf("EyeLike: requires 2D input") }
	rows, cols := shape[0], shape[1]
	out := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		j := i + k
		if j >= 0 && j < cols { out[i*cols+j] = 1 }
	}
	return []tensor.Tensor{tensor.NewDense[float32](shape.Clone(), out)}, nil
}

// ── Compress ──

func opCompress(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", -1))
	cond := inputs[1].(*tensor.Dense[uint8]).Data()
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := x.Data()
		shape := x.Shape()
		if axis < 0 {
			// Flatten mode
			var out []float32
			for i, c := range cond {
				if c != 0 && i < len(data) { out = append(out, data[i]) }
			}
			return []tensor.Tensor{tensor.NewDense[float32](tensor.Shape{len(out)}, out)}, nil
		}
		ndim := shape.NDim()
		if axis < 0 { axis += ndim }
		outerSize := 1; for d := 0; d < axis; d++ { outerSize *= shape[d] }
		axisSize := shape[axis]
		innerSize := 1; for d := axis + 1; d < ndim; d++ { innerSize *= shape[d] }
		// Count selected
		count := 0; for i := 0; i < axisSize && i < len(cond); i++ { if cond[i] != 0 { count++ } }
		outShape := shape.Clone(); outShape[axis] = count
		out := make([]float32, 0, outerSize*count*innerSize)
		for o := 0; o < outerSize; o++ {
			for a := 0; a < axisSize && a < len(cond); a++ {
				if cond[a] != 0 {
					base := (o*axisSize + a) * innerSize
					out = append(out, data[base:base+innerSize]...)
				}
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](outShape, out)}, nil
	default:
		return nil, fmt.Errorf("Compress: unsupported type %T", inputs[0])
	}
}

// ── Unique ──

func opUnique(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	sorted := node.GetAttrInt("sorted", 1) != 0
	_ = sorted
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := x.Data()
		seen := make(map[float32]int)
		var unique []float32
		indices := make([]int64, len(data))
		var origIdx []int64
		for i, v := range data {
			if idx, ok := seen[v]; ok {
				indices[i] = int64(idx)
			} else {
				idx := len(unique)
				seen[v] = idx
				unique = append(unique, v)
				indices[i] = int64(idx)
				origIdx = append(origIdx, int64(i))
			}
		}
		counts := make([]int64, len(unique))
		for _, idx := range indices { counts[idx]++ }
		return []tensor.Tensor{
			tensor.NewDense[float32](tensor.Shape{len(unique)}, unique),
			tensor.NewDense[int64](tensor.Shape{len(origIdx)}, origIdx),
			tensor.NewDense[int64](tensor.Shape{len(indices)}, indices),
			tensor.NewDense[int64](tensor.Shape{len(counts)}, counts),
		}, nil
	default:
		return nil, fmt.Errorf("Unique: unsupported type %T", inputs[0])
	}
}
