package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// GatherElements
func opGatherElements(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", 0))
	idxT := inputs[1].(*tensor.Dense[int64])
	idxData := idxT.Data()
	idxShape := idxT.Shape()

	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{gatherElementsDense(dt, idxData, idxShape, axis)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{gatherElementsDense(dt, idxData, idxShape, axis)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{gatherElementsDense(dt, idxData, idxShape, axis)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{gatherElementsDense(dt, idxData, idxShape, axis)}, nil
	case *tensor.Dense[uint8]:
		return []tensor.Tensor{gatherElementsDense(dt, idxData, idxShape, axis)}, nil
	case *tensor.Dense[int8]:
		return []tensor.Tensor{gatherElementsDense(dt, idxData, idxShape, axis)}, nil
	default:
		return nil, fmt.Errorf("GatherElements: unsupported type %T", inputs[0])
	}
}

func gatherElementsDense[T tensor.Numeric](data *tensor.Dense[T], indices []int64, idxShape tensor.Shape, axis int) *tensor.Dense[T] {
	srcShape := data.Shape()
	ndim := srcShape.NDim()
	if axis < 0 {
		axis += ndim
	}
	src := data.Data()
	out := make([]T, idxShape.Size())
	srcStrides := tensor.Strides(srcShape)
	idxStrides := tensor.Strides(idxShape)

	for i := 0; i < len(out); i++ {
		srcIdx := 0
		rem := i
		for d := 0; d < ndim; d++ {
			coord := rem / idxStrides[d]
			rem %= idxStrides[d]
			if d == axis {
				idx := int(indices[i])
				if idx < 0 {
					idx += srcShape[d]
				}
				srcIdx += idx * srcStrides[d]
			} else {
				srcIdx += coord * srcStrides[d]
			}
		}
		out[i] = src[srcIdx]
	}
	return tensor.NewDense[T](idxShape.Clone(), out)
}

// Resize (nearest + linear)
func opResize(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	mode := node.GetAttrString("mode", "nearest")
	// opset 10: no coordinate_transformation_mode (implicit asymmetric)
	// opset 11+: coordinate_transformation_mode attribute
	defaultCoordMode := "half_pixel"
	if node.OpsetVersion > 0 && node.OpsetVersion <= 10 {
		defaultCoordMode = "asymmetric"
	}
	coordMode := node.GetAttrString("coordinate_transformation_mode", defaultCoordMode)
	nearestMode := node.GetAttrString("nearest_mode", "round_prefer_floor")
	cubicCoeffA := float64(node.GetAttrFloat("cubic_coeff_a", -0.75))

	// inputs: X, roi (unused), scales, sizes
	var scales []float32
	var sizes []int64
	if len(inputs) > 2 && inputs[2] != nil {
		switch sc := inputs[2].(type) {
		case *tensor.Dense[float32]:
			if sc.Len() > 0 {
				scales = sc.Data()
			}
		case *tensor.Dense[float64]:
			if sc.Len() > 0 {
				for _, v := range sc.Data() {
					scales = append(scales, float32(v))
				}
			}
		case *tensor.Dense[int64]:
			if sc.Len() > 0 {
				sizes = sc.Data()
			}
		case *tensor.Dense[int32]:
			if sc.Len() > 0 {
				for _, v := range sc.Data() {
					sizes = append(sizes, int64(v))
				}
			}
		}
	}
	if len(inputs) > 3 && inputs[3] != nil {
		if sz, ok := inputs[3].(*tensor.Dense[int64]); ok && sz.Len() > 0 {
			sizes = sz.Data()
		}
	}

	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{resizeDense(dt, scales, sizes, mode, coordMode, nearestMode, cubicCoeffA)}, nil
	default:
		return nil, fmt.Errorf("Resize: unsupported type %T", inputs[0])
	}
}

func resizeDense[T tensor.Numeric](t *tensor.Dense[T], scales []float32, sizes []int64, mode, coordMode, nearestMode string, cubicCoeffA float64) *tensor.Dense[T] {
	shape := t.Shape()
	ndim := shape.NDim()
	outShape := make(tensor.Shape, ndim)

	if len(sizes) == ndim {
		for d := 0; d < ndim; d++ {
			outShape[d] = int(sizes[d])
		}
	} else if len(scales) == ndim {
		for d := 0; d < ndim; d++ {
			outShape[d] = int(float32(shape[d]) * scales[d])
		}
	} else {
		copy(outShape, shape)
	}

	if mode == "linear" && ndim == 4 {
		return resizeLinearNCHW(t, outShape, coordMode)
	}
	if mode == "cubic" && ndim == 4 {
		return resizeCubicNCHW(t, outShape, coordMode, cubicCoeffA)
	}

	src := t.Data()
	out := make([]T, outShape.Size())
	srcStrides := tensor.Strides(shape)
	outStrides := tensor.Strides(outShape)

	for i := 0; i < len(out); i++ {
		srcIdx := 0
		rem := i
		for d := 0; d < ndim; d++ {
			outCoord := rem / outStrides[d]
			rem %= outStrides[d]
			srcCoord := resizeNearestIndex(outCoord, shape[d], outShape[d], coordMode, nearestMode)
			if srcCoord >= shape[d] {
				srcCoord = shape[d] - 1
			}
			srcIdx += srcCoord * srcStrides[d]
		}
		out[i] = src[srcIdx]
	}
	return tensor.NewDense[T](outShape, out)
}

func resizeCubicNCHW[T tensor.Numeric](t *tensor.Dense[T], outShape tensor.Shape, coordMode string, cubicCoeffA float64) *tensor.Dense[T] {
	inShape := t.Shape()
	N, C, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	outH, outW := outShape[2], outShape[3]
	src := t.Data()
	out := make([]T, outShape.Size())

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			baseIn := (n*C + c) * inH * inW
			baseOut := (n*C + c) * outH * outW
			for oh := 0; oh < outH; oh++ {
				iy := resizeInputCoord(oh, inH, outH, coordMode)
				yInt := int(math.Floor(iy))
				for ow := 0; ow < outW; ow++ {
					ix := resizeInputCoord(ow, inW, outW, coordMode)
					xInt := int(math.Floor(ix))
					sum := 0.0
					for ky := -1; ky <= 2; ky++ {
						sy := clampIndex(yInt+ky, inH)
						wy := cubicWeight(iy-float64(yInt+ky), cubicCoeffA)
						rowBase := baseIn + sy*inW
						for kx := -1; kx <= 2; kx++ {
							sx := clampIndex(xInt+kx, inW)
							wx := cubicWeight(ix-float64(xInt+kx), cubicCoeffA)
							sum += float64(src[rowBase+sx]) * wy * wx
						}
					}
					out[baseOut+oh*outW+ow] = T(sum)
				}
			}
		}
	}
	return tensor.NewDense[T](outShape, out)
}

func cubicWeight(x, a float64) float64 {
	ax := math.Abs(x)
	if ax <= 1 {
		return (a+2)*ax*ax*ax - (a+3)*ax*ax + 1
	}
	if ax < 2 {
		return a*ax*ax*ax - 5*a*ax*ax + 8*a*ax - 4*a
	}
	return 0
}

func resizeNearestIndex(outCoord, inSize, outSize int, coordMode, nearestMode string) int {
	coord := resizeInputCoord(outCoord, inSize, outSize, coordMode)
	switch nearestMode {
	case "floor":
		return int(math.Floor(coord))
	case "ceil":
		return int(math.Ceil(coord))
	default:
		return int(math.Floor(coord + 0.5))
	}
}

func resizeInputCoord(outCoord, inSize, outSize int, coordMode string) float64 {
	if outSize <= 0 {
		return 0
	}
	switch coordMode {
	case "asymmetric":
		return float64(outCoord) * float64(inSize) / float64(outSize)
	case "align_corners":
		if outSize == 1 {
			return 0
		}
		return float64(outCoord) * float64(inSize-1) / float64(outSize-1)
	case "half_pixel":
		return (float64(outCoord)+0.5)*float64(inSize)/float64(outSize) - 0.5
	default:
		return float64(outCoord) * float64(inSize) / float64(outSize)
	}
}

func resizeLinearNCHW[T tensor.Numeric](t *tensor.Dense[T], outShape tensor.Shape, coordMode string) *tensor.Dense[T] {
	inShape := t.Shape()
	N, C, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	outH, outW := outShape[2], outShape[3]
	src := t.Data()
	out := make([]T, outShape.Size())

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			baseIn := (n*C + c) * inH * inW
			baseOut := (n*C + c) * outH * outW
			for oh := 0; oh < outH; oh++ {
				iy := resizeInputCoord(oh, inH, outH, coordMode)
				y0 := int(math.Floor(iy))
				y1 := y0 + 1
				ly := iy - float64(y0)
				hy := 1.0 - ly
				y0 = clampIndex(y0, inH)
				y1 = clampIndex(y1, inH)
				for ow := 0; ow < outW; ow++ {
					ix := resizeInputCoord(ow, inW, outW, coordMode)
					x0 := int(math.Floor(ix))
					x1 := x0 + 1
					lx := ix - float64(x0)
					hx := 1.0 - lx
					x0 = clampIndex(x0, inW)
					x1 = clampIndex(x1, inW)

					v00 := float64(src[baseIn+y0*inW+x0])
					v01 := float64(src[baseIn+y0*inW+x1])
					v10 := float64(src[baseIn+y1*inW+x0])
					v11 := float64(src[baseIn+y1*inW+x1])
					v := v00*hy*hx + v01*hy*lx + v10*ly*hx + v11*ly*lx
					out[baseOut+oh*outW+ow] = T(v)
				}
			}
		}
	}
	return tensor.NewDense[T](outShape, out)
}

func clampIndex(v, size int) int {
	if v < 0 {
		return 0
	}
	if v >= size {
		return size - 1
	}
	return v
}

// GridSample (bilinear, 4D only)
func opGridSample(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	mode := node.GetAttrString("mode", "bilinear")
	paddingMode := node.GetAttrString("padding_mode", "zeros")
	alignCorners := node.GetAttrInt("align_corners", 0) != 0

	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		grid := inputs[1].(*tensor.Dense[float32])
		return []tensor.Tensor{gridSampleF32(x, grid, mode, paddingMode, alignCorners)}, nil
	default:
		return nil, fmt.Errorf("GridSample: unsupported type %T", inputs[0])
	}
}

func gridSampleF32(x, grid *tensor.Dense[float32], mode, paddingMode string, alignCorners bool) *tensor.Dense[float32] {
	xs := x.Shape()    // [N, C, Hin, Win]
	gs := grid.Shape() // [N, Hout, Wout, 2]
	N, C, Hin, Win := xs[0], xs[1], xs[2], xs[3]
	Hout, Wout := gs[1], gs[2]

	xData := x.Data()
	gData := grid.Data()
	out := make([]float32, N*C*Hout*Wout)

	for n := 0; n < N; n++ {
		for h := 0; h < Hout; h++ {
			for w := 0; w < Wout; w++ {
				gIdx := n*Hout*Wout*2 + h*Wout*2 + w*2
				gx := float64(gData[gIdx])
				gy := float64(gData[gIdx+1])

				// Denormalize grid coordinates
				var ix, iy float64
				if alignCorners {
					ix = (gx + 1) / 2 * float64(Win-1)
					iy = (gy + 1) / 2 * float64(Hin-1)
				} else {
					ix = ((gx+1)*float64(Win) - 1) / 2
					iy = ((gy+1)*float64(Hin) - 1) / 2
				}

				for c := 0; c < C; c++ {
					var val float32
					if mode == "nearest" {
						rx := int(math.Round(ix))
						ry := int(math.Round(iy))
						if rx >= 0 && rx < Win && ry >= 0 && ry < Hin {
							val = xData[n*C*Hin*Win+c*Hin*Win+ry*Win+rx]
						}
					} else { // bilinear
						x0 := int(math.Floor(ix))
						y0 := int(math.Floor(iy))
						x1 := x0 + 1
						y1 := y0 + 1
						wa := float32((float64(x1) - ix) * (float64(y1) - iy))
						wb := float32((ix - float64(x0)) * (float64(y1) - iy))
						wc := float32((float64(x1) - ix) * (iy - float64(y0)))
						wd := float32((ix - float64(x0)) * (iy - float64(y0)))
						base := n*C*Hin*Win + c*Hin*Win
						getSafe := func(y, x int) float32 {
							if y >= 0 && y < Hin && x >= 0 && x < Win {
								return xData[base+y*Win+x]
							}
							return 0
						}
						val = wa*getSafe(y0, x0) + wb*getSafe(y0, x1) + wc*getSafe(y1, x0) + wd*getSafe(y1, x1)
					}
					out[n*C*Hout*Wout+c*Hout*Wout+h*Wout+w] = val
				}
			}
		}
	}
	return tensor.NewDense[float32](tensor.Shape{N, C, Hout, Wout}, out)
}

// Upsample (opset <= 9, equivalent to Resize with scales)
func opUpsample(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	mode := node.GetAttrString("mode", "nearest")

	// scales from input[1] (opset 9) or attribute (opset 7)
	var scales []float32
	if len(inputs) > 1 && inputs[1] != nil {
		switch sc := inputs[1].(type) {
		case *tensor.Dense[float32]:
			scales = sc.Data()
		case *tensor.Dense[float64]:
			for _, v := range sc.Data() {
				scales = append(scales, float32(v))
			}
		}
	}
	if len(scales) == 0 {
		attrScales := node.GetAttrFloats("scales", nil)
		scales = attrScales
	}

	switch dt := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{resizeDense(dt, scales, nil, mode, "half_pixel", "round_prefer_floor", -0.75)}, nil
	default:
		return nil, fmt.Errorf("Upsample: unsupported type %T", inputs[0])
	}
}
