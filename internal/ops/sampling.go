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
	if mode == "nearest" && ndim == 4 && outShape[0] == shape[0] && outShape[1] == shape[1] {
		return resizeNearestNCHW(t, outShape, coordMode, nearestMode)
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

// resizeNearestNCHW は空間次元のみの nearest resize を
// 行・列インデックステーブルの事前計算+チャネル並列で処理する
// (汎用パスの毎要素座標分解を回避。写像は resizeNearestIndex と同一)。
func resizeNearestNCHW[T tensor.Numeric](t *tensor.Dense[T], outShape tensor.Shape, coordMode, nearestMode string) *tensor.Dense[T] {
	inShape := t.Shape()
	N, C, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	outH, outW := outShape[2], outShape[3]
	src := t.Data()
	out := make([]T, outShape.Size())

	rowIdx := make([]int, outH)
	for oh := range rowIdx {
		rowIdx[oh] = clampIndex(resizeNearestIndex(oh, inH, outH, coordMode, nearestMode), inH)
	}
	colIdx := make([]int, outW)
	for ow := range colIdx {
		colIdx[ow] = clampIndex(resizeNearestIndex(ow, inW, outW, coordMode, nearestMode), inW)
	}

	workers := 1
	if N*C*outH*outW >= elementwiseParallelMin {
		workers = activeActConfig.ParallelOpsWorkers()
	}
	forEachIndexParallel(N*C, workers, func(nc int) {
		baseIn := nc * inH * inW
		baseOut := nc * outH * outW
		for oh := 0; oh < outH; oh++ {
			srcRow := src[baseIn+rowIdx[oh]*inW : baseIn+rowIdx[oh]*inW+inW]
			outRow := out[baseOut+oh*outW : baseOut+oh*outW+outW]
			for ow, ci := range colIdx {
				outRow[ow] = srcRow[ci]
			}
		}
	})
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

	// 補間係数は行・列それぞれ独立なので事前計算する(ピクセルごとの再計算を排除)
	x0s := make([]int, outW)
	x1s := make([]int, outW)
	lxs := make([]float64, outW)
	for ow := 0; ow < outW; ow++ {
		ix := resizeInputCoord(ow, inW, outW, coordMode)
		x0 := int(math.Floor(ix))
		lxs[ow] = ix - float64(x0)
		x0s[ow] = clampIndex(x0, inW)
		x1s[ow] = clampIndex(x0+1, inW)
	}
	y0s := make([]int, outH)
	y1s := make([]int, outH)
	lys := make([]float64, outH)
	for oh := 0; oh < outH; oh++ {
		iy := resizeInputCoord(oh, inH, outH, coordMode)
		y0 := int(math.Floor(iy))
		lys[oh] = iy - float64(y0)
		y0s[oh] = clampIndex(y0, inH)
		y1s[oh] = clampIndex(y0+1, inH)
	}

	workers := 1
	if N*C*outH*outW >= elementwiseParallelMin {
		workers = activeActConfig.ParallelOpsWorkers()
	}
	forEachIndexParallel(N*C, workers, func(nc int) {
		baseIn := nc * inH * inW
		baseOut := nc * outH * outW
		for oh := 0; oh < outH; oh++ {
			ly := lys[oh]
			hy := 1.0 - ly
			row0 := baseIn + y0s[oh]*inW
			row1 := baseIn + y1s[oh]*inW
			outRow := baseOut + oh*outW
			for ow := 0; ow < outW; ow++ {
				lx := lxs[ow]
				hx := 1.0 - lx
				v00 := float64(src[row0+x0s[ow]])
				v01 := float64(src[row0+x1s[ow]])
				v10 := float64(src[row1+x0s[ow]])
				v11 := float64(src[row1+x1s[ow]])
				out[outRow+ow] = T(v00*hy*hx + v01*hy*lx + v10*ly*hx + v11*ly*lx)
			}
		}
	})
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

// gridSamplePad applies the padding_mode coordinate transform (border clamp /
// reflection) to a denormalized coordinate. "zeros" is left untouched here —
// it is instead handled by bounds-checking at sample time so out-of-range
// reads become 0.
func gridSamplePad(v float64, size int, paddingMode string, alignCorners bool) float64 {
	switch paddingMode {
	case "border":
		return clampFloat64(v, 0, float64(size-1))
	case "reflection":
		return clampFloat64(gridSampleReflect(v, size, alignCorners), 0, float64(size-1))
	default: // "zeros"
		return v
	}
}

func clampFloat64(v, lo, hi float64) float64 {
	if hi < lo {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampInt(v, size int) int {
	if v < 0 {
		return 0
	}
	if v > size-1 {
		return size - 1
	}
	return v
}

// gridSampleReflect mirrors PyTorch/ONNX GridSample's reflection padding:
// reflects v back and forth over [twiceLow/2, twiceHigh/2] until it lands in range.
func gridSampleReflect(v float64, size int, alignCorners bool) float64 {
	if size == 1 {
		return 0
	}
	var twiceLow, twiceHigh float64
	if alignCorners {
		twiceLow, twiceHigh = 0, float64(2*(size-1))
	} else {
		twiceLow, twiceHigh = -1, float64(2*size-1)
	}
	lo := twiceLow / 2
	span := (twiceHigh - twiceLow) / 2
	v = math.Abs(v - lo)
	extra := math.Mod(v, span)
	flips := int(math.Floor(v / span))
	if flips%2 == 0 {
		return extra + lo
	}
	return span - extra + lo
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
				ix = gridSamplePad(ix, Win, paddingMode, alignCorners)
				iy = gridSamplePad(iy, Hin, paddingMode, alignCorners)

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
							if paddingMode != "zeros" {
								// Coordinates were already clamped/reflected into range above;
								// the floor+1 corner can still step one pixel past the border.
								return xData[base+clampInt(y, Hin)*Win+clampInt(x, Win)]
							}
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
