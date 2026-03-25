package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── MaxRoiPool ──

func opMaxRoiPool(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	pooledShape := node.GetAttrInts("pooled_shape", []int64{1, 1})
	pH, pW := int(pooledShape[0]), int(pooledShape[1])
	spatialScale := float64(node.GetAttrFloat("spatial_scale", 1.0))

	x := inputs[0].(*tensor.Dense[float32])
	rois := inputs[1].(*tensor.Dense[float32])
	xShape := x.Shape()
	C, H, W := xShape[1], xShape[2], xShape[3]
	numRois := rois.Shape()[0]
	xData := x.Data()
	roisData := rois.Data()

	outShape := tensor.Shape{numRois, C, pH, pW}
	out := make([]float32, outShape.Size())
	for r := 0; r < numRois; r++ {
		bIdx := int(roisData[r*5])
		x1 := int(math.Round(float64(roisData[r*5+1]) * spatialScale))
		y1 := int(math.Round(float64(roisData[r*5+2]) * spatialScale))
		x2 := int(math.Round(float64(roisData[r*5+3]) * spatialScale))
		y2 := int(math.Round(float64(roisData[r*5+4]) * spatialScale))
		roiH := y2 - y1; if roiH <= 0 { roiH = 1 }
		roiW := x2 - x1; if roiW <= 0 { roiW = 1 }
		for c := 0; c < C; c++ {
			cOff := (bIdx*C + c) * H * W
			for oh := 0; oh < pH; oh++ {
				hStart := y1 + oh*roiH/pH
				hEnd := y1 + (oh+1)*roiH/pH
				if hEnd <= hStart { hEnd = hStart + 1 }
				for ow := 0; ow < pW; ow++ {
					wStart := x1 + ow*roiW/pW
					wEnd := x1 + (ow+1)*roiW/pW
					if wEnd <= wStart { wEnd = wStart + 1 }
					maxVal := float32(-math.MaxFloat32)
					for h := hStart; h < hEnd; h++ {
						for w := wStart; w < wEnd; w++ {
							if h >= 0 && h < H && w >= 0 && w < W {
								v := xData[cOff+h*W+w]
								if v > maxVal { maxVal = v }
							}
						}
					}
					if maxVal == float32(-math.MaxFloat32) { maxVal = 0 }
					out[((r*C+c)*pH+oh)*pW+ow] = maxVal
				}
			}
		}
	}
	return []tensor.Tensor{tensor.NewDense[float32](outShape, out)}, nil
}

// ── DynamicQuantizeLinear ──

func opDynamicQuantizeLinear(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0].(*tensor.Dense[float32])
	data := x.Data()
	minVal, maxVal := float32(0), float32(0)
	for _, v := range data {
		if v < minVal { minVal = v }
		if v > maxVal { maxVal = v }
	}
	scale := (maxVal - minVal) / 255.0
	if scale == 0 { scale = 1 }
	zp := uint8(math.Round(float64(-minVal / scale)))
	out := make([]uint8, len(data))
	for i, v := range data {
		q := math.Round(float64(v)/float64(scale)) + float64(zp)
		if q < 0 { q = 0 }; if q > 255 { q = 255 }
		out[i] = uint8(q)
	}
	return []tensor.Tensor{
		tensor.NewDense[uint8](x.Shape().Clone(), out),
		tensor.NewDense[float32](tensor.Shape{}, []float32{scale}),
		tensor.NewDense[uint8](tensor.Shape{}, []uint8{zp}),
	}, nil
}

// ── Det ──

func opDet(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		shape := x.Shape()
		ndim := shape.NDim()
		n := shape[ndim-1]
		if shape[ndim-2] != n { return nil, fmt.Errorf("Det: not square") }
		batchSize := 1
		for d := 0; d < ndim-2; d++ { batchSize *= shape[d] }
		data := x.Data()
		outData := make([]float32, batchSize)
		for b := 0; b < batchSize; b++ {
			// LU decomposition
			mat := make([]float64, n*n)
			off := b * n * n
			for i := range mat { mat[i] = float64(data[off+i]) }
			det := 1.0
			for i := 0; i < n; i++ {
				// Partial pivoting
				maxRow := i; maxVal := math.Abs(mat[i*n+i])
				for k := i + 1; k < n; k++ {
					if v := math.Abs(mat[k*n+i]); v > maxVal { maxVal = v; maxRow = k }
				}
				if maxRow != i {
					for j := 0; j < n; j++ { mat[i*n+j], mat[maxRow*n+j] = mat[maxRow*n+j], mat[i*n+j] }
					det = -det
				}
				if mat[i*n+i] == 0 { det = 0; break }
				det *= mat[i*n+i]
				for k := i + 1; k < n; k++ {
					f := mat[k*n+i] / mat[i*n+i]
					for j := i + 1; j < n; j++ { mat[k*n+j] -= f * mat[i*n+j] }
				}
			}
			outData[b] = float32(det)
		}
		outShape := make(tensor.Shape, ndim-2)
		for d := 0; d < ndim-2; d++ { outShape[d] = shape[d] }
		if len(outShape) == 0 { outShape = tensor.Shape{} }
		return []tensor.Tensor{tensor.NewDense[float32](outShape, outData)}, nil
	default:
		return nil, fmt.Errorf("Det: unsupported type %T", inputs[0])
	}
}
