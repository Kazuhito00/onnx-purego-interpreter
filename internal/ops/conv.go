package ops

import (
	"fmt"
	"math"
	"sync"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

var float32ScratchPool sync.Pool

func getFloat32Scratch(size int) []float32 {
	if v := float32ScratchPool.Get(); v != nil {
		buf := v.([]float32)
		if cap(buf) >= size {
			return buf[:size]
		}
	}
	return make([]float32, size)
}

func putFloat32Scratch(buf []float32) {
	if buf == nil {
		return
	}
	float32ScratchPool.Put(buf[:0])
}

// activeKernelConfig is set by closure factories. nil means default (all enabled).
var activeConvConfig *KernelConfig

func makeConv(kc *KernelConfig) OpFunc {
	return func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
		activeConvConfig = kc
		return opConv(node, inputs)
	}
}
func makeFusedConv(kc *KernelConfig) OpFunc {
	return func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
		activeConvConfig = kc
		return opFusedConv(node, inputs)
	}
}
func makeConvTranspose(kc *KernelConfig) OpFunc {
	return func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
		activeConvConfig = kc
		return opConvTranspose(node, inputs)
	}
}

// opFusedConv handles Conv with fused activation (relu, clip).
func opFusedConv(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	results, err := opConv(node, inputs)
	if err != nil {
		return nil, err
	}

	activation := node.GetAttrString("activation", "none")
	switch activation {
	case "relu":
		out := results[0]
		switch t := out.(type) {
		case *tensor.Dense[float32]:
			data := t.Data()
			for i, v := range data {
				if v < 0 {
					data[i] = 0
				}
			}
		case *tensor.Dense[float64]:
			data := t.Data()
			for i, v := range data {
				if v < 0 {
					data[i] = 0
				}
			}
		}
	case "clip":
		clipMin := node.GetAttrFloat("clip_min", 0)
		clipMax := node.GetAttrFloat("clip_max", 6) // ReLU6 default
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			data := t.Data()
			for i, v := range data {
				if v < clipMin {
					data[i] = clipMin
				} else if v > clipMax {
					data[i] = clipMax
				}
			}
		}
	case "silu":
		// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			data := t.Data()
			for i, v := range data {
				data[i] = v * float32(1.0/(1.0+math.Exp(-float64(v))))
			}
		case *tensor.Dense[float64]:
			data := t.Data()
			for i, v := range data {
				data[i] = v / (1.0 + math.Exp(-v))
			}
		}
	case "leakyrelu":
		alpha := node.GetAttrFloat("leakyrelu_alpha", 0.01)
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			data := t.Data()
			for i, v := range data {
				if v < 0 { data[i] = v * alpha }
			}
		case *tensor.Dense[float64]:
			data := t.Data()
			a64 := float64(alpha)
			for i, v := range data {
				if v < 0 { data[i] = v * a64 }
			}
		}
	}

	return results, nil
}

func opConv(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		w := inputs[1].(*tensor.Dense[float32])
		var b *tensor.Dense[float32]
		if len(inputs) > 2 && inputs[2] != nil {
			b = inputs[2].(*tensor.Dense[float32])
		}
		var (
			out *tensor.Dense[float32]
			err error
		)
		switch x.Shape().NDim() {
		case 3:
			out, err = conv1d(x, w, b, node)
		default:
			out, err = conv2d(x, w, b, node)
		}
		if err != nil {
			return nil, fmt.Errorf("Conv: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		w := inputs[1].(*tensor.Dense[float64])
		var b *tensor.Dense[float64]
		if len(inputs) > 2 && inputs[2] != nil {
			b = inputs[2].(*tensor.Dense[float64])
		}
		var (
			out *tensor.Dense[float64]
			err error
		)
		switch x.Shape().NDim() {
		case 3:
			out, err = conv1d(x, w, b, node)
		default:
			out, err = conv2d(x, w, b, node)
		}
		if err != nil {
			return nil, fmt.Errorf("Conv: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("Conv: unsupported type %T", inputs[0])
	}
}

func opConvTranspose(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		w := inputs[1].(*tensor.Dense[float32])
		var b *tensor.Dense[float32]
		if len(inputs) > 2 && inputs[2] != nil {
			b = inputs[2].(*tensor.Dense[float32])
		}
		out, err := convTranspose2d(x, w, b, node)
		if err != nil {
			return nil, fmt.Errorf("ConvTranspose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		w := inputs[1].(*tensor.Dense[float64])
		var b *tensor.Dense[float64]
		if len(inputs) > 2 && inputs[2] != nil {
			b = inputs[2].(*tensor.Dense[float64])
		}
		out, err := convTranspose2d(x, w, b, node)
		if err != nil {
			return nil, fmt.Errorf("ConvTranspose: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("ConvTranspose: unsupported type %T", inputs[0])
	}
}

func conv1d[T tensor.Numeric](x, w *tensor.Dense[T], b *tensor.Dense[T], node *ir.Node) (*tensor.Dense[T], error) {
	xShape := x.Shape() // [N, C, L]
	wShape := w.Shape() // [OC, IC/group, K]
	if xShape.NDim() != 3 || wShape.NDim() != 3 {
		return nil, fmt.Errorf("conv1d requires 3D input and weight, got %v and %v", xShape, wShape)
	}

	N, C, L := xShape[0], xShape[1], xShape[2]
	OC, K := wShape[0], wShape[2]
	group := int(node.GetAttrInt("group", 1))
	icPerGroup := C / group
	ocPerGroup := OC / group

	strides := node.GetAttrInts("strides", []int64{1})
	dilations := node.GetAttrInts("dilations", []int64{1})
	stride := int(strides[0])
	dilation := int(dilations[0])
	effK := (K-1)*dilation + 1

	pads := normalizePads(node.GetAttrInts("pads", nil), 1)
	padLeft, padRight := int(pads[0]), int(pads[1])
	outL := (L+padLeft+padRight-effK)/stride + 1

	outShape := tensor.Shape{N, OC, outL}
	outData := make([]T, outShape.Size())
	xData, wData := x.Data(), w.Data()
	var biasData []T
	if b != nil {
		biasData = b.Data()
	}

	for n := 0; n < N; n++ {
		for g := 0; g < group; g++ {
			for oc := 0; oc < ocPerGroup; oc++ {
				absOC := g*ocPerGroup + oc
				for outPos := 0; outPos < outL; outPos++ {
					sum := T(0)
					if biasData != nil {
						sum = biasData[absOC]
					}
					for ic := 0; ic < icPerGroup; ic++ {
						absIC := g*icPerGroup + ic
						for k := 0; k < K; k++ {
							inPos := outPos*stride - padLeft + k*dilation
							if inPos < 0 || inPos >= L {
								continue
							}
							xIdx := (n*C+absIC)*L + inPos
							wIdx := ((absOC*icPerGroup+ic)*K + k)
							sum += xData[xIdx] * wData[wIdx]
						}
					}
					outData[(n*OC+absOC)*outL+outPos] = sum
				}
			}
		}
	}

	return tensor.NewDense[T](outShape, outData), nil
}

func conv2d[T tensor.Numeric](x, w *tensor.Dense[T], b *tensor.Dense[T], node *ir.Node) (*tensor.Dense[T], error) {
	xShape := x.Shape() // [N, C, H, W]
	wShape := w.Shape() // [OC, IC/group, KH, KW]

	if xShape.NDim() != 4 || wShape.NDim() != 4 {
		return nil, fmt.Errorf("conv2d requires 4D input and weight, got %v and %v", xShape, wShape)
	}

	N := xShape[0]
	C := xShape[1]
	H := xShape[2]
	W := xShape[3]
	OC := wShape[0]
	KH := wShape[2]
	KW := wShape[3]

	group := int(node.GetAttrInt("group", 1))
	icPerGroup := C / group
	ocPerGroup := OC / group

	strides := node.GetAttrInts("strides", []int64{1, 1})
	strideH, strideW := int(strides[0]), int(strides[1])

	dilations := node.GetAttrInts("dilations", []int64{1, 1})
	dilH, dilW := int(dilations[0]), int(dilations[1])

	effKH := (KH-1)*dilH + 1
	effKW := (KW-1)*dilW + 1

	// Fast path: depthwise float32 conv (group==C>1, icPerGroup==1, no dilation)
	kc := activeConvConfig
	useDepthwise := kc == nil || kc.UseDepthwiseKernel
	if useDepthwise && group > 1 && group == C && icPerGroup == 1 && dilH == 1 && dilW == 1 {
		if xf, ok := any(x.Data()).([]float32); ok {
			wf := any(w.Data()).([]float32)
			padTop, padLeft, padBottom, padRight := computePads(node, H, W, KH, KW, strideH, strideW)
			OH := (H+padTop+padBottom-KH)/strideH + 1
			OW := (W+padLeft+padRight-KW)/strideW + 1
			outShape := tensor.Shape{N, OC, OH, OW}
			var bf []float32
			if b != nil {
				bf = any(b.Data()).([]float32)
			}
			out := depthwiseF32(xf, wf, bf, N, C, H, W, KH, KW, OH, OW,
				strideH, strideW, padTop, padLeft)
			return any(tensor.NewDense[float32](outShape, out)).(*tensor.Dense[T]), nil
		}
	}

	padTop, padLeft, padBottom, padRight := computePads(node, H, W, effKH, effKW, strideH, strideW)

	OH := (H+padTop+padBottom-effKH)/strideH + 1
	OW := (W+padLeft+padRight-effKW)/strideW + 1

	// Fast path: 1x1 conv, stride 1, no padding, group 1, float32
	use1x1 := kc == nil || kc.Use1x1FastPath
	if use1x1 && KH == 1 && KW == 1 && strideH == 1 && strideW == 1 &&
		padTop == 0 && padLeft == 0 && padBottom == 0 && padRight == 0 &&
		group == 1 && dilH == 1 && dilW == 1 {
		if xf, ok := any(x.Data()).([]float32); ok {
			wf := any(w.Data()).([]float32)
			outShape := tensor.Shape{N, OC, H, W}
			HW := H * W
			outData := make([]float32, N*OC*HW)
			for n := 0; n < N; n++ {
				// X[n] is [C, H*W], W is [OC, C] — GEMM: W × X[n] → out[n]
				xSlice := xf[n*C*HW : (n+1)*C*HW]
				oSlice := outData[n*OC*HW : (n+1)*OC*HW]
				gemmF32(wf, xSlice, oSlice, OC, HW, C)
			}
			if b != nil {
				bf := any(b.Data()).([]float32)
				for n := 0; n < N; n++ {
					oBase := n * OC * HW
					for oc := 0; oc < OC; oc++ {
						bv := bf[oc]
						off := oBase + oc*HW
						oSlice := outData[off : off+HW : off+HW] // BCE
						for i := 0; i < HW; i++ {
							oSlice[i] += bv
						}
					}
				}
			}
			// Note: activation is applied by opFusedConv, not here
			return any(tensor.NewDense[float32](outShape, outData)).(*tensor.Dense[T]), nil
		}
	}

	outShape := tensor.Shape{N, OC, OH, OW}
	outData := make([]T, outShape.Size())
	xData := x.Data()
	wData := w.Data()

	// im2col + GEMM approach with optional goroutine parallelism.
	colSize := icPerGroup * KH * KW
	patchSize := OH * OW

	// Parallelism threshold: only when GEMM is large enough to amortize overhead
	gemmWork := ocPerGroup * patchSize * colSize
	useParallel := gemmWork > 500_000 && ocPerGroup >= 16 && (kc == nil || kc.UseParallelConv)
	nWorkers := 1
	if useParallel {
		nWorkers = kc.Workers()
		if nWorkers > ocPerGroup {
			nWorkers = ocPerGroup
		}
	}

	for n := 0; n < N; n++ {
		for g := 0; g < group; g++ {
			var pooledCol []float32
			var col []T
			if _, ok := any(xData).([]float32); ok {
				pooledCol = getFloat32Scratch(colSize * patchSize)
				col = any(pooledCol).([]T)
			} else {
				col = make([]T, colSize*patchSize)
			}
			im2col(xData, col, n, g, C, H, W, icPerGroup,
				KH, KW, OH, OW, strideH, strideW, padTop, padLeft, dilH, dilW)

			wOff := g * ocPerGroup * colSize
			oOff := n*OC*patchSize + g*ocPerGroup*patchSize

			// Fused GEMM + bias (same cache pass)
			if nWorkers <= 1 {
				if b != nil {
					if _, ok := any(wData).([]float32); ok && len(wData[wOff:]) >= ocPerGroup*colSize && len(col) >= colSize*patchSize && len(outData[oOff:]) >= ocPerGroup*patchSize && len(b.Data()[g*ocPerGroup:]) >= ocPerGroup {
						gemmF32WithBias(
							any(wData[wOff:]).([]float32),
							any(col).([]float32),
							any(outData[oOff:]).([]float32),
							ocPerGroup, patchSize, colSize,
							any(b.Data()[g*ocPerGroup:]).([]float32),
						)
					} else {
						gemmNN(wData[wOff:], col, outData[oOff:], ocPerGroup, patchSize, colSize)
						addBiasGroup(outData[oOff:], b.Data()[g*ocPerGroup:], ocPerGroup, patchSize)
					}
				} else {
					gemmNN(wData[wOff:], col, outData[oOff:], ocPerGroup, patchSize, colSize)
				}
			} else {
				var wg sync.WaitGroup
				// Round chunk size up to MR=4 boundary to avoid tail kernel explosion
				chunkSize := ((ocPerGroup+nWorkers-1)/nWorkers + 3) &^ 3
				for w := 0; w < nWorkers; w++ {
					ocStart := w * chunkSize
					ocEnd := min(ocStart+chunkSize, ocPerGroup)
					if ocStart >= ocEnd {
						break
					}
					wg.Add(1)
					go func(ocStart, ocEnd int) {
						defer wg.Done()
						gemmNN(wData[wOff+ocStart*colSize:], col,
							outData[oOff+ocStart*patchSize:],
							ocEnd-ocStart, patchSize, colSize)
					}(ocStart, ocEnd)
				}
				wg.Wait()
				if b != nil {
					addBiasGroup(outData[oOff:], b.Data()[g*ocPerGroup:], ocPerGroup, patchSize)
				}
			}
			putFloat32Scratch(pooledCol)
		}
	}

	return tensor.NewDense[T](outShape, outData), nil
}

func convTranspose2d[T tensor.Numeric](x, w *tensor.Dense[T], b *tensor.Dense[T], node *ir.Node) (*tensor.Dense[T], error) {
	xShape := x.Shape() // [N, C, H, W]
	wShape := w.Shape() // [C, OC/group, KH, KW]
	if xShape.NDim() != 4 || wShape.NDim() != 4 {
		return nil, fmt.Errorf("convTranspose2d requires 4D input and weight, got %v and %v", xShape, wShape)
	}

	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	if wShape[0] != C {
		return nil, fmt.Errorf("convTranspose2d channel mismatch: input %d vs weight %d", C, wShape[0])
	}
	group := int(node.GetAttrInt("group", 1))
	icPerGroup := C / group
	ocPerGroup := wShape[1]
	OC := ocPerGroup * group
	KH, KW := wShape[2], wShape[3]

	strides := node.GetAttrInts("strides", []int64{1, 1})
	strideH, strideW := int(strides[0]), int(strides[1])

	dilations := node.GetAttrInts("dilations", []int64{1, 1})
	dilH, dilW := int(dilations[0]), int(dilations[1])
	effKH := (KH-1)*dilH + 1
	effKW := (KW-1)*dilW + 1

	pads := normalizePads(node.GetAttrInts("pads", nil), 2)
	padTop, padLeft, padBottom, padRight := int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])

	outPads := normalizePads(node.GetAttrInts("output_padding", nil), 2)
	outPadH, outPadW := int(outPads[0]), int(outPads[1])

	OH := strideH*(H-1) + outPadH + effKH - padTop - padBottom
	OW := strideW*(W-1) + outPadW + effKW - padLeft - padRight

	if outShapeAttr := node.GetAttrInts("output_shape", nil); len(outShapeAttr) == 2 {
		OH, OW = int(outShapeAttr[0]), int(outShapeAttr[1])
	}

	outShape := tensor.Shape{N, OC, OH, OW}
	outData := make([]T, outShape.Size())
	xData := x.Data()
	wData := w.Data()

	// Fast path: float32, GEMM-based ConvTranspose
	// ConvTranspose = GEMM(W^T, X_col) + col2im
	// W is [IC, OC/g, KH, KW] → reshape to [IC, OC/g*KH*KW]
	// W^T is [OC/g*KH*KW, IC]
	// For each spatial position (ih, iw): output = W^T × x_vec → scatter to output
	useConvTrGEMM := activeConvConfig == nil || activeConvConfig.UseConvTransposeGEMM
	if useConvTrGEMM {
	if xf, ok := any(xData).([]float32); ok && group == 1 {
		wf := any(wData).([]float32)
		of := any(outData).([]float32)
		colSize := OC * KH * KW // = ocPerGroup * KH * KW since group=1

		// Precompute W transposed: W[IC, OC*KH*KW] → WT[OC*KH*KW, IC]
		wt := make([]float32, colSize*C)
		for ic := 0; ic < C; ic++ {
			for ock := 0; ock < colSize; ock++ {
				wt[ock*C+ic] = wf[ic*colSize+ock]
			}
		}

		HW := H * W
		for n := 0; n < N; n++ {
			// GEMM: WT[colSize, C] × X[C, H*W] → col[colSize, H*W]
			xOff := n * C * HW
			col := make([]float32, colSize*HW)
			gemmF32(wt, xf[xOff:xOff+C*HW], col, colSize, HW, C)

			// col2im: scatter col[OC*KH*KW, H*W] → output[OC, OH, OW]
			oOff := n * OC * OH * OW
			for ih := 0; ih < H; ih++ {
				for iw := 0; iw < W; iw++ {
					colIdx := ih*W + iw
					for oc := 0; oc < OC; oc++ {
						for kh := 0; kh < KH; kh++ {
							oh := ih*strideH - padTop + kh*dilH
							if oh < 0 || oh >= OH {
								continue
							}
							for kw := 0; kw < KW; kw++ {
								ow := iw*strideW - padLeft + kw*dilW
								if ow < 0 || ow >= OW {
									continue
								}
								ci := (oc*KH+kh)*KW + kw
								of[oOff+(oc*OH+oh)*OW+ow] += col[ci*HW+colIdx]
							}
						}
					}
				}
			}
		}
		goto addBias
	}
	} // end useConvTrGEMM

	// Generic fallback: 7-nested loop
	for n := 0; n < N; n++ {
		for g := 0; g < group; g++ {
			for ic := 0; ic < icPerGroup; ic++ {
				absIC := g*icPerGroup + ic
				for ih := 0; ih < H; ih++ {
					for iw := 0; iw < W; iw++ {
						xv := xData[((n*C+absIC)*H+ih)*W+iw]
						if xv == 0 {
							continue
						}
						for oc := 0; oc < ocPerGroup; oc++ {
							absOC := g*ocPerGroup + oc
							for kh := 0; kh < KH; kh++ {
								oh := ih*strideH - padTop + kh*dilH
								if oh < 0 || oh >= OH {
									continue
								}
								for kw := 0; kw < KW; kw++ {
									ow := iw*strideW - padLeft + kw*dilW
									if ow < 0 || ow >= OW {
										continue
									}
									wIdx := (((absIC*ocPerGroup+oc)*KH + kh) * KW) + kw
									outIdx := ((n*OC+absOC)*OH+oh)*OW + ow
									outData[outIdx] += xv * wData[wIdx]
								}
							}
						}
					}
				}
			}
		}
	}

addBias:

	if b != nil {
		bias := b.Data()
		for n := 0; n < N; n++ {
			for oc := 0; oc < OC; oc++ {
				bv := bias[oc]
				base := (n*OC + oc) * OH * OW
				hw := OH * OW
				oSlice := outData[base : base+hw : base+hw] // BCE
				for i := 0; i < hw; i++ {
					oSlice[i] += bv
				}
			}
		}
	}

	return tensor.NewDense[T](outShape, outData), nil
}

// addBiasGroup adds bias[oc] to each spatial position in out[oc, patchSize].
func addBiasGroup[T tensor.Numeric](out []T, bias []T, ocPerGroup, patchSize int) {
	for oc := 0; oc < ocPerGroup; oc++ {
		bv := bias[oc]
		off := oc * patchSize
		oSlice := out[off : off+patchSize : off+patchSize] // BCE
		for i := 0; i < patchSize; i++ {
			oSlice[i] += bv
		}
	}
}

// im2col extracts image patches into a column matrix.
// col layout: [icPerGroup*KH*KW, OH*OW]
func im2col[T tensor.Numeric](
	xData []T, col []T,
	n, g, C, H, W, icPerGroup, KH, KW, OH, OW,
	strideH, strideW, padTop, padLeft, dilH, dilW int,
) {
	HW := H * W

	// Fast path: 1x1 kernel, stride 1, no padding → just copy input rows
	if KH == 1 && KW == 1 && strideH == 1 && strideW == 1 &&
		padTop == 0 && padLeft == 0 && OH == H && OW == W {
		for ic := 0; ic < icPerGroup; ic++ {
			absIC := g*icPerGroup + ic
			src := n*C*HW + absIC*HW
			copy(col[ic*HW:(ic+1)*HW], xData[src:src+HW])
		}
		return
	}

	// Fast path: no padding → skip bounds checks on ih/iw
	// Verify OH/OW match no-padding case to avoid asymmetric padding issues
	noPadOH := (H-KH)/strideH + 1
	noPadOW := (W-KW)/strideW + 1
	if padTop == 0 && padLeft == 0 && dilH == 1 && dilW == 1 && OH == noPadOH && OW == noPadOW {
		colIdx := 0
		for ic := 0; ic < icPerGroup; ic++ {
			absIC := g*icPerGroup + ic
			xBase := n*C*HW + absIC*HW
			for kh := 0; kh < KH; kh++ {
				for kw := 0; kw < KW; kw++ {
					for oh := 0; oh < OH; oh++ {
						ih := oh*strideH + kh
						rowBase := xBase + ih*W
						for ow := 0; ow < OW; ow++ {
							col[colIdx] = xData[rowBase+ow*strideW+kw]
							colIdx++
						}
					}
				}
			}
		}
		return
	}

	// General path with padding/dilation
	colIdx := 0
	for ic := 0; ic < icPerGroup; ic++ {
		absIC := g*icPerGroup + ic
		xBase := n*C*HW + absIC*HW
		for kh := 0; kh < KH; kh++ {
			for kw := 0; kw < KW; kw++ {
				for oh := 0; oh < OH; oh++ {
					ih := oh*strideH - padTop + kh*dilH
					if ih < 0 || ih >= H {
						for ow := 0; ow < OW; ow++ {
							col[colIdx] = 0
							colIdx++
						}
						continue
					}
					rowBase := xBase + ih*W
					for ow := 0; ow < OW; ow++ {
						iw := ow*strideW - padLeft + kw*dilW
						if iw >= 0 && iw < W {
							col[colIdx] = xData[rowBase+iw]
						} else {
							col[colIdx] = 0
						}
						colIdx++
					}
				}
			}
		}
	}
}

// gemmNN computes C += A * B where A is [M,K], B is [K,N], C is [M,N].
// Dispatches to float32-specialized kernel when possible.
func gemmNN[T tensor.Numeric](A, B, C []T, M, N, K int) {
	if N <= 0 || K <= 0 || M <= 0 {
		return
	}
	if maxM := len(C) / N; M > maxM {
		M = maxM
	}
	if maxM := len(A) / K; M > maxM {
		M = maxM
	}
	if maxK := len(B) / N; K > maxK {
		K = maxK
	}

	// Float32 fast path with unrolled inner loops.
	// Keep thin-M cases on the generic path; some encoder shapes hit edge cases
	// in the tiled microkernel and correctness matters more than peak speed there.
	if af, ok := any(A).([]float32); ok && M >= 16 && len(A) >= M*K && len(B) >= K*N && len(C) >= M*N {
		bf := any(B).([]float32)
		cf := any(C).([]float32)
		gemmF32(af, bf, cf, M, N, K)
		return
	}
	if M*K+K*N > 32*1024 {
		gemmTiledGeneric(A, B, C, M, N, K)
		return
	}
	for i := 0; i < M; i++ {
		cRow := C[i*N : i*N+N : i*N+N] // BCE
		aBase := i * K
		for k := 0; k < K; k++ {
			aik := A[aBase+k]
			if aik == 0 {
				continue
			}
			bRow := B[k*N : k*N+N : k*N+N] // BCE
			for j := 0; j < N; j++ {
				cRow[j] += aik * bRow[j]
			}
		}
	}
}

func gemmTiledGeneric[T tensor.Numeric](A, B, C []T, M, N, K int) {
	const tM = 32
	const tN = 128
	const tK = 64

	for i0 := 0; i0 < M; i0 += tM {
		iEnd := min(i0+tM, M)
		for k0 := 0; k0 < K; k0 += tK {
			kEnd := min(k0+tK, K)
			for j0 := 0; j0 < N; j0 += tN {
				jEnd := min(j0+tN, N)
				jLen := jEnd - j0
				for i := i0; i < iEnd; i++ {
					cRow := C[i*N+j0 : i*N+jEnd : i*N+jEnd] // BCE
					aBase := i * K
					for k := k0; k < kEnd; k++ {
						aik := A[aBase+k]
						if aik == 0 {
							continue
						}
						bRow := B[k*N+j0 : k*N+jEnd : k*N+jEnd] // BCE
						for j := 0; j < jLen; j++ {
							cRow[j] += aik * bRow[j]
						}
					}
				}
			}
		}
	}
}
