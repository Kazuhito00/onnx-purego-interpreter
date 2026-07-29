package ops

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"

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
	postScale := node.GetAttrFloat("post_scale", 1)
	postBias := node.GetAttrFloat("post_bias", 0)

	// float32 の epilogue は要素独立なので大きなテンソルでは並列化する。
	// minLen は演算の重さに応じた並列化閾値(軽量なクランプ系は大きめ)。
	epilogueF32With := func(minLen int, data []float32, fn func(v float32) float32) {
		workers := 1
		if len(data) >= minLen && (activeConvConfig == nil || activeConvConfig.UseParallelConv) {
			workers = activeConvConfig.Workers()
		}
		forEachRangeParallel(len(data), workers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				data[i] = fn(data[i])
			}
		})
	}
	// 軽量(クランプ・積和)系: 帯域律速のため大きなテンソルのみ並列化
	epilogueF32 := func(data []float32, fn func(v float32) float32) {
		epilogueF32With(cheapParallelMin, data, fn)
	}
	// 重量(exp 等)系: 計算律速のため小さめの閾値で並列化
	epilogueF32Heavy := func(data []float32, fn func(v float32) float32) {
		epilogueF32With(elementwiseParallelMin, data, fn)
	}

	switch activation {
	case "relu":
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			epilogueF32(t.Data(), func(v float32) float32 {
				if v < 0 {
					return 0
				}
				return v
			})
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
			epilogueF32(t.Data(), func(v float32) float32 {
				if v < clipMin {
					return clipMin
				}
				if v > clipMax {
					return clipMax
				}
				return v
			})
		}
	case "silu":
		// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			epilogueF32Heavy(t.Data(), func(v float32) float32 {
				return v * float32(1.0/(1.0+math.Exp(-float64(v))))
			})
		case *tensor.Dense[float64]:
			data := t.Data()
			for i, v := range data {
				data[i] = v / (1.0 + math.Exp(-v))
			}
		}
	case "hardswish":
		// HardSwish: x * clip(x+3, 0, 6) / 6
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			epilogueF32(t.Data(), func(v float32) float32 {
				hsig := float32(math.Min(math.Max(float64(v+3), 0), 6) / 6.0)
				return v * hsig
			})
		case *tensor.Dense[float64]:
			data := t.Data()
			for i, v := range data {
				data[i] = v * (math.Min(math.Max(v+3, 0), 6) / 6.0)
			}
		}
	case "leakyrelu":
		alpha := node.GetAttrFloat("leakyrelu_alpha", 0.01)
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			epilogueF32(t.Data(), func(v float32) float32 {
				if v < 0 {
					return v * alpha
				}
				return v
			})
		case *tensor.Dense[float64]:
			data := t.Data()
			a64 := float64(alpha)
			for i, v := range data {
				if v < 0 {
					data[i] = v * a64
				}
			}
		}
	}

	if postScale != 1 || postBias != 0 {
		switch t := results[0].(type) {
		case *tensor.Dense[float32]:
			epilogueF32(t.Data(), func(v float32) float32 {
				return v*postScale + postBias
			})
		case *tensor.Dense[float64]:
			data := t.Data()
			s, b := float64(postScale), float64(postBias)
			for i, v := range data {
				data[i] = v*s + b
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
			dwWorkers := 1
			if N*C*OH*OW*KH*KW > 250_000 && (kc == nil || kc.UseParallelConv) {
				cfg := kc
				if cfg == nil {
					cfg = DefaultKernelConfig()
				}
				dwWorkers = cfg.Workers()
			}
			out := depthwiseF32(xf, wf, bf, N, C, H, W, KH, KW, OH, OW,
				strideH, strideW, padTop, padLeft, dwWorkers)
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
				work := OC * HW * C
				if work > 500_000 && (kc == nil || kc.UseParallelConv) {
					cfg := kc
					if cfg == nil {
						cfg = DefaultKernelConfig()
					}
					// OC(行)が小さいと行分割では worker 数が頭打ちになるため、
					// 行分割で worker を飽和できず、かつ列ストライプ数の方が多い場合のみ
					// 列(HW)方向に分割する(行分割で足りる形状は行分割の方が速い)
					colStripes := (HW + nc - 1) / nc
					preferCol := OC/4 < cfg.Workers() && colStripes > OC/4
					// B(入力)がキャッシュを大きく超え A(重み)が L2 に収まる形状では、
					// 行分割だと worker ごとに B 全体を読み直すため列分割が有利
					if !preferCol && C*HW*4 > 8<<20 && OC*C*4 <= 2<<20 && colStripes >= cfg.Workers() {
						preferCol = true
					}
					if preferCol {
						gemmF32ParallelCols(wf, xSlice, oSlice, OC, HW, C, cfg.Workers())
					} else if OC >= 16 {
						nWorkers := min(cfg.Workers(), OC)
						chunk := ((OC+nWorkers-1)/nWorkers + 3) &^ 3
						var wg sync.WaitGroup
						for worker := 0; worker < nWorkers; worker++ {
							start := worker * chunk
							end := min(start+chunk, OC)
							if start >= end {
								break
							}
							wg.Add(1)
							go func(start, end int) {
								defer wg.Done()
								gemmF32(wf[start*C:end*C], xSlice, oSlice[start*HW:end*HW], end-start, HW, C)
							}(start, end)
						}
						wg.Wait()
					} else {
						gemmF32(wf, xSlice, oSlice, OC, HW, C)
					}
				} else {
					gemmF32(wf, xSlice, oSlice, OC, HW, C)
				}
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
	maxWorkers := 1
	if gemmWork > 500_000 && (kc == nil || kc.UseParallelConv) {
		cfg := kc
		if cfg == nil {
			cfg = DefaultKernelConfig()
		}
		maxWorkers = cfg.Workers()
	}

	xf, xIsF32 := any(xData).([]float32)
	wf, wIsF32 := any(wData).([]float32)
	for n := 0; n < N; n++ {
		for g := 0; g < group; g++ {
			wOff := g * ocPerGroup * colSize
			oOff := n*OC*patchSize + g*ocPerGroup*patchSize

			if xIsF32 && wIsF32 {
				var bias []float32
				if b != nil {
					bias = any(b.Data()).([]float32)[g*ocPerGroup : (g+1)*ocPerGroup]
				}
				convIm2colGemmStripsF32(
					xf, wf[wOff:wOff+ocPerGroup*colSize], bias,
					any(outData).([]float32)[oOff:oOff+ocPerGroup*patchSize],
					n, g, C, H, W, icPerGroup, ocPerGroup,
					KH, KW, OH, OW, strideH, strideW, padTop, padLeft, dilH, dilW,
					maxWorkers)
				continue
			}

			// 汎用フォールバック(float32 以外)
			col := make([]T, colSize*patchSize)
			im2col(xData, col, n, g, C, H, W, icPerGroup,
				KH, KW, OH, OW, strideH, strideW, padTop, padLeft, dilH, dilW, 0, OH)
			gemmNN(wData[wOff:], col, outData[oOff:], ocPerGroup, patchSize, colSize)
			if b != nil {
				addBiasGroup(outData[oOff:], b.Data()[g*ocPerGroup:], ocPerGroup, patchSize)
			}
		}
	}

	return tensor.NewDense[T](outShape, outData), nil
}

// convStripTargetFloats は 1 ストリップの col バッファの目標要素数(≈768KB)。
// col を L2 キャッシュ内に保つことで im2col 結果の RAM 往復を避ける。
// テストから縮小してストリップ分割を強制できるよう変数にしている。
var convStripTargetFloats = 192 * 1024

// convIm2colGemmStripsF32 は im2col + GEMM を出力行ストリップ単位で融合実行する。
// col 行列全体(数十 MB になり得る)を作らず、L2 に収まるストリップごとに
// im2col → GEMM → bias 付きコピーを行い、ストリップを worker に動的分配する。
func convIm2colGemmStripsF32(
	xf, wf, bias, of []float32,
	n, g, C, H, W, icPerGroup, ocPerGroup int,
	KH, KW, OH, OW, strideH, strideW, padTop, padLeft, dilH, dilW int,
	maxWorkers int,
) {
	colSize := icPerGroup * KH * KW
	patchSize := OH * OW

	// ストリップ行数: A(重み)の再読込を抑えるため 1 ストリップ ≥ nc 列を確保しつつ、
	// col ストリップが L2 に収まる範囲に抑える
	minRows := (nc + OW - 1) / OW
	l2Rows := convStripTargetFloats / (colSize * OW)
	stripRows := max(minRows, l2Rows)
	// worker より十分多くのストリップを作り端数を平準化(下限は minRows)
	if maxWorkers > 1 {
		if perW := (OH + maxWorkers - 1) / maxWorkers; stripRows > perW {
			stripRows = max(minRows, perW)
		}
	}
	stripRows = min(stripRows, OH)

	nStrips := (OH + stripRows - 1) / stripRows

	// ストリップ数で十分な並列度が得られない場合(深い層の小空間 conv 等)は、
	// col 全体を作って OC(行)方向に分割する従来方式の方が速い
	if nStrips < maxWorkers && ocPerGroup/4 > nStrips {
		col := getFloat32Scratch(colSize * patchSize)
		im2col(xf, col, n, g, C, H, W, icPerGroup,
			KH, KW, OH, OW, strideH, strideW, padTop, padLeft, dilH, dilW, 0, OH)
		nWorkers := min(maxWorkers, ocPerGroup/4)
		chunk := ((ocPerGroup+nWorkers-1)/nWorkers + 3) &^ 3
		var wg sync.WaitGroup
		for w := 0; w < nWorkers; w++ {
			ocStart := w * chunk
			if ocStart >= ocPerGroup {
				break
			}
			ocEnd := min(ocStart+chunk, ocPerGroup)
			wg.Add(1)
			go func(ocStart, ocEnd int) {
				defer wg.Done()
				if bias != nil {
					gemmF32WithBias(wf[ocStart*colSize:], col, of[ocStart*patchSize:],
						ocEnd-ocStart, patchSize, colSize, bias[ocStart:])
				} else {
					gemmF32(wf[ocStart*colSize:], col, of[ocStart*patchSize:],
						ocEnd-ocStart, patchSize, colSize)
				}
			}(ocStart, ocEnd)
		}
		wg.Wait()
		putFloat32Scratch(col)
		return
	}

	// 単一ストリップなら中間バッファを介さず出力へ直接 GEMM(bias 融合)
	if nStrips <= 1 {
		col := getFloat32Scratch(colSize * patchSize)
		im2col(xf, col, n, g, C, H, W, icPerGroup,
			KH, KW, OH, OW, strideH, strideW, padTop, padLeft, dilH, dilW, 0, OH)
		if bias != nil {
			gemmF32WithBias(wf, col, of, ocPerGroup, patchSize, colSize, bias)
		} else {
			gemmF32(wf, col, of, ocPerGroup, patchSize, colSize)
		}
		putFloat32Scratch(col)
		return
	}
	runStrip := func(ohFrom, ohTo int) {
		stripPatch := (ohTo - ohFrom) * OW
		col := getFloat32Scratch(colSize * stripPatch)
		im2col(xf, col, n, g, C, H, W, icPerGroup,
			KH, KW, OH, OW, strideH, strideW, padTop, padLeft, dilH, dilW, ohFrom, ohTo)
		cbuf := getFloat32Scratch(ocPerGroup * stripPatch)
		clear(cbuf)
		gemmF32(wf, col, cbuf, ocPerGroup, stripPatch, colSize)
		// C ストリップを出力の該当列範囲へ行コピー(bias 同時適用)
		dstBase := ohFrom * OW
		for oc := 0; oc < ocPerGroup; oc++ {
			src := cbuf[oc*stripPatch : (oc+1)*stripPatch]
			dst := of[oc*patchSize+dstBase : oc*patchSize+dstBase+stripPatch]
			if bias != nil {
				bv := bias[oc]
				for i, v := range src {
					dst[i] = v + bv
				}
			} else {
				copy(dst, src)
			}
		}
		putFloat32Scratch(cbuf)
		putFloat32Scratch(col)
	}

	nWorkers := min(maxWorkers, nStrips)
	if nWorkers <= 1 {
		for s := 0; s < nStrips; s++ {
			ohFrom := s * stripRows
			runStrip(ohFrom, min(ohFrom+stripRows, OH))
		}
		return
	}
	// 動的分配: P/E コア混在でも遅い worker がボトルネックにならない
	var nextStrip atomic.Int32
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				s := int(nextStrip.Add(1)) - 1
				if s >= nStrips {
					return
				}
				ohFrom := s * stripRows
				runStrip(ohFrom, min(ohFrom+stripRows, OH))
			}
		}()
	}
	wg.Wait()
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
			// GEMM(列分割)と col2im(oc 分割)を並列化する
			ctWorkers := 1
			if colSize*HW*C > 500_000 && (activeConvConfig == nil || activeConvConfig.UseParallelConv) {
				ctWorkers = activeConvConfig.Workers()
			}
			for n := 0; n < N; n++ {
				// GEMM: WT[colSize, C] × X[C, H*W] → col[colSize, H*W]
				xOff := n * C * HW
				col := getFloat32Scratch(colSize * HW)
				clear(col)
				gemmF32ParallelCols(wt, xf[xOff:xOff+C*HW], col, colSize, HW, C, ctWorkers)

				// col2im: scatter col[OC*KH*KW, H*W] → output[OC, OH, OW]
				// oc ごとに出力先が互いに素なため oc 単位で並列化できる
				oOff := n * OC * OH * OW
				forEachIndexParallel(OC, ctWorkers, func(oc int) {
					for kh := 0; kh < KH; kh++ {
						oh0 := -padTop + kh*dilH
						for kw := 0; kw < KW; kw++ {
							ci := (oc*KH+kh)*KW + kw
							colRow := col[ci*HW : (ci+1)*HW]
							ow0 := -padLeft + kw*dilW
							for ih := 0; ih < H; ih++ {
								oh := ih*strideH + oh0
								if oh < 0 || oh >= OH {
									continue
								}
								outRow := oOff + (oc*OH+oh)*OW
								base := ih * W
								for iw := 0; iw < W; iw++ {
									ow := iw*strideW + ow0
									if ow < 0 || ow >= OW {
										continue
									}
									of[outRow+ow] += colRow[base+iw]
								}
							}
						}
					}
				})
				putFloat32Scratch(col)
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
// 出力行の範囲 [ohFrom, ohTo) のみを書き出す(ストリップ実行用)。
// col layout: [icPerGroup*KH*KW, (ohTo-ohFrom)*OW]
func im2col[T tensor.Numeric](
	xData []T, col []T,
	n, g, C, H, W, icPerGroup, KH, KW, OH, OW,
	strideH, strideW, padTop, padLeft, dilH, dilW, ohFrom, ohTo int,
) {
	HW := H * W
	stripPatch := (ohTo - ohFrom) * OW

	// Fast path: 1x1 kernel, stride 1, no padding → just copy input rows
	if KH == 1 && KW == 1 && strideH == 1 && strideW == 1 &&
		padTop == 0 && padLeft == 0 && OH == H && OW == W {
		for ic := 0; ic < icPerGroup; ic++ {
			absIC := g*icPerGroup + ic
			src := n*C*HW + absIC*HW + ohFrom*W
			copy(col[ic*stripPatch:(ic+1)*stripPatch], xData[src:src+stripPatch])
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
					for oh := ohFrom; oh < ohTo; oh++ {
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
	// kw ごとに水平方向の有効範囲 [owLo, owHi) を事前計算し、
	// 内側は境界チェックなし(stride 1 なら行 copy)で書き出す
	owLos := make([]int, KW)
	owHis := make([]int, KW)
	for kw := 0; kw < KW; kw++ {
		off := kw*dilW - padLeft // iw = ow*strideW + off
		lo := 0
		if off < 0 {
			lo = (-off + strideW - 1) / strideW
		}
		hi := 0
		if off <= W-1 {
			hi = min(OW, (W-1-off)/strideW+1)
		}
		if hi < lo {
			hi = lo
		}
		owLos[kw], owHis[kw] = lo, hi
	}

	colIdx := 0
	for ic := 0; ic < icPerGroup; ic++ {
		absIC := g*icPerGroup + ic
		xBase := n*C*HW + absIC*HW
		for kh := 0; kh < KH; kh++ {
			for kw := 0; kw < KW; kw++ {
				off := kw*dilW - padLeft
				owLo, owHi := owLos[kw], owHis[kw]
				for oh := ohFrom; oh < ohTo; oh++ {
					ih := oh*strideH - padTop + kh*dilH
					if ih < 0 || ih >= H {
						zeroFill(col[colIdx : colIdx+OW])
						colIdx += OW
						continue
					}
					rowBase := xBase + ih*W
					zeroFill(col[colIdx : colIdx+owLo])
					colIdx += owLo
					if strideW == 1 {
						src := rowBase + owLo + off
						copy(col[colIdx:colIdx+owHi-owLo], xData[src:src+owHi-owLo])
						colIdx += owHi - owLo
					} else {
						src := rowBase + owLo*strideW + off
						for ow := owLo; ow < owHi; ow++ {
							col[colIdx] = xData[src]
							colIdx++
							src += strideW
						}
					}
					zeroFill(col[colIdx : colIdx+OW-owHi])
					colIdx += OW - owHi
				}
			}
		}
	}
}

// zeroFill は s を 0 クリアする(scratch バッファは前回の値が残っているため必須)。
func zeroFill[T tensor.Numeric](s []T) {
	clear(s)
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
