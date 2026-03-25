package ops

import (
	"fmt"
	"math"
	"strings"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Einsum ──

func opEinsum(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	equation := node.GetAttrString("equation", "")
	if equation == "" {
		return nil, fmt.Errorf("Einsum: no equation")
	}

	// Support common 2-input patterns: "ij,jk->ik" (matmul), "ij,ij->ij" (element-wise), "ij->ji" (transpose)
	parts := strings.Split(equation, "->")
	if len(parts) != 2 {
		return nil, fmt.Errorf("Einsum: unsupported equation %q", equation)
	}
	inputParts := strings.Split(parts[0], ",")
	outputSubscripts := strings.TrimSpace(parts[1])

	if len(inputParts) == 2 && len(inputs) == 2 {
		// 2-input einsum
		return einsum2(inputs[0], inputs[1], strings.TrimSpace(inputParts[0]), strings.TrimSpace(inputParts[1]), outputSubscripts)
	}
	if len(inputParts) == 1 && len(inputs) == 1 {
		// 1-input einsum (transpose, trace, etc.)
		return einsum1(inputs[0], strings.TrimSpace(inputParts[0]), outputSubscripts)
	}

	return nil, fmt.Errorf("Einsum: unsupported %d-input equation %q", len(inputs), equation)
}

func einsum1(a tensor.Tensor, subsA, subsOut string) ([]tensor.Tensor, error) {
	// Simple transpose: "ij->ji", "ijk->ikj", etc.
	at, ok := a.(*tensor.Dense[float32])
	if !ok {
		return nil, fmt.Errorf("Einsum: unsupported type %T", a)
	}
	as := at.Shape()
	ndim := len(subsA)

	// Build permutation
	perm := make([]int, len(subsOut))
	for i, c := range subsOut {
		for j, d := range subsA {
			if c == d {
				perm[i] = j
				break
			}
		}
	}

	// Compute output shape
	outShape := make(tensor.Shape, len(subsOut))
	for i, p := range perm {
		outShape[i] = as[p]
	}

	// Transpose
	srcStrides := tensor.Strides(as)
	dstStrides := tensor.Strides(outShape)
	data := at.Data()
	out := make([]float32, outShape.Size())
	for i := 0; i < len(out); i++ {
		srcIdx := 0
		rem := i
		for d := 0; d < len(outShape); d++ {
			coord := rem / dstStrides[d]
			rem %= dstStrides[d]
			srcIdx += coord * srcStrides[perm[d]]
		}
		out[i] = data[srcIdx]
	}
	_ = ndim
	return []tensor.Tensor{tensor.NewDense[float32](outShape, out)}, nil
}

func einsum2(a, b tensor.Tensor, subsA, subsB, subsOut string) ([]tensor.Tensor, error) {
	at, ok := a.(*tensor.Dense[float32])
	if !ok {
		return nil, fmt.Errorf("Einsum: unsupported type %T", a)
	}
	bt := b.(*tensor.Dense[float32])
	as, bs := at.Shape(), bt.Shape()

	// Map subscript char → dimension size
	dimMap := make(map[byte]int)
	for i := 0; i < len(subsA); i++ {
		dimMap[subsA[i]] = as[i]
	}
	for i := 0; i < len(subsB); i++ {
		dimMap[subsB[i]] = bs[i]
	}

	// Find contraction indices (in both A and B but not in output)
	outSet := make(map[byte]bool)
	for i := 0; i < len(subsOut); i++ {
		outSet[subsOut[i]] = true
	}
	var contractChars []byte
	for i := 0; i < len(subsA); i++ {
		c := subsA[i]
		if !outSet[c] {
			for j := 0; j < len(subsB); j++ {
				if subsB[j] == c {
					contractChars = append(contractChars, c)
					break
				}
			}
		}
	}

	// Output shape
	outShape := make(tensor.Shape, len(subsOut))
	for i := 0; i < len(subsOut); i++ {
		outShape[i] = dimMap[subsOut[i]]
	}

	ad, bd := at.Data(), bt.Data()
	out := make([]float32, outShape.Size())

	// General contraction via nested iteration
	aStrides := tensor.Strides(as)
	bStrides := tensor.Strides(bs)
	outStrides := tensor.Strides(outShape)

	// Compute contraction size
	contractSize := 1
	for _, c := range contractChars {
		contractSize *= dimMap[c]
	}

	// For each output element
	for oi := 0; oi < len(out); oi++ {
		// Decode output coordinates
		outCoords := make(map[byte]int)
		rem := oi
		for d := 0; d < len(subsOut); d++ {
			outCoords[subsOut[d]] = rem / outStrides[d]
			rem %= outStrides[d]
		}

		var sum float32
		// Iterate over contraction dims
		for ci := 0; ci < contractSize; ci++ {
			cCoords := make(map[byte]int)
			crem := ci
			for _, c := range contractChars {
				sz := dimMap[c]
				cCoords[c] = crem % sz
				crem /= sz
			}

			// Compute A index
			aIdx := 0
			for d := 0; d < len(subsA); d++ {
				c := subsA[d]
				if coord, ok := outCoords[c]; ok {
					aIdx += coord * aStrides[d]
				} else if coord, ok := cCoords[c]; ok {
					aIdx += coord * aStrides[d]
				}
			}
			// Compute B index
			bIdx := 0
			for d := 0; d < len(subsB); d++ {
				c := subsB[d]
				if coord, ok := outCoords[c]; ok {
					bIdx += coord * bStrides[d]
				} else if coord, ok := cCoords[c]; ok {
					bIdx += coord * bStrides[d]
				}
			}
			sum += ad[aIdx] * bd[bIdx]
		}
		out[oi] = sum
	}

	return []tensor.Tensor{tensor.NewDense[float32](outShape, out)}, nil
}

// ── RNN (simple/vanilla) ──

func opRNN(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	hiddenSize := int(node.GetAttrInt("hidden_size", 0))
	activations := node.GetAttrString("activations", "Tanh")
	_ = activations // only Tanh supported

	x := inputs[0].(*tensor.Dense[float32])
	w := inputs[1].(*tensor.Dense[float32])
	r := inputs[2].(*tensor.Dense[float32])

	xShape := x.Shape()
	seqLen, batch, inputSize := xShape[0], xShape[1], xShape[2]
	if hiddenSize == 0 {
		hiddenSize = r.Shape()[2]
	}
	H := hiddenSize

	var biasData []float32
	if len(inputs) > 3 && inputs[3] != nil {
		biasData = inputs[3].(*tensor.Dense[float32]).Data()
	}
	var hData []float32
	if len(inputs) > 5 && inputs[5] != nil {
		h := inputs[5].(*tensor.Dense[float32])
		hData = make([]float32, h.Len())
		copy(hData, h.Data())
	} else {
		hData = make([]float32, batch*H)
	}

	wData, rData, xData := w.Data(), r.Data(), x.Data()
	yData := make([]float32, seqLen*batch*H)

	for t := 0; t < seqLen; t++ {
		for b := 0; b < batch; b++ {
			xOff := t*batch*inputSize + b*inputSize
			hOff := b * H
			for h := 0; h < H; h++ {
				var sum float32
				for j := 0; j < inputSize; j++ {
					sum += wData[h*inputSize+j] * xData[xOff+j]
				}
				for j := 0; j < H; j++ {
					sum += rData[h*H+j] * hData[hOff+j]
				}
				if biasData != nil {
					sum += biasData[h] + biasData[H+h]
				}
				hData[hOff+h] = float32(math.Tanh(float64(sum)))
			}
			copy(yData[t*batch*H+b*H:], hData[hOff:hOff+H])
		}
	}

	Y := tensor.NewDense[float32](tensor.Shape{seqLen, 1, batch, H}, yData)
	Yh := tensor.NewDense[float32](tensor.Shape{1, batch, H}, hData)
	return []tensor.Tensor{Y, Yh}, nil
}

// ── Quantization stubs ──
// These provide minimal support for dequantize/quantize paths.

func opDequantizeLinear(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// x_output = (x - zero_point) * scale
	x := inputs[0]
	scale := inputs[1].(*tensor.Dense[float32])
	scaleData := scale.Data()

	switch xt := x.(type) {
	case *tensor.Dense[uint8]:
		var zpData []uint8
		if len(inputs) > 2 && inputs[2] != nil {
			zpData = inputs[2].(*tensor.Dense[uint8]).Data()
		}
		data := xt.Data()
		out := make([]float32, len(data))
		if len(scaleData) == 1 {
			s := scaleData[0]
			zp := uint8(0)
			if zpData != nil {
				zp = zpData[0]
			}
			for i, v := range data {
				out[i] = float32(int(v)-int(zp)) * s
			}
		} else {
			// Per-axis (axis=1 for weights typically)
			axis := int(node.GetAttrInt("axis", 1))
			shape := xt.Shape()
			outerSize := 1
			for d := 0; d < axis; d++ {
				outerSize *= shape[d]
			}
			axisSize := shape[axis]
			innerSize := 1
			for d := axis + 1; d < shape.NDim(); d++ {
				innerSize *= shape[d]
			}
			for o := 0; o < outerSize; o++ {
				for a := 0; a < axisSize; a++ {
					s := scaleData[a]
					zp := uint8(0)
					if zpData != nil {
						zp = zpData[a]
					}
					base := o*axisSize*innerSize + a*innerSize
					for i := 0; i < innerSize; i++ {
						out[base+i] = float32(int(data[base+i])-int(zp)) * s
					}
				}
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](xt.Shape().Clone(), out)}, nil
	case *tensor.Dense[int8]:
		data := xt.Data()
		out := make([]float32, len(data))
		s := scaleData[0]
		zp := int8(0)
		if len(inputs) > 2 && inputs[2] != nil {
			if z, ok := inputs[2].(*tensor.Dense[int8]); ok && z.Len() > 0 {
				zp = z.Data()[0]
			}
		}
		for i, v := range data {
			out[i] = float32(int(v)-int(zp)) * s
		}
		return []tensor.Tensor{tensor.NewDense[float32](xt.Shape().Clone(), out)}, nil
	case *tensor.Dense[float32]:
		// Float dequant is identity * scale
		data := xt.Data()
		out := make([]float32, len(data))
		s := scaleData[0]
		for i, v := range data {
			out[i] = v * s
		}
		return []tensor.Tensor{tensor.NewDense[float32](xt.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("DequantizeLinear: unsupported type %T", x)
	}
}

func opQuantizeLinear(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0].(*tensor.Dense[float32])
	scale := inputs[1].(*tensor.Dense[float32])
	s := scale.Data()[0]
	zp := uint8(0)
	if len(inputs) > 2 && inputs[2] != nil {
		if z, ok := inputs[2].(*tensor.Dense[uint8]); ok && z.Len() > 0 {
			zp = z.Data()[0]
		}
	}

	data := x.Data()
	out := make([]uint8, len(data))
	for i, v := range data {
		q := math.Round(float64(v)/float64(s)) + float64(zp)
		if q < 0 {
			q = 0
		}
		if q > 255 {
			q = 255
		}
		out[i] = uint8(q)
	}
	return []tensor.Tensor{tensor.NewDense[uint8](x.Shape().Clone(), out)}, nil
}

// ── Signal processing ops ──

// DFT: Discrete Fourier Transform.
// inputs: input (real or complex), [dft_length], [axis]
// output: complex tensor [... , 2] where last dim is [real, imag]
func opDFT(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	onesided := node.GetAttrInt("onesided", 0) != 0
	inverse := node.GetAttrInt("inverse", 0) != 0

	x := inputs[0].(*tensor.Dense[float32])
	xShape := x.Shape()
	xData := x.Data()

	// Determine axis (default: second-to-last for signal dim)
	axis := 1
	if len(inputs) > 2 && inputs[2] != nil {
		axis = int(inputs[2].(*tensor.Dense[int64]).Data()[0])
		if axis < 0 {
			axis += xShape.NDim()
		}
	}

	// Determine if input is complex (last dim == 2)
	isComplex := xShape[xShape.NDim()-1] == 2
	signalLen := xShape[axis]
	dftLen := signalLen
	if len(inputs) > 1 && inputs[1] != nil {
		dftLen = int(inputs[1].(*tensor.Dense[int64]).Data()[0])
	}

	outLen := dftLen
	if onesided {
		outLen = dftLen/2 + 1
	}

	// Build output shape: replace axis dim with outLen, ensure last dim is 2
	outShape := make(tensor.Shape, 0, xShape.NDim()+1)
	for i := 0; i < xShape.NDim(); i++ {
		if i == axis {
			outShape = append(outShape, outLen)
		} else if isComplex && i == xShape.NDim()-1 {
			outShape = append(outShape, 2)
		} else {
			outShape = append(outShape, xShape[i])
		}
	}
	if !isComplex {
		outShape = append(outShape, 2)
	}

	// Compute strides for the input
	xStrides := tensor.Strides(xShape)

	// Compute batch size (all dims except axis and optional complex dim)
	batchSize := 1
	for i := 0; i < xShape.NDim(); i++ {
		if i == axis {
			continue
		}
		if isComplex && i == xShape.NDim()-1 {
			continue
		}
		batchSize *= xShape[i]
	}

	outData := make([]float32, outShape.Size())
	outStrides := tensor.Strides(outShape)

	// For each batch element, compute DFT along axis
	sign := -1.0
	if inverse {
		sign = 1.0
	}

	// Iterate over all positions except the axis dimension
	batchDims := make([]int, 0)
	for i := 0; i < xShape.NDim(); i++ {
		if i != axis && !(isComplex && i == xShape.NDim()-1) {
			batchDims = append(batchDims, i)
		}
	}

	coords := make([]int, xShape.NDim())
	for bi := 0; bi < batchSize; bi++ {
		// Decode batch index into coordinates
		rem := bi
		for j := len(batchDims) - 1; j >= 0; j-- {
			d := batchDims[j]
			coords[d] = rem % xShape[d]
			rem /= xShape[d]
		}

		for k := 0; k < outLen; k++ {
			var sumR, sumI float64
			for n := 0; n < signalLen; n++ {
				angle := sign * 2.0 * math.Pi * float64(k) * float64(n) / float64(dftLen)
				cosA := math.Cos(angle)
				sinA := math.Sin(angle)

				coords[axis] = n
				var xR, xI float32
				if isComplex {
					coords[xShape.NDim()-1] = 0
					idxR := 0
					for d, c := range coords {
						idxR += c * xStrides[d]
					}
					xR = xData[idxR]
					coords[xShape.NDim()-1] = 1
					idxI := 0
					for d, c := range coords {
						idxI += c * xStrides[d]
					}
					xI = xData[idxI]
				} else {
					idx := 0
					for d, c := range coords {
						idx += c * xStrides[d]
					}
					xR = xData[idx]
				}
				sumR += float64(xR)*cosA - float64(xI)*sinA
				sumI += float64(xR)*sinA + float64(xI)*cosA
			}
			if inverse {
				sumR /= float64(dftLen)
				sumI /= float64(dftLen)
			}

			// Write output
			outCoords := make([]int, len(outShape))
			for d := range batchDims {
				outCoords[batchDims[d]] = coords[batchDims[d]]
			}
			outCoords[axis] = k
			cDim := len(outShape) - 1
			outCoords[cDim] = 0
			idxR := 0
			for d, c := range outCoords {
				idxR += c * outStrides[d]
			}
			outData[idxR] = float32(sumR)
			outCoords[cDim] = 1
			idxI := 0
			for d, c := range outCoords {
				idxI += c * outStrides[d]
			}
			outData[idxI] = float32(sumI)
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](outShape, outData)}, nil
}

// STFT: Short-Time Fourier Transform.
// inputs: signal[B,signal_length,1], frame_step, [window], [frame_length]
// output: [B, num_frames, fft_length/2+1, 2]
func opSTFT(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	onesided := node.GetAttrInt("onesided", 1) != 0
	signal := inputs[0].(*tensor.Dense[float32])
	frameStep := int(inputs[1].(*tensor.Dense[int64]).Data()[0])

	sShape := signal.Shape()
	batch := sShape[0]
	sigLen := sShape[1]
	sData := signal.Data()

	var window []float32
	frameLen := 0
	if len(inputs) > 2 && inputs[2] != nil {
		window = inputs[2].(*tensor.Dense[float32]).Data()
		frameLen = len(window)
	}
	if len(inputs) > 3 && inputs[3] != nil {
		frameLen = int(inputs[3].(*tensor.Dense[int64]).Data()[0])
	}
	if frameLen == 0 {
		return nil, fmt.Errorf("STFT: frame_length required")
	}

	numFrames := (sigLen - frameLen) / frameStep + 1
	fftLen := frameLen
	outFreqs := fftLen
	if onesided {
		outFreqs = fftLen/2 + 1
	}

	outShape := tensor.Shape{batch, numFrames, outFreqs, 2}
	outData := make([]float32, outShape.Size())

	for b := 0; b < batch; b++ {
		bOff := b * sigLen
		for f := 0; f < numFrames; f++ {
			start := f * frameStep
			for k := 0; k < outFreqs; k++ {
				var sumR, sumI float64
				for n := 0; n < frameLen; n++ {
					val := float64(sData[bOff+start+n])
					if window != nil {
						val *= float64(window[n])
					}
					angle := -2.0 * math.Pi * float64(k) * float64(n) / float64(fftLen)
					sumR += val * math.Cos(angle)
					sumI += val * math.Sin(angle)
				}
				idx := ((b*numFrames+f)*outFreqs+k)*2
				outData[idx] = float32(sumR)
				outData[idx+1] = float32(sumI)
			}
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](outShape, outData)}, nil
}

// MelWeightMatrix: generates a Mel filter bank matrix.
// inputs: num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz
// output: [num_spectrogram_bins, num_mel_bins]
func opMelWeightMatrix(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	outputDType := int(node.GetAttrInt("output_datatype", 1))
	_ = outputDType // only float32 for now

	numMelBins := int(inputs[0].(*tensor.Dense[int64]).Data()[0])
	dftLength := int(inputs[1].(*tensor.Dense[int64]).Data()[0])
	sampleRate := int(inputs[2].(*tensor.Dense[int64]).Data()[0])
	lowerEdgeHz := float64(inputs[3].(*tensor.Dense[float32]).Data()[0])
	upperEdgeHz := float64(inputs[4].(*tensor.Dense[float32]).Data()[0])

	numSpectBins := dftLength/2 + 1

	// Hz to Mel conversion
	hzToMel := func(hz float64) float64 {
		return 2595.0 * math.Log10(1.0+hz/700.0)
	}
	melToHz := func(mel float64) float64 {
		return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
	}

	lowerMel := hzToMel(lowerEdgeHz)
	upperMel := hzToMel(upperEdgeHz)

	// Create numMelBins+2 equally spaced points in mel space
	melPoints := make([]float64, numMelBins+2)
	for i := range melPoints {
		melPoints[i] = lowerMel + float64(i)*(upperMel-lowerMel)/float64(numMelBins+1)
	}

	// Convert mel points to frequency bins
	fftFreqs := make([]float64, numSpectBins)
	for i := range fftFreqs {
		fftFreqs[i] = float64(i) * float64(sampleRate) / float64(dftLength)
	}

	out := make([]float32, numSpectBins*numMelBins)
	for m := 0; m < numMelBins; m++ {
		startHz := melToHz(melPoints[m])
		centerHz := melToHz(melPoints[m+1])
		endHz := melToHz(melPoints[m+2])
		for f := 0; f < numSpectBins; f++ {
			freq := fftFreqs[f]
			var weight float64
			if freq >= startHz && freq <= centerHz && centerHz > startHz {
				weight = (freq - startHz) / (centerHz - startHz)
			} else if freq > centerHz && freq <= endHz && endHz > centerHz {
				weight = (endHz - freq) / (endHz - centerHz)
			}
			out[f*numMelBins+m] = float32(weight)
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](tensor.Shape{numSpectBins, numMelBins}, out)}, nil
}

// ── Detection ops ──

// RoiAlign: region of interest align.
// inputs: X[N,C,H,W], rois[num_rois,4], batch_indices[num_rois]
// output: [num_rois, C, output_height, output_width]
func opRoiAlign(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	mode := node.GetAttrString("mode", "avg")
	outputH := int(node.GetAttrInt("output_height", 1))
	outputW := int(node.GetAttrInt("output_width", 1))
	samplingRatio := int(node.GetAttrInt("sampling_ratio", 0))
	spatialScale := float64(node.GetAttrFloat("spatial_scale", 1.0))

	x := inputs[0].(*tensor.Dense[float32])
	rois := inputs[1].(*tensor.Dense[float32])
	batchIdx := inputs[2].(*tensor.Dense[int64])

	xShape := x.Shape()
	C, H, W := xShape[1], xShape[2], xShape[3]
	numRois := rois.Shape()[0]
	xData := x.Data()
	roisData := rois.Data()
	batchData := batchIdx.Data()

	outShape := tensor.Shape{numRois, C, outputH, outputW}
	outData := make([]float32, outShape.Size())

	bilinearSample := func(data []float32, off, H, W int, y, xx float64) float32 {
		if y < -1.0 || y > float64(H) || xx < -1.0 || xx > float64(W) {
			return 0
		}
		y = math.Max(y, 0)
		xx = math.Max(xx, 0)
		yLow := int(math.Floor(y))
		xLow := int(math.Floor(xx))
		yHigh := yLow + 1
		xHigh := xLow + 1
		if yLow >= H-1 {
			yLow = H - 1
			yHigh = yLow
		}
		if xLow >= W-1 {
			xLow = W - 1
			xHigh = xLow
		}
		ly := y - float64(yLow)
		lx := xx - float64(xLow)
		hy := 1.0 - ly
		hx := 1.0 - lx
		v1 := float64(data[off+yLow*W+xLow])
		v2 := float64(data[off+yLow*W+xHigh])
		v3 := float64(data[off+yHigh*W+xLow])
		v4 := float64(data[off+yHigh*W+xHigh])
		return float32(hy*hx*v1 + hy*lx*v2 + ly*hx*v3 + ly*lx*v4)
	}

	for r := 0; r < numRois; r++ {
		bIdx := int(batchData[r])
		x1 := float64(roisData[r*4]) * spatialScale
		y1 := float64(roisData[r*4+1]) * spatialScale
		x2 := float64(roisData[r*4+2]) * spatialScale
		y2 := float64(roisData[r*4+3]) * spatialScale
		roiH := y2 - y1
		roiW := x2 - x1
		if roiH <= 0 {
			roiH = 1
		}
		if roiW <= 0 {
			roiW = 1
		}

		binH := roiH / float64(outputH)
		binW := roiW / float64(outputW)
		sH := samplingRatio
		if sH <= 0 {
			sH = int(math.Ceil(binH))
		}
		sW := samplingRatio
		if sW <= 0 {
			sW = int(math.Ceil(binW))
		}

		for c := 0; c < C; c++ {
			cOff := (bIdx*C + c) * H * W
			for oh := 0; oh < outputH; oh++ {
				for ow := 0; ow < outputW; ow++ {
					var sum float64
					count := sH * sW
					for iy := 0; iy < sH; iy++ {
						yy := y1 + binH*(float64(oh)+(float64(iy)+0.5)/float64(sH))
						for ix := 0; ix < sW; ix++ {
							xx := x1 + binW*(float64(ow)+(float64(ix)+0.5)/float64(sW))
							v := bilinearSample(xData, cOff, H, W, yy, xx)
							if mode == "max" {
								if iy == 0 && ix == 0 {
									sum = float64(v)
								} else if float64(v) > sum {
									sum = float64(v)
								}
							} else {
								sum += float64(v)
							}
						}
					}
					if mode != "max" {
						sum /= float64(count)
					}
					outData[((r*C+c)*outputH+oh)*outputW+ow] = float32(sum)
				}
			}
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](outShape, outData)}, nil
}

// DeformConv: deformable convolution.
// inputs: X[N,C,H,W], W[OC,IC/g,KH,KW], offset[N,offset_g*KH*KW*2,OH,OW], [B], [mask]
func opDeformConv(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0].(*tensor.Dense[float32])
	w := inputs[1].(*tensor.Dense[float32])
	offset := inputs[2].(*tensor.Dense[float32])

	xShape := x.Shape()
	wShape := w.Shape()
	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	OC := wShape[0]
	KH, KW := wShape[2], wShape[3]

	group := int(node.GetAttrInt("group", 1))
	offsetGroup := int(node.GetAttrInt("offset_group", 1))
	strides := node.GetAttrInts("strides", []int64{1, 1})
	strideH, strideW := int(strides[0]), int(strides[1])
	dilations := node.GetAttrInts("dilations", []int64{1, 1})
	dilH, dilW := int(dilations[0]), int(dilations[1])
	pads := node.GetAttrInts("pads", []int64{0, 0, 0, 0})
	padTop, padLeft := int(pads[0]), int(pads[1])

	effKH := (KH-1)*dilH + 1
	effKW := (KW-1)*dilW + 1
	OH := (H + int(pads[0]) + int(pads[2]) - effKH) / strideH + 1
	OW := (W + int(pads[1]) + int(pads[3]) - effKW) / strideW + 1

	var biasData []float32
	if len(inputs) > 3 && inputs[3] != nil {
		biasData = inputs[3].(*tensor.Dense[float32]).Data()
	}
	var maskData []float32
	if len(inputs) > 4 && inputs[4] != nil {
		maskData = inputs[4].(*tensor.Dense[float32]).Data()
	}

	xData := x.Data()
	wData := w.Data()
	offData := offset.Data()

	icPerGroup := C / group
	ocPerGroup := OC / group
	icPerOffGroup := C / offsetGroup

	outShape := tensor.Shape{N, OC, OH, OW}
	outData := make([]float32, outShape.Size())

	for n := 0; n < N; n++ {
		for g := 0; g < group; g++ {
			for oc := 0; oc < ocPerGroup; oc++ {
				outC := g*ocPerGroup + oc
				for oh := 0; oh < OH; oh++ {
					for ow := 0; ow < OW; ow++ {
						var sum float64
						for ic := 0; ic < icPerGroup; ic++ {
							inC := g*icPerGroup + ic
							og := inC / icPerOffGroup // offset group index
							for kh := 0; kh < KH; kh++ {
								for kw := 0; kw < KW; kw++ {
									// Offset index
									offIdx := ((n*offsetGroup+og)*KH*KW*2 + (kh*KW+kw)*2) * OH * OW
									offY := float64(offData[offIdx+oh*OW+ow])
									offX := float64(offData[offIdx+OH*OW+oh*OW+ow])

									y := float64(oh*strideH-padTop+kh*dilH) + offY
									xx := float64(ow*strideW-padLeft+kw*dilW) + offX

									// Bilinear interpolation
									var val float64
									if y >= 0 && y < float64(H) && xx >= 0 && xx < float64(W) {
										yLow := int(math.Floor(y))
										xLow := int(math.Floor(xx))
										yHigh := yLow + 1
										xHigh := xLow + 1
										ly := y - float64(yLow)
										lx := xx - float64(xLow)
										cOff := (n*C + inC) * H * W
										v00 := float64(0)
										if yLow >= 0 && yLow < H && xLow >= 0 && xLow < W {
											v00 = float64(xData[cOff+yLow*W+xLow])
										}
										v01 := float64(0)
										if yLow >= 0 && yLow < H && xHigh >= 0 && xHigh < W {
											v01 = float64(xData[cOff+yLow*W+xHigh])
										}
										v10 := float64(0)
										if yHigh >= 0 && yHigh < H && xLow >= 0 && xLow < W {
											v10 = float64(xData[cOff+yHigh*W+xLow])
										}
										v11 := float64(0)
										if yHigh >= 0 && yHigh < H && xHigh >= 0 && xHigh < W {
											v11 = float64(xData[cOff+yHigh*W+xHigh])
										}
										val = (1-ly)*(1-lx)*v00 + (1-ly)*lx*v01 + ly*(1-lx)*v10 + ly*lx*v11
									}

									// Apply mask if present
									if maskData != nil {
										mIdx := ((n*offsetGroup+og)*KH*KW+(kh*KW+kw))*OH*OW + oh*OW + ow
										val *= float64(maskData[mIdx])
									}

									wIdx := ((outC*icPerGroup + ic) * KH + kh) * KW + kw
									sum += val * float64(wData[wIdx])
								}
							}
						}
						if biasData != nil {
							sum += float64(biasData[outC])
						}
						outData[((n*OC+outC)*OH+oh)*OW+ow] = float32(sum)
					}
				}
			}
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](outShape, outData)}, nil
}

// ── Transformer ops ──

// Attention: multi-head attention (com.microsoft domain).
// inputs: input(BxSxE), weights(Ex3E), bias(3E), [mask_index], [past], [extra_add]
// output: output(BxSxE), [present]
func opAttention(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	numHeads := int(node.GetAttrInt("num_heads", 1))
	input := inputs[0].(*tensor.Dense[float32])
	weights := inputs[1].(*tensor.Dense[float32])
	bias := inputs[2].(*tensor.Dense[float32])

	iShape := input.Shape()
	batch, seqLen, embed := iShape[0], iShape[1], iShape[2]
	headSize := embed / numHeads
	iData := input.Data()
	wData := weights.Data()
	bData := bias.Data()

	// QKV projection: [B,S,E] x [E,3E] + bias → [B,S,3E]
	qkv := make([]float32, batch*seqLen*3*embed)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * embed
			for j := 0; j < 3*embed; j++ {
				var sum float32
				for k := 0; k < embed; k++ {
					sum += iData[off+k] * wData[k*3*embed+j]
				}
				qkv[(b*seqLen+s)*3*embed+j] = sum + bData[j]
			}
		}
	}

	// Split into Q, K, V and compute scaled dot-product attention per head
	scale := float32(1.0 / math.Sqrt(float64(headSize)))
	outData := make([]float32, batch*seqLen*embed)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			// Compute attention scores
			scores := make([]float32, seqLen*seqLen)
			for qi := 0; qi < seqLen; qi++ {
				qOff := (b*seqLen+qi)*3*embed + h*headSize
				for ki := 0; ki < seqLen; ki++ {
					kOff := (b*seqLen+ki)*3*embed + embed + h*headSize
					var dot float32
					for d := 0; d < headSize; d++ {
						dot += qkv[qOff+d] * qkv[kOff+d]
					}
					scores[qi*seqLen+ki] = dot * scale
				}
			}
			// Softmax per query
			for qi := 0; qi < seqLen; qi++ {
				row := scores[qi*seqLen : qi*seqLen+seqLen]
				maxVal := row[0]
				for _, v := range row[1:] { if v > maxVal { maxVal = v } }
				var sum float32
				for i := range row { row[i] = float32(math.Exp(float64(row[i] - maxVal))); sum += row[i] }
				for i := range row { row[i] /= sum }
			}
			// Weighted sum of V
			for qi := 0; qi < seqLen; qi++ {
				for d := 0; d < headSize; d++ {
					var sum float32
					for vi := 0; vi < seqLen; vi++ {
						vOff := (b*seqLen+vi)*3*embed + 2*embed + h*headSize
						sum += scores[qi*seqLen+vi] * qkv[vOff+d]
					}
					outData[(b*seqLen+qi)*embed+h*headSize+d] = sum
				}
			}
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](tensor.Shape{batch, seqLen, embed}, outData)}, nil
}

// RotaryEmbedding: applies rotary position embedding.
// inputs: input(BxSxNxH), position_ids(Bx?), cos_cache(MxH/2), sin_cache(MxH/2)
// outputs: output (same shape as input)
func opRotaryEmbedding(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0].(*tensor.Dense[float32])
	posIds := inputs[1].(*tensor.Dense[int64])
	cosCache := inputs[2].(*tensor.Dense[float32])
	sinCache := inputs[3].(*tensor.Dense[float32])

	iShape := input.Shape()
	batch, seqLen, numHeads, headSize := iShape[0], iShape[1], iShape[2], iShape[3]
	halfH := headSize / 2

	iData := input.Data()
	pData := posIds.Data()
	cData := cosCache.Data()
	sData := sinCache.Data()
	out := make([]float32, len(iData))
	copy(out, iData)

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			posIdx := int(pData[b*seqLen+s])
			cosOff := posIdx * halfH
			sinOff := posIdx * halfH
			for h := 0; h < numHeads; h++ {
				base := ((b*seqLen+s)*numHeads+h)*headSize
				for d := 0; d < halfH; d++ {
					x0 := iData[base+d]
					x1 := iData[base+halfH+d]
					cos := cData[cosOff+d]
					sin := sData[sinOff+d]
					out[base+d] = x0*cos - x1*sin
					out[base+halfH+d] = x0*sin + x1*cos
				}
			}
		}
	}

	return []tensor.Tensor{tensor.NewDense[float32](iShape.Clone(), out)}, nil
}

// ── QLinear ops ──

// QLinearConv: quantized convolution.
// inputs: x, x_scale, x_zero_point, w, w_scale, w_zero_point, [bias], [y_scale, y_zero_point]
func opQLinearConv(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Dequantize x and w, run conv, quantize output
	xDeq, err := opDequantizeLinear(node, []tensor.Tensor{inputs[0], inputs[1], inputs[2]})
	if err != nil {
		return nil, fmt.Errorf("QLinearConv: dequant x: %w", err)
	}
	wDeq, err := opDequantizeLinear(node, []tensor.Tensor{inputs[3], inputs[4], inputs[5]})
	if err != nil {
		return nil, fmt.Errorf("QLinearConv: dequant w: %w", err)
	}

	convInputs := []tensor.Tensor{xDeq[0], wDeq[0]}
	if len(inputs) > 8 && inputs[8] != nil {
		convInputs = append(convInputs, inputs[8])
	} else {
		convInputs = append(convInputs, nil)
	}

	result, err := opConv(node, convInputs)
	if err != nil {
		return nil, fmt.Errorf("QLinearConv: conv: %w", err)
	}

	// Quantize output if y_scale/y_zero_point provided
	if len(inputs) > 7 && inputs[6] != nil && inputs[7] != nil {
		return opQuantizeLinear(node, []tensor.Tensor{result[0], inputs[6], inputs[7]})
	}
	return result, nil
}

// QLinearMatMul: quantized matmul.
// inputs: a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
func opQLinearMatMul(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	aDeq, err := opDequantizeLinear(node, []tensor.Tensor{inputs[0], inputs[1], inputs[2]})
	if err != nil {
		return nil, fmt.Errorf("QLinearMatMul: dequant a: %w", err)
	}
	bDeq, err := opDequantizeLinear(node, []tensor.Tensor{inputs[3], inputs[4], inputs[5]})
	if err != nil {
		return nil, fmt.Errorf("QLinearMatMul: dequant b: %w", err)
	}

	result, err := opMatMul(nil, []tensor.Tensor{aDeq[0], bDeq[0]})
	if err != nil {
		return nil, fmt.Errorf("QLinearMatMul: matmul: %w", err)
	}

	if len(inputs) > 7 && inputs[6] != nil && inputs[7] != nil {
		return opQuantizeLinear(node, []tensor.Tensor{result[0], inputs[6], inputs[7]})
	}
	return result, nil
}

// ── Scan ──

func opScan(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return nil, fmt.Errorf("Scan: handled by engine runtime")
}

// ── Loop ──

func opLoop(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return nil, fmt.Errorf("Loop: handled by engine runtime")
}
