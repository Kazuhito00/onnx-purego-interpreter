package ops

import (
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// LSTM op — Long Short-Term Memory (recurrent).
// Inputs: X[seq_len, batch, input_size], W[num_dir, 4*hidden, input], R[num_dir, 4*hidden, hidden],
//         B[num_dir, 8*hidden] (optional), sequence_lens (optional),
//         initial_h[num_dir, batch, hidden] (optional), initial_c[num_dir, batch, hidden] (optional)
// Outputs: Y[seq_len, num_dir, batch, hidden], Y_h[num_dir, batch, hidden], Y_c[num_dir, batch, hidden]
func opLSTM(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	direction := node.GetAttrString("direction", "forward")
	hiddenSize := int(node.GetAttrInt("hidden_size", 0))
	_ = direction // only forward supported for now

	x := inputs[0].(*tensor.Dense[float32])
	w := inputs[1].(*tensor.Dense[float32])
	r := inputs[2].(*tensor.Dense[float32])

	xShape := x.Shape() // [seq_len, batch, input_size]
	seqLen := xShape[0]
	batch := xShape[1]
	inputSize := xShape[2]

	if hiddenSize == 0 {
		hiddenSize = r.Shape()[2] // R is [num_dir, 4*hidden, hidden]
	}

	// Bias (optional)
	var biasData []float32
	if len(inputs) > 3 && inputs[3] != nil {
		biasData = inputs[3].(*tensor.Dense[float32]).Data()
	}

	// Initial hidden state (optional)
	var hData []float32
	if len(inputs) > 5 && inputs[5] != nil {
		h := inputs[5].(*tensor.Dense[float32])
		hData = make([]float32, h.Len())
		copy(hData, h.Data())
	} else {
		hData = make([]float32, batch*hiddenSize)
	}

	// Initial cell state (optional)
	var cData []float32
	if len(inputs) > 6 && inputs[6] != nil {
		c := inputs[6].(*tensor.Dense[float32])
		cData = make([]float32, c.Len())
		copy(cData, c.Data())
	} else {
		cData = make([]float32, batch*hiddenSize)
	}

	wData := w.Data() // [1, 4*hidden, input_size]
	rData := r.Data() // [1, 4*hidden, hidden]
	xData := x.Data()

	H := hiddenSize
	gate4H := 4 * H

	// Output: Y[seq_len, 1, batch, hidden]
	yData := make([]float32, seqLen*batch*H)

	for t := 0; t < seqLen; t++ {
		for b := 0; b < batch; b++ {
			xOff := t*batch*inputSize + b*inputSize
			hOff := b * H

			// Compute gates: i, o, f, c = W*x + R*h + bias
			gates := make([]float32, gate4H)

			// W * x_t: gates[g] += sum_j W[g,j] * x[t,b,j]
			for g := 0; g < gate4H; g++ {
				var sum float32
				wOff := g * inputSize // W is [1, 4H, input_size], skip dir=0
				for j := 0; j < inputSize; j++ {
					sum += wData[wOff+j] * xData[xOff+j]
				}
				gates[g] = sum
			}

			// R * h_{t-1}: gates[g] += sum_j R[g,j] * h[b,j]
			for g := 0; g < gate4H; g++ {
				var sum float32
				rOff := g * H
				for j := 0; j < H; j++ {
					sum += rData[rOff+j] * hData[hOff+j]
				}
				gates[g] += sum
			}

			// Add bias (Wb + Rb): bias is [1, 8*H] = [Wb_i,Wb_o,Wb_f,Wb_c, Rb_i,Rb_o,Rb_f,Rb_c]
			if biasData != nil {
				for g := 0; g < gate4H; g++ {
					gates[g] += biasData[g] + biasData[gate4H+g]
				}
			}

			// ONNX LSTM gate order: i, o, f, c (iofc)
			for h := 0; h < H; h++ {
				it := sigmoid32(gates[0*H+h])     // input gate
				ot := sigmoid32(gates[1*H+h])     // output gate
				ft := sigmoid32(gates[2*H+h])     // forget gate
				ct := float32(math.Tanh(float64(gates[3*H+h]))) // cell candidate

				cellIdx := hOff + h
				cData[cellIdx] = ft*cData[cellIdx] + it*ct
				hData[hOff+h] = ot * float32(math.Tanh(float64(cData[cellIdx])))
			}

			// Store output
			yOff := t*batch*H + b*H
			copy(yData[yOff:yOff+H], hData[hOff:hOff+H])
		}
	}

	// Build output tensors
	Y := tensor.NewDense[float32](tensor.Shape{seqLen, 1, batch, H}, yData)
	Yh := tensor.NewDense[float32](tensor.Shape{1, batch, H}, hData)
	Yc := tensor.NewDense[float32](tensor.Shape{1, batch, H}, cData)

	return []tensor.Tensor{Y, Yh, Yc}, nil
}

func sigmoid32(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// GRU op — Gated Recurrent Unit.
// Inputs: X[seq_len, batch, input_size], W[num_dir, 3*hidden, input], R[num_dir, 3*hidden, hidden],
//         B[num_dir, 6*hidden] (optional), sequence_lens (optional),
//         initial_h[num_dir, batch, hidden] (optional)
// Outputs: Y[seq_len, num_dir, batch, hidden], Y_h[num_dir, batch, hidden]
func opGRU(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	hiddenSize := int(node.GetAttrInt("hidden_size", 0))
	linearBeforeReset := node.GetAttrInt("linear_before_reset", 0) != 0

	x := inputs[0].(*tensor.Dense[float32])
	w := inputs[1].(*tensor.Dense[float32])
	r := inputs[2].(*tensor.Dense[float32])

	xShape := x.Shape()
	seqLen := xShape[0]
	batch := xShape[1]
	inputSize := xShape[2]

	if hiddenSize == 0 {
		hiddenSize = r.Shape()[2]
	}
	H := hiddenSize
	gate3H := 3 * H

	var biasData []float32
	if len(inputs) > 3 && inputs[3] != nil {
		biasData = inputs[3].(*tensor.Dense[float32]).Data()
	}

	numDir := w.Shape()[0] // 1 = forward, 2 = bidirectional

	var hData []float32
	if len(inputs) > 5 && inputs[5] != nil {
		h := inputs[5].(*tensor.Dense[float32])
		hData = make([]float32, h.Len())
		copy(hData, h.Data())
	} else {
		hData = make([]float32, numDir*batch*H)
	}

	wData := w.Data()
	rData := r.Data()
	xData := x.Data()

	wDirStride := gate3H * inputSize  // stride per direction in W
	rDirStride := gate3H * H          // stride per direction in R
	bDirStride := 2 * gate3H          // stride per direction in B

	yData := make([]float32, seqLen*numDir*batch*H)

	for dir := 0; dir < numDir; dir++ {
		wOff := dir * wDirStride
		rOff := dir * rDirStride
		bOff := 0
		if biasData != nil {
			bOff = dir * bDirStride
		}
		hOff0 := dir * batch * H // offset into hData for this direction

		// Forward: t=0..seqLen-1; Backward (dir=1): t=seqLen-1..0
		for step := 0; step < seqLen; step++ {
			t := step
			if dir == 1 {
				t = seqLen - 1 - step
			}

			for b := 0; b < batch; b++ {
				xOff := t*batch*inputSize + b*inputSize
				hIdx := hOff0 + b*H

				gates := make([]float32, gate3H)
				for g := 0; g < gate3H; g++ {
					var sum float32
					for j := 0; j < inputSize; j++ {
						sum += wData[wOff+g*inputSize+j] * xData[xOff+j]
					}
					gates[g] = sum
				}
				rGates := make([]float32, gate3H)
				for g := 0; g < gate3H; g++ {
					var sum float32
					for j := 0; j < H; j++ {
						sum += rData[rOff+g*H+j] * hData[hIdx+j]
					}
					rGates[g] = sum
				}

				if biasData != nil {
					for g := 0; g < gate3H; g++ {
						gates[g] += biasData[bOff+g]
						rGates[g] += biasData[bOff+gate3H+g]
					}
				}

				for h := 0; h < H; h++ {
					zt := sigmoid32(gates[0*H+h] + rGates[0*H+h])
					rt := sigmoid32(gates[1*H+h] + rGates[1*H+h])

					var ht float32
					if linearBeforeReset {
						ht = float32(math.Tanh(float64(gates[2*H+h] + rt*rGates[2*H+h])))
					} else {
						var sum float32
						for j := 0; j < H; j++ {
							sum += rData[rOff+(2*H+h)*H+j] * (rt * hData[hIdx+j])
						}
						if biasData != nil {
							sum += biasData[bOff+gate3H+2*H+h]
						}
						ht = float32(math.Tanh(float64(gates[2*H+h] + sum)))
					}

					hData[hIdx+h] = (1-zt)*ht + zt*hData[hIdx+h]
				}

				yOff := t*numDir*batch*H + dir*batch*H + b*H
				copy(yData[yOff:yOff+H], hData[hIdx:hIdx+H])
			}
		}
	}

	Y := tensor.NewDense[float32](tensor.Shape{seqLen, numDir, batch, H}, yData)
	Yh := tensor.NewDense[float32](tensor.Shape{numDir, batch, H}, hData)
	return []tensor.Tensor{Y, Yh}, nil
}
