package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Comparison ──

func opLessOrEqual(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	outShape, err := tensor.BroadcastShape(a.Shape(), b.Shape())
	if err != nil { return nil, fmt.Errorf("LessOrEqual: %w", err) }
	size := outShape.Size()
	data := make([]uint8, size)
	switch at := a.(type) {
	case *tensor.Dense[float32]:
		bt := b.(*tensor.Dense[float32])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] <= bd[i] { data[i] = 1 } }
		} else {
			os := tensor.Strides(outShape); as := tensor.Strides(at.Shape()); bs := tensor.Strides(bt.Shape())
			for i := 0; i < size; i++ {
				if ad[tensor.BroadcastIndex(i, outShape, at.Shape(), os, as)] <= bd[tensor.BroadcastIndex(i, outShape, bt.Shape(), os, bs)] { data[i] = 1 }
			}
		}
	case *tensor.Dense[int64]:
		bt := b.(*tensor.Dense[int64])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] <= bd[i] { data[i] = 1 } }
		}
	case *tensor.Dense[float64]:
		bt := b.(*tensor.Dense[float64])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] <= bd[i] { data[i] = 1 } }
		}
	case *tensor.Dense[int32]:
		bt := b.(*tensor.Dense[int32])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] <= bd[i] { data[i] = 1 } }
		}
	case *tensor.Dense[uint8]:
		bt := b.(*tensor.Dense[uint8])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] <= bd[i] { data[i] = 1 } }
		}
	default:
		return nil, fmt.Errorf("LessOrEqual: unsupported type %T", a)
	}
	return []tensor.Tensor{tensor.NewDense[uint8](outShape, data)}, nil
}

// ── Activation extras ──

func opGelu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Gelu: x * 0.5 * (1 + erf(x / sqrt(2)))
	invSqrt2 := 1.0 / math.Sqrt(2.0)
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(0.5 * float64(v) * (1.0 + math.Erf(float64(v)*invSqrt2)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = 0.5 * v * (1.0 + math.Erf(v*invSqrt2))
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Gelu: unsupported type %T", inputs[0])
	}
}

func opMish(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			sp := math.Log(1.0 + math.Exp(float64(v)))
			data[i] = float32(float64(v) * math.Tanh(sp))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Mish: unsupported type %T", inputs[0])
	}
}

func opSelu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	alpha := float64(node.GetAttrFloat("alpha", 1.67326319217681884765625))
	gamma := float64(node.GetAttrFloat("gamma", 1.05070102214813232421875))
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			if v > 0 { data[i] = float32(gamma * float64(v)) } else { data[i] = float32(gamma * (alpha*math.Exp(float64(v)) - alpha)) }
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Selu: unsupported type %T", inputs[0])
	}
}

func opSoftplus(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Log(1.0 + math.Exp(float64(v)))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Softplus: unsupported type %T", inputs[0])
	}
}

func opSoftsign(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = v / (1.0 + float32(math.Abs(float64(v)))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Softsign: unsupported type %T", inputs[0])
	}
}

func opLogSoftmax(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// LogSoftmax = Log(Softmax(x))
	axis := int(node.GetAttrInt("axis", -1))
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		sm, err := softmaxDense(t, axis)
		if err != nil { return nil, fmt.Errorf("LogSoftmax: %w", err) }
		data := sm.Data()
		for i, v := range data { data[i] = float32(math.Log(float64(v))) }
		return []tensor.Tensor{sm}, nil
	default:
		return nil, fmt.Errorf("LogSoftmax: unsupported type %T", inputs[0])
	}
}

// ── Celu ──

func opCelu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	alpha := float64(node.GetAttrFloat("alpha", 1.0))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out := make([]float32, x.Len())
		for i, v := range x.Data() {
			fv := float64(v)
			out[i] = float32(math.Max(0, fv) + math.Min(0, alpha*(math.Exp(fv/alpha)-1)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("Celu: unsupported type %T", inputs[0])
	}
}

// ── Hardmax ──

func opHardmax(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := int(node.GetAttrInt("axis", -1))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		shape := x.Shape()
		ndim := shape.NDim()
		if axis < 0 { axis += ndim }
		outerSize := 1; for d := 0; d < axis; d++ { outerSize *= shape[d] }
		axisSize := shape[axis]
		innerSize := 1; for d := axis + 1; d < ndim; d++ { innerSize *= shape[d] }
		data := x.Data()
		out := make([]float32, len(data))
		for o := 0; o < outerSize; o++ {
			for i := 0; i < innerSize; i++ {
				bestIdx := 0
				bestVal := data[o*axisSize*innerSize+i]
				for a := 1; a < axisSize; a++ {
					v := data[o*axisSize*innerSize+a*innerSize+i]
					if v > bestVal { bestVal = v; bestIdx = a }
				}
				out[o*axisSize*innerSize+bestIdx*innerSize+i] = 1
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](shape.Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("Hardmax: unsupported type %T", inputs[0])
	}
}

// ── Shrink ──

func opShrink(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	bias := float64(node.GetAttrFloat("bias", 0))
	lambd := float64(node.GetAttrFloat("lambd", 0.5))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out := make([]float32, x.Len())
		for i, v := range x.Data() {
			fv := float64(v)
			if fv < -lambd {
				out[i] = float32(fv + bias)
			} else if fv > lambd {
				out[i] = float32(fv - bias)
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("Shrink: unsupported type %T", inputs[0])
	}
}

// ── ThresholdedRelu ──

func opThresholdedRelu(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	alpha := float64(node.GetAttrFloat("alpha", 1.0))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out := make([]float32, x.Len())
		for i, v := range x.Data() {
			if float64(v) > alpha { out[i] = v }
		}
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("ThresholdedRelu: unsupported type %T", inputs[0])
	}
}
