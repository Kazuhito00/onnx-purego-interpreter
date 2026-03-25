package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func clampF32(v, lo, hi float32) float32 {
	if v < lo { return lo }
	if v > hi { return hi }
	return v
}

func opLog(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Log(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Log(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Log: unsupported type %T", inputs[0])
	}
}

func opAbs(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, x.Len())
		for i, v := range x.Data() {
			if v < 0 {
				v = -v
			}
			data[i] = v
		}
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, x.Len())
		for i, v := range x.Data() {
			if v < 0 {
				v = -v
			}
			data[i] = v
		}
		return []tensor.Tensor{tensor.NewDense[float64](x.Shape().Clone(), data)}, nil
	case *tensor.Dense[int32]:
		data := make([]int32, x.Len())
		for i, v := range x.Data() {
			if v < 0 {
				v = -v
			}
			data[i] = v
		}
		return []tensor.Tensor{tensor.NewDense[int32](x.Shape().Clone(), data)}, nil
	case *tensor.Dense[int64]:
		data := make([]int64, x.Len())
		for i, v := range x.Data() {
			if v < 0 {
				v = -v
			}
			data[i] = v
		}
		return []tensor.Tensor{tensor.NewDense[int64](x.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Abs: unsupported type %T", inputs[0])
	}
}

func opReciprocal(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, x.Len())
		for i, v := range x.Data() {
			data[i] = 1 / v
		}
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, x.Len())
		for i, v := range x.Data() {
			data[i] = 1 / v
		}
		return []tensor.Tensor{tensor.NewDense[float64](x.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Reciprocal: unsupported type %T", inputs[0])
	}
}

func opSqrt(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Sqrt(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Sqrt(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Sqrt: unsupported type %T", inputs[0])
	}
}

func opNeg(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = -v
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = -v
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[int64]:
		data := make([]int64, t.Len())
		for i, v := range t.Data() {
			data[i] = -v
		}
		return []tensor.Tensor{tensor.NewDense[int64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Neg: unsupported type %T", inputs[0])
	}
}

func opErf(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Erf(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Erf(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Erf: unsupported type %T", inputs[0])
	}
}

func opPow(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := dispatchBinaryOp(inputs,
		func(a, b float32) float32 { return float32(math.Pow(float64(a), float64(b))) },
		func(a, b float64) float64 { return math.Pow(a, b) },
		func(a, b int32) int32 { return int32(math.Pow(float64(a), float64(b))) },
		func(a, b int64) int64 { return int64(math.Pow(float64(a), float64(b))) },
	)
	if err != nil {
		return nil, fmt.Errorf("Pow: %w", err)
	}
	return []tensor.Tensor{out}, nil
}

func opCast(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	to := int(node.GetAttrInt("to", 1))
	// opset 19+: saturate attribute (default 1 = saturate/clamp)
	saturate := node.GetAttrInt("saturate", 1) != 0
	src := inputs[0]
	srcLen := src.Len()
	shape := src.Shape().Clone()
	_ = saturate // used in integer cast branches below

	switch to {
	case 1: // FLOAT
		data := make([]float32, srcLen)
		switch s := src.(type) {
		case *tensor.Dense[float32]:
			copy(data, s.Data())
		case *tensor.Dense[float64]:
			for i, v := range s.Data() { data[i] = float32(v) }
		case *tensor.Dense[int32]:
			for i, v := range s.Data() { data[i] = float32(v) }
		case *tensor.Dense[int64]:
			for i, v := range s.Data() { data[i] = float32(v) }
		case *tensor.Dense[uint8]:
			for i, v := range s.Data() { data[i] = float32(v) }
		case *tensor.Dense[int8]:
			for i, v := range s.Data() { data[i] = float32(v) }
		default:
			return nil, fmt.Errorf("Cast: unsupported source type %T", src)
		}
		return []tensor.Tensor{tensor.NewDense[float32](shape, data)}, nil
	case 3: // INT8
		data := make([]int8, srcLen)
		switch s := src.(type) {
		case *tensor.Dense[float32]:
			for i, v := range s.Data() {
				if saturate { v = clampF32(v, -128, 127) }
				data[i] = int8(v)
			}
		case *tensor.Dense[int32]:
			for i, v := range s.Data() { data[i] = int8(v) }
		case *tensor.Dense[int64]:
			for i, v := range s.Data() { data[i] = int8(v) }
		case *tensor.Dense[uint8]:
			for i, v := range s.Data() { data[i] = int8(v) }
		case *tensor.Dense[int8]:
			copy(data, s.Data())
		default:
			return nil, fmt.Errorf("Cast: unsupported source type %T", src)
		}
		return []tensor.Tensor{tensor.NewDense[int8](shape, data)}, nil
	case 6: // INT32
		data := make([]int32, srcLen)
		switch s := src.(type) {
		case *tensor.Dense[float32]:
			for i, v := range s.Data() {
				if saturate { v = clampF32(v, -2147483648, 2147483647) }
				data[i] = int32(v)
			}
		case *tensor.Dense[float64]:
			for i, v := range s.Data() {
				if saturate { v = float64(clampF32(float32(v), -2147483648, 2147483647)) }
				data[i] = int32(v)
			}
		case *tensor.Dense[int32]:
			copy(data, s.Data())
		case *tensor.Dense[int64]:
			for i, v := range s.Data() { data[i] = int32(v) }
		case *tensor.Dense[int8]:
			for i, v := range s.Data() { data[i] = int32(v) }
		default:
			return nil, fmt.Errorf("Cast: unsupported source type %T", src)
		}
		return []tensor.Tensor{tensor.NewDense[int32](shape, data)}, nil
	case 7: // INT64
		data := make([]int64, srcLen)
		switch s := src.(type) {
		case *tensor.Dense[float32]:
			for i, v := range s.Data() { data[i] = int64(v) }
		case *tensor.Dense[float64]:
			for i, v := range s.Data() { data[i] = int64(v) }
		case *tensor.Dense[int32]:
			for i, v := range s.Data() { data[i] = int64(v) }
		case *tensor.Dense[int64]:
			copy(data, s.Data())
		case *tensor.Dense[int8]:
			for i, v := range s.Data() { data[i] = int64(v) }
		default:
			return nil, fmt.Errorf("Cast: unsupported source type %T", src)
		}
		return []tensor.Tensor{tensor.NewDense[int64](shape, data)}, nil
	case 9: // BOOL → uint8
		data := make([]uint8, srcLen)
		switch s := src.(type) {
		case *tensor.Dense[float32]:
			for i, v := range s.Data() { if v != 0 { data[i] = 1 } }
		case *tensor.Dense[int64]:
			for i, v := range s.Data() { if v != 0 { data[i] = 1 } }
		case *tensor.Dense[int8]:
			for i, v := range s.Data() { if v != 0 { data[i] = 1 } }
		case *tensor.Dense[uint8]:
			copy(data, s.Data())
		default:
			return nil, fmt.Errorf("Cast: unsupported source type %T for bool", src)
		}
		return []tensor.Tensor{tensor.NewDense[uint8](shape, data)}, nil
	case 11: // DOUBLE
		data := make([]float64, srcLen)
		switch s := src.(type) {
		case *tensor.Dense[float32]:
			for i, v := range s.Data() { data[i] = float64(v) }
		case *tensor.Dense[float64]:
			copy(data, s.Data())
		case *tensor.Dense[int64]:
			for i, v := range s.Data() { data[i] = float64(v) }
		case *tensor.Dense[int8]:
			for i, v := range s.Data() { data[i] = float64(v) }
		default:
			return nil, fmt.Errorf("Cast: unsupported source type %T", src)
		}
		return []tensor.Tensor{tensor.NewDense[float64](shape, data)}, nil
	default:
		return nil, fmt.Errorf("Cast: unsupported target dtype %d", to)
	}
}

func opEqual(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	outShape, err := tensor.BroadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, fmt.Errorf("Equal: %w", err)
	}
	size := outShape.Size()
	data := make([]uint8, size)

	switch at := a.(type) {
	case *tensor.Dense[float32]:
		bt := b.(*tensor.Dense[float32])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad {
				if ad[i] == bd[i] { data[i] = 1 }
			}
		} else {
			outStrides := tensor.Strides(outShape)
			aStrides := tensor.Strides(at.Shape())
			bStrides := tensor.Strides(bt.Shape())
			for i := 0; i < size; i++ {
				ai := tensor.BroadcastIndex(i, outShape, at.Shape(), outStrides, aStrides)
				bi := tensor.BroadcastIndex(i, outShape, bt.Shape(), outStrides, bStrides)
				if ad[ai] == bd[bi] { data[i] = 1 }
			}
		}
	case *tensor.Dense[int64]:
		bt := b.(*tensor.Dense[int64])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad {
				if ad[i] == bd[i] { data[i] = 1 }
			}
		}
	case *tensor.Dense[int32]:
		bt := b.(*tensor.Dense[int32])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad {
				if ad[i] == bd[i] { data[i] = 1 }
			}
		} else {
			outStrides := tensor.Strides(outShape)
			aStrides := tensor.Strides(at.Shape())
			bStrides := tensor.Strides(bt.Shape())
			for i := 0; i < size; i++ {
				ai := tensor.BroadcastIndex(i, outShape, at.Shape(), outStrides, aStrides)
				bi := tensor.BroadcastIndex(i, outShape, bt.Shape(), outStrides, bStrides)
				if ad[ai] == bd[bi] { data[i] = 1 }
			}
		}
	case *tensor.Dense[uint8]:
		bt := b.(*tensor.Dense[uint8])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad {
				if ad[i] == bd[i] { data[i] = 1 }
			}
		} else {
			outStrides := tensor.Strides(outShape)
			aStrides := tensor.Strides(at.Shape())
			bStrides := tensor.Strides(bt.Shape())
			for i := 0; i < size; i++ {
				ai := tensor.BroadcastIndex(i, outShape, at.Shape(), outStrides, aStrides)
				bi := tensor.BroadcastIndex(i, outShape, bt.Shape(), outStrides, bStrides)
				if ad[ai] == bd[bi] { data[i] = 1 }
			}
		}
	case *tensor.Dense[int8]:
		bt := b.(*tensor.Dense[int8])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] == bd[i] { data[i] = 1 } }
		}
	default:
		return nil, fmt.Errorf("Equal: unsupported type %T", a)
	}
	return []tensor.Tensor{tensor.NewDense[uint8](outShape, data)}, nil
}

func opWhere(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	cond := inputs[0].(*tensor.Dense[uint8])
	switch xt := inputs[1].(type) {
	case *tensor.Dense[float32]:
		return whereDense(cond, xt, inputs[2].(*tensor.Dense[float32]))
	case *tensor.Dense[float64]:
		return whereDense(cond, xt, inputs[2].(*tensor.Dense[float64]))
	case *tensor.Dense[int32]:
		return whereDense(cond, xt, inputs[2].(*tensor.Dense[int32]))
	case *tensor.Dense[int64]:
		return whereDense(cond, xt, inputs[2].(*tensor.Dense[int64]))
	case *tensor.Dense[uint8]:
		return whereDense(cond, xt, inputs[2].(*tensor.Dense[uint8]))
	case *tensor.Dense[int8]:
		return whereDense(cond, xt, inputs[2].(*tensor.Dense[int8]))
	default:
		return nil, fmt.Errorf("Where: unsupported type %T", inputs[1])
	}
}

func whereDense[T tensor.Numeric](cond *tensor.Dense[uint8], x, y *tensor.Dense[T]) ([]tensor.Tensor, error) {
	outShape, _ := tensor.BroadcastShape(cond.Shape(), x.Shape())
	outShape, _ = tensor.BroadcastShape(outShape, y.Shape())
	size := outShape.Size()
	data := make([]T, size)
	cd, xd, yd := cond.Data(), x.Data(), y.Data()
	if cond.Shape().Equal(x.Shape()) && x.Shape().Equal(y.Shape()) {
		for i := range data {
			if cd[i] != 0 { data[i] = xd[i] } else { data[i] = yd[i] }
		}
	} else {
		outStrides := tensor.Strides(outShape)
		cStrides := tensor.Strides(cond.Shape())
		xStrides := tensor.Strides(x.Shape())
		yStrides := tensor.Strides(y.Shape())
		for i := 0; i < size; i++ {
			ci := tensor.BroadcastIndex(i, outShape, cond.Shape(), outStrides, cStrides)
			xi := tensor.BroadcastIndex(i, outShape, x.Shape(), outStrides, xStrides)
			yi := tensor.BroadcastIndex(i, outShape, y.Shape(), outStrides, yStrides)
			if cd[ci] != 0 { data[i] = xd[xi] } else { data[i] = yd[yi] }
		}
	}
	return []tensor.Tensor{tensor.NewDense[T](outShape, data)}, nil
}

func opExpand(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	shapeTensor := inputs[1].(*tensor.Dense[int64])
	targetShape := make(tensor.Shape, shapeTensor.Len())
	for i, v := range shapeTensor.Data() {
		targetShape[i] = int(v)
	}

	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{expandDense(t, targetShape)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{expandDense(t, targetShape)}, nil
	case *tensor.Dense[int64]:
		return []tensor.Tensor{expandDense(t, targetShape)}, nil
	case *tensor.Dense[int32]:
		return []tensor.Tensor{expandDense(t, targetShape)}, nil
	case *tensor.Dense[uint8]:
		return []tensor.Tensor{expandDense(t, targetShape)}, nil
	case *tensor.Dense[int8]:
		return []tensor.Tensor{expandDense(t, targetShape)}, nil
	default:
		return nil, fmt.Errorf("Expand: unsupported type %T", inputs[0])
	}
}

func expandDense[T tensor.Numeric](t *tensor.Dense[T], targetShape tensor.Shape) *tensor.Dense[T] {
	outShape, _ := tensor.BroadcastShape(t.Shape(), targetShape)
	size := outShape.Size()
	data := make([]T, size)
	src := t.Data()
	if t.Len() == 1 {
		v := src[0]
		for i := range data { data[i] = v }
	} else {
		outStrides := tensor.Strides(outShape)
		inStrides := tensor.Strides(t.Shape())
		for i := 0; i < size; i++ {
			data[i] = src[tensor.BroadcastIndex(i, outShape, t.Shape(), outStrides, inStrides)]
		}
	}
	return tensor.NewDense[T](outShape, data)
}

func opFloor(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Floor(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Floor(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Floor: unsupported type %T", inputs[0])
	}
}

func opExp(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Exp(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Exp(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Exp: unsupported type %T", inputs[0])
	}
}


func opSin(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Sin(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Sin(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Sin: unsupported type %T", inputs[0])
	}
}

func opCos(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			data[i] = float32(math.Cos(float64(v)))
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			data[i] = math.Cos(v)
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Cos: unsupported type %T", inputs[0])
	}
}

func opGreaterOrEqual(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	outShape, err := tensor.BroadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, fmt.Errorf("GreaterOrEqual: %w", err)
	}
	size := outShape.Size()
	data := make([]uint8, size)
	switch at := a.(type) {
	case *tensor.Dense[float32]:
		bt := b.(*tensor.Dense[float32])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad {
				if ad[i] >= bd[i] {
					data[i] = 1
				}
			}
		} else {
			outStrides := tensor.Strides(outShape)
			aStrides := tensor.Strides(at.Shape())
			bStrides := tensor.Strides(bt.Shape())
			for i := 0; i < size; i++ {
				ai := tensor.BroadcastIndex(i, outShape, at.Shape(), outStrides, aStrides)
				bi := tensor.BroadcastIndex(i, outShape, bt.Shape(), outStrides, bStrides)
				if ad[ai] >= bd[bi] {
					data[i] = 1
				}
			}
		}
	case *tensor.Dense[int64]:
		bt := b.(*tensor.Dense[int64])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] >= bd[i] { data[i] = 1 } }
		}
	case *tensor.Dense[float64]:
		bt := b.(*tensor.Dense[float64])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] >= bd[i] { data[i] = 1 } }
		}
	case *tensor.Dense[int32]:
		bt := b.(*tensor.Dense[int32])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] >= bd[i] { data[i] = 1 } }
		}
	case *tensor.Dense[uint8]:
		bt := b.(*tensor.Dense[uint8])
		ad, bd := at.Data(), bt.Data()
		if at.Shape().Equal(bt.Shape()) {
			for i := range ad { if ad[i] >= bd[i] { data[i] = 1 } }
		}
	default:
		return nil, fmt.Errorf("GreaterOrEqual: unsupported type %T", a)
	}
	return []tensor.Tensor{tensor.NewDense[uint8](outShape, data)}, nil
}

func opOr(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	a := inputs[0].(*tensor.Dense[uint8])
	b := inputs[1].(*tensor.Dense[uint8])
	outShape, _ := tensor.BroadcastShape(a.Shape(), b.Shape())
	size := outShape.Size()
	data := make([]uint8, size)
	ad, bd := a.Data(), b.Data()
	if a.Shape().Equal(b.Shape()) {
		for i := range ad {
			if ad[i] != 0 || bd[i] != 0 {
				data[i] = 1
			}
		}
	} else {
		outStrides := tensor.Strides(outShape)
		aStrides := tensor.Strides(a.Shape())
		bStrides := tensor.Strides(b.Shape())
		for i := 0; i < size; i++ {
			ai := tensor.BroadcastIndex(i, outShape, a.Shape(), outStrides, aStrides)
			bi := tensor.BroadcastIndex(i, outShape, b.Shape(), outStrides, bStrides)
			if ad[ai] != 0 || bd[bi] != 0 {
				data[i] = 1
			}
		}
	}
	return []tensor.Tensor{tensor.NewDense[uint8](outShape, data)}, nil
}

func opNonZero(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	t := inputs[0]
	shape := t.Shape()
	ndim := shape.NDim()

	// Collect indices of non-zero elements
	var indices [][]int64
	for d := 0; d < ndim; d++ {
		indices = append(indices, nil)
	}

	switch dt := t.(type) {
	case *tensor.Dense[float32]:
		strides := tensor.Strides(shape)
		for i, v := range dt.Data() {
			if v != 0 {
				rem := i
				for d := 0; d < ndim; d++ {
					coord := rem / strides[d]
					rem %= strides[d]
					indices[d] = append(indices[d], int64(coord))
				}
			}
		}
	case *tensor.Dense[int64]:
		strides := tensor.Strides(shape)
		for i, v := range dt.Data() {
			if v != 0 {
				rem := i
				for d := 0; d < ndim; d++ {
					coord := rem / strides[d]
					rem %= strides[d]
					indices[d] = append(indices[d], int64(coord))
				}
			}
		}
	case *tensor.Dense[uint8]:
		strides := tensor.Strides(shape)
		for i, v := range dt.Data() {
			if v != 0 {
				rem := i
				for d := 0; d < ndim; d++ {
					coord := rem / strides[d]
					rem %= strides[d]
					indices[d] = append(indices[d], int64(coord))
				}
			}
		}
	default:
		return nil, fmt.Errorf("NonZero: unsupported type %T", t)
	}

	nnz := 0
	if len(indices) > 0 {
		nnz = len(indices[0])
	}
	// Output shape: [ndim, nnz]
	outData := make([]int64, ndim*nnz)
	for d := 0; d < ndim; d++ {
		for i := 0; i < nnz; i++ {
			outData[d*nnz+i] = indices[d][i]
		}
	}
	return []tensor.Tensor{tensor.NewDense[int64](tensor.Shape{ndim, nnz}, outData)}, nil
}

// ── IsNaN ──

func opIsNaN(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := x.Data()
		out := make([]uint8, len(data))
		for i, v := range data {
			if math.IsNaN(float64(v)) {
				out[i] = 1
			}
		}
		return []tensor.Tensor{tensor.NewDense[uint8](x.Shape().Clone(), out)}, nil
	case *tensor.Dense[float64]:
		data := x.Data()
		out := make([]uint8, len(data))
		for i, v := range data {
			if math.IsNaN(v) {
				out[i] = 1
			}
		}
		return []tensor.Tensor{tensor.NewDense[uint8](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("IsNaN: unsupported type %T", inputs[0])
	}
}

// ── IsInf ──

func opIsInf(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	detectNeg := node.GetAttrInt("detect_negative", 1) != 0
	detectPos := node.GetAttrInt("detect_positive", 1) != 0
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := x.Data()
		out := make([]uint8, len(data))
		for i, v := range data {
			f := float64(v)
			if math.IsInf(f, 0) {
				if (f > 0 && detectPos) || (f < 0 && detectNeg) {
					out[i] = 1
				}
			}
		}
		return []tensor.Tensor{tensor.NewDense[uint8](x.Shape().Clone(), out)}, nil
	case *tensor.Dense[float64]:
		data := x.Data()
		out := make([]uint8, len(data))
		for i, v := range data {
			if math.IsInf(v, 0) {
				if (v > 0 && detectPos) || (v < 0 && detectNeg) {
					out[i] = 1
				}
			}
		}
		return []tensor.Tensor{tensor.NewDense[uint8](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("IsInf: unsupported type %T", inputs[0])
	}
}

// ── CastLike ──

func opCastLike(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// CastLike: cast input[0] to same dtype as input[1]
	target := inputs[1]
	targetDType := target.DType()

	// Build a fake node with "to" attribute matching target dtype
	toVal := int64(1) // default float32
	switch targetDType {
	case 1:
		toVal = 1 // float32
	case 6:
		toVal = 6 // int32
	case 7:
		toVal = 7 // int64
	case 9:
		toVal = 9 // bool
	case 11:
		toVal = 11 // float64
	default:
		toVal = int64(targetDType)
	}
	fakeNode := &ir.Node{Attrs: map[string]ir.AttrValue{
		"to": ir.AttrInt{Value: toVal},
	}}
	return opCast(fakeNode, []tensor.Tensor{inputs[0]})
}
