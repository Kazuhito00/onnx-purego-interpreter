package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Simple unary math ops ──

func opCeil(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Ceil(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.Ceil(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Ceil: unsupported type %T", inputs[0])
	}
}

func opRound(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.RoundToEven(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.RoundToEven(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Round: unsupported type %T", inputs[0])
	}
}

func opSign(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() {
			if v > 0 { data[i] = 1 } else if v < 0 { data[i] = -1 }
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() {
			if v > 0 { data[i] = 1 } else if v < 0 { data[i] = -1 }
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[int64]:
		data := make([]int64, t.Len())
		for i, v := range t.Data() {
			if v > 0 { data[i] = 1 } else if v < 0 { data[i] = -1 }
		}
		return []tensor.Tensor{tensor.NewDense[int64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Sign: unsupported type %T", inputs[0])
	}
}

func opTan(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Tan(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.Tan(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Tan: unsupported type %T", inputs[0])
	}
}

func opSinh(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Sinh(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.Sinh(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Sinh: unsupported type %T", inputs[0])
	}
}

func opCosh(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Cosh(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.Cosh(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Cosh: unsupported type %T", inputs[0])
	}
}

func opAsin(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Asin(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.Asin(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Asin: unsupported type %T", inputs[0])
	}
}

func opAcos(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Acos(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.Acos(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Acos: unsupported type %T", inputs[0])
	}
}

func opAtan(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		for i, v := range t.Data() { data[i] = float32(math.Atan(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		for i, v := range t.Data() { data[i] = math.Atan(v) }
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("Atan: unsupported type %T", inputs[0])
	}
}

// ── Acosh / Asinh / Atanh ──

func opAcosh(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out := make([]float32, x.Len())
		for i, v := range x.Data() { out[i] = float32(math.Acosh(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("Acosh: unsupported type %T", inputs[0])
	}
}

func opAsinh(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out := make([]float32, x.Len())
		for i, v := range x.Data() { out[i] = float32(math.Asinh(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("Asinh: unsupported type %T", inputs[0])
	}
}

func opAtanh(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out := make([]float32, x.Len())
		for i, v := range x.Data() { out[i] = float32(math.Atanh(float64(v))) }
		return []tensor.Tensor{tensor.NewDense[float32](x.Shape().Clone(), out)}, nil
	default:
		return nil, fmt.Errorf("Atanh: unsupported type %T", inputs[0])
	}
}
