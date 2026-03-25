package ops

import (
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// FastGELU: y = 0.5 * x * (1 + erf(x / sqrt(2)))
// fastErfF32 approximates erf(x) for float32 using Abramowitz & Stegun 7.1.26.
// Max error < 1.5e-7.
func fastErfF32(x float32) float32 {
	const a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
	const p = 0.3275911
	sign := float32(1)
	if x < 0 { sign = -1; x = -x }
	t := 1.0 / (1.0 + p*x)
	y := 1.0 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*float32(math.Exp(float64(-x*x)))
	return sign * y
}

func makeFastGELU(kc *KernelConfig) OpFunc {
	return func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
		return opFastGELUWithConfig(node, inputs, kc)
	}
}

func opFastGELU(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return opFastGELUWithConfig(node, inputs, nil)
}

func opFastGELUWithConfig(node *ir.Node, inputs []tensor.Tensor, kc *KernelConfig) ([]tensor.Tensor, error) {
	useFastErf := kc == nil || kc.UseFastErf
	switch t := inputs[0].(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, t.Len())
		src := t.Data()
		if useFastErf {
			const invSqrt2 = float32(0.7071067811865476)
			for i, v := range src {
				data[i] = 0.5 * v * (1.0 + fastErfF32(v*invSqrt2))
			}
		} else {
			invSqrt2 := float64(1.0 / math.Sqrt(2.0))
			for i, v := range src {
				data[i] = float32(0.5 * float64(v) * (1.0 + math.Erf(float64(v)*invSqrt2)))
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](t.Shape().Clone(), data)}, nil
	case *tensor.Dense[float64]:
		data := make([]float64, t.Len())
		src := t.Data()
		invSqrt2 := 1.0 / math.Sqrt(2.0)
		for i, v := range src {
			data[i] = 0.5 * v * (1.0 + math.Erf(v*invSqrt2))
		}
		return []tensor.Tensor{tensor.NewDense[float64](t.Shape().Clone(), data)}, nil
	default:
		return nil, fmt.Errorf("FastGELU: unsupported type %T", inputs[0])
	}
}

// FusedMatMul: Y = MatMul(A, B) + bias
// inputs: A, B, bias
func opFusedMatMul(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// First do MatMul
	matmulInputs := inputs[:2]
	results, err := opMatMul(node, matmulInputs)
	if err != nil {
		return nil, fmt.Errorf("FusedMatMul: %w", err)
	}

	// Then add bias
	if len(inputs) > 2 && inputs[2] != nil {
		biasInputs := []tensor.Tensor{results[0], inputs[2]}
		biased, err := dispatchBinaryOp(biasInputs,
			func(a, b float32) float32 { return a + b },
			func(a, b float64) float64 { return a + b },
			func(a, b int32) int32 { return a + b },
			func(a, b int64) int64 { return a + b },
		)
		if err != nil {
			return nil, fmt.Errorf("FusedMatMul bias: %w", err)
		}
		return []tensor.Tensor{biased}, nil
	}

	return results, nil
}

// FusedAffine: Y = X * scale + bias (element-wise with broadcasting)
func opFusedAffine(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x, scale, bias := inputs[0], inputs[1], inputs[2]

	// Compute X * scale
	scaled, err := dispatchBinaryOp([]tensor.Tensor{x, scale},
		func(a, b float32) float32 { return a*b },
		func(a, b float64) float64 { return a*b },
		func(a, b int32) int32 { return a*b },
		func(a, b int64) int64 { return a*b },
	)
	if err != nil {
		return nil, fmt.Errorf("FusedAffine mul: %w", err)
	}

	// Add bias
	result, err := dispatchBinaryOp([]tensor.Tensor{scaled, bias},
		func(a, b float32) float32 { return a+b },
		func(a, b float64) float64 { return a+b },
		func(a, b int32) int32 { return a+b },
		func(a, b int64) int64 { return a+b },
	)
	if err != nil {
		return nil, fmt.Errorf("FusedAffine add: %w", err)
	}

	return []tensor.Tensor{result}, nil
}
