package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

var activePoolConfig *KernelConfig

func makeMaxPool(kc *KernelConfig) OpFunc {
	return func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
		activePoolConfig = kc
		return opMaxPool(node, inputs)
	}
}

func opMaxPool(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out, err := maxPool2d(x, node)
		if err != nil {
			return nil, fmt.Errorf("MaxPool: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		out, err := maxPool2d(x, node)
		if err != nil {
			return nil, fmt.Errorf("MaxPool: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("MaxPool: unsupported type %T", inputs[0])
	}
}

func maxPool2d[T tensor.Numeric](x *tensor.Dense[T], node *ir.Node) (*tensor.Dense[T], error) {
	xShape := x.Shape() // [N, C, H, W]
	if xShape.NDim() != 4 {
		return nil, fmt.Errorf("maxPool2d requires 4D input, got %v", xShape)
	}

	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]

	kernelShape := node.GetAttrInts("kernel_shape", nil)
	if kernelShape == nil || len(kernelShape) < 2 {
		return nil, fmt.Errorf("MaxPool: kernel_shape required")
	}
	KH, KW := int(kernelShape[0]), int(kernelShape[1])

	strides := node.GetAttrInts("strides", []int64{1, 1})
	strideH, strideW := int(strides[0]), int(strides[1])

	padTop, padLeft, padBottom, padRight := computePads(node, H, W, KH, KW, strideH, strideW)

	OH := (H + padTop + padBottom - KH) / strideH + 1
	OW := (W + padLeft + padRight - KW) / strideW + 1

	outShape := tensor.Shape{N, C, OH, OW}
	outData := make([]T, outShape.Size())
	xData := x.Data()

	// Fast path: 2x2 stride 2, no padding
	usePoolFP := activePoolConfig == nil || activePoolConfig.UsePoolFastPath
	if usePoolFP && KH == 2 && KW == 2 && strideH == 2 && strideW == 2 && padTop == 0 && padLeft == 0 {
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				xBase := n*C*H*W + c*H*W
				oBase := n*C*OH*OW + c*OH*OW
				for oh := 0; oh < OH; oh++ {
					ih := oh * 2
					for ow := 0; ow < OW; ow++ {
						iw := ow * 2
						r0 := xBase + ih*W + iw
						r1 := r0 + W
						v0 := xData[r0]; v1 := xData[r0+1]
						v2 := xData[r1]; v3 := xData[r1+1]
						m := v0
						if v1 > m { m = v1 }
						if v2 > m { m = v2 }
						if v3 > m { m = v3 }
						outData[oBase+oh*OW+ow] = m
					}
				}
			}
		}
		return tensor.NewDense[T](outShape, outData), nil
	}

	// Fast path: 3x3 stride 2 (with or without padding)
	if usePoolFP && KH == 3 && KW == 3 && strideH == 2 && strideW == 2 {
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				xBase := n*C*H*W + c*H*W
				oBase := n*C*OH*OW + c*OH*OW
				for oh := 0; oh < OH; oh++ {
					ih0 := oh*2 - padTop
					for ow := 0; ow < OW; ow++ {
						iw0 := ow*2 - padLeft
						first := true
						var maxVal T
						for kh := 0; kh < 3; kh++ {
							ih := ih0 + kh
							if ih < 0 || ih >= H { continue }
							row := xBase + ih*W
							for kw := 0; kw < 3; kw++ {
								iw := iw0 + kw
								if iw < 0 || iw >= W { continue }
								v := xData[row+iw]
								if first || v > maxVal {
									first = false
									maxVal = v
								}
							}
						}
						outData[oBase+oh*OW+ow] = maxVal
					}
				}
			}
		}
		return tensor.NewDense[T](outShape, outData), nil
	}

	// General path
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < OH; oh++ {
				for ow := 0; ow < OW; ow++ {
					first := true
					var maxVal T
					for kh := 0; kh < KH; kh++ {
						for kw := 0; kw < KW; kw++ {
							ih := oh*strideH - padTop + kh
							iw := ow*strideW - padLeft + kw
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								v := xData[n*C*H*W+c*H*W+ih*W+iw]
								if first || v > maxVal {
									first = false
									maxVal = v
								}
							}
						}
					}
					outData[n*C*OH*OW+c*OH*OW+oh*OW+ow] = maxVal
				}
			}
		}
	}

	return tensor.NewDense[T](outShape, outData), nil
}

func opAveragePool(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		out, err := avgPool2d(x, node)
		if err != nil {
			return nil, fmt.Errorf("AveragePool: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		out, err := avgPool2d(x, node)
		if err != nil {
			return nil, fmt.Errorf("AveragePool: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("AveragePool: unsupported type %T", inputs[0])
	}
}

func avgPool2d[T tensor.Numeric](x *tensor.Dense[T], node *ir.Node) (*tensor.Dense[T], error) {
	xShape := x.Shape()
	if xShape.NDim() != 4 {
		return nil, fmt.Errorf("avgPool2d requires 4D input, got %v", xShape)
	}

	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]

	kernelShape := node.GetAttrInts("kernel_shape", nil)
	if kernelShape == nil || len(kernelShape) < 2 {
		return nil, fmt.Errorf("AveragePool: kernel_shape required")
	}
	KH, KW := int(kernelShape[0]), int(kernelShape[1])

	strides := node.GetAttrInts("strides", []int64{1, 1})
	strideH, strideW := int(strides[0]), int(strides[1])

	padTop, padLeft, padBottom, padRight := computePads(node, H, W, KH, KW, strideH, strideW)

	countIncludePad := node.GetAttrInt("count_include_pad", 0) != 0

	OH := (H + padTop + padBottom - KH) / strideH + 1
	OW := (W + padLeft + padRight - KW) / strideW + 1

	outShape := tensor.Shape{N, C, OH, OW}
	outData := make([]T, outShape.Size())
	xData := x.Data()

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < OH; oh++ {
				for ow := 0; ow < OW; ow++ {
					var sum float64
					count := 0
					for kh := 0; kh < KH; kh++ {
						for kw := 0; kw < KW; kw++ {
							ih := oh*strideH - padTop + kh
							iw := ow*strideW - padLeft + kw
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								sum += float64(xData[n*C*H*W+c*H*W+ih*W+iw])
								count++
							} else if countIncludePad {
								count++
							}
						}
					}
					if count > 0 {
						outData[n*C*OH*OW+c*OH*OW+oh*OW+ow] = T(sum / float64(count))
					}
				}
			}
		}
	}

	return tensor.NewDense[T](outShape, outData), nil
}

func opGlobalAveragePool(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{globalAvgPool(x)}, nil
	case *tensor.Dense[float64]:
		return []tensor.Tensor{globalAvgPool(x)}, nil
	default:
		return nil, fmt.Errorf("GlobalAveragePool: unsupported type %T", inputs[0])
	}
}

func globalAvgPool[T tensor.Numeric](x *tensor.Dense[T]) *tensor.Dense[T] {
	xShape := x.Shape() // [N, C, H, W, ...]
	N, C := xShape[0], xShape[1]
	spatialSize := 1
	for i := 2; i < xShape.NDim(); i++ {
		spatialSize *= xShape[i]
	}

	outShape := make(tensor.Shape, xShape.NDim())
	outShape[0] = N
	outShape[1] = C
	for i := 2; i < xShape.NDim(); i++ {
		outShape[i] = 1
	}

	outData := make([]T, N*C)
	xData := x.Data()

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			var sum float64
			base := n*C*spatialSize + c*spatialSize
			for i := 0; i < spatialSize; i++ {
				sum += float64(xData[base+i])
			}
			outData[n*C+c] = T(sum / float64(spatialSize))
		}
	}

	return tensor.NewDense[T](outShape, outData)
}
