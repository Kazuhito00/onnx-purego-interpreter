package ops

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── Pooling ──

func opGlobalMaxPool(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{globalMaxPoolF32(x)}, nil
	default:
		return nil, fmt.Errorf("GlobalMaxPool: unsupported type %T", inputs[0])
	}
}

func globalMaxPoolF32(x *tensor.Dense[float32]) *tensor.Dense[float32] {
	s := x.Shape()
	N, C := s[0], s[1]
	spatialSize := 1
	for i := 2; i < s.NDim(); i++ { spatialSize *= s[i] }
	outShape := make(tensor.Shape, s.NDim())
	outShape[0], outShape[1] = N, C
	for i := 2; i < s.NDim(); i++ { outShape[i] = 1 }
	data := x.Data()
	out := make([]float32, N*C)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			base := n*C*spatialSize + c*spatialSize
			m := data[base]
			for i := 1; i < spatialSize; i++ { if data[base+i] > m { m = data[base+i] } }
			out[n*C+c] = m
		}
	}
	return tensor.NewDense[float32](outShape, out)
}

// ── Shape ops ──

func opDepthToSpace(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	blocksize := int(node.GetAttrInt("blocksize", 1))
	mode := node.GetAttrString("mode", "DCR")
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		return []tensor.Tensor{depthToSpaceF32(x, blocksize, mode)}, nil
	default:
		return nil, fmt.Errorf("DepthToSpace: unsupported type %T", inputs[0])
	}
}

func depthToSpaceF32(x *tensor.Dense[float32], bs int, mode string) *tensor.Dense[float32] {
	s := x.Shape() // [N, C, H, W]
	N, C, H, W := s[0], s[1], s[2], s[3]
	oC := C / (bs * bs)
	oH, oW := H*bs, W*bs
	data := x.Data()
	out := make([]float32, N*oC*oH*oW)
	for n := 0; n < N; n++ {
		for oc := 0; oc < oC; oc++ {
			for oh := 0; oh < oH; oh++ {
				for ow := 0; ow < oW; ow++ {
					h, bh := oh/bs, oh%bs
					w, bw := ow/bs, ow%bs
					var ic int
					if mode == "CRD" {
						ic = oc*bs*bs + bh*bs + bw
					} else { // DCR
						ic = bh*bs*oC + bw*oC + oc
					}
					out[n*oC*oH*oW+oc*oH*oW+oh*oW+ow] = data[n*C*H*W+ic*H*W+h*W+w]
				}
			}
		}
	}
	return tensor.NewDense[float32](tensor.Shape{N, oC, oH, oW}, out)
}

func opSpaceToDepth(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	blocksize := int(node.GetAttrInt("blocksize", 1))
	switch x := inputs[0].(type) {
	case *tensor.Dense[float32]:
		s := x.Shape()
		N, C, H, W := s[0], s[1], s[2], s[3]
		oC := C * blocksize * blocksize
		oH, oW := H/blocksize, W/blocksize
		data := x.Data()
		out := make([]float32, N*oC*oH*oW)
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				for h := 0; h < oH; h++ {
					for w := 0; w < oW; w++ {
						for bh := 0; bh < blocksize; bh++ {
							for bw := 0; bw < blocksize; bw++ {
								oc := c*blocksize*blocksize + bh*blocksize + bw
								ih, iw := h*blocksize+bh, w*blocksize+bw
								out[n*oC*oH*oW+oc*oH*oW+h*oW+w] = data[n*C*H*W+c*H*W+ih*W+iw]
							}
						}
					}
				}
			}
		}
		return []tensor.Tensor{tensor.NewDense[float32](tensor.Shape{N, oC, oH, oW}, out)}, nil
	default:
		return nil, fmt.Errorf("SpaceToDepth: unsupported type %T", inputs[0])
	}
}
