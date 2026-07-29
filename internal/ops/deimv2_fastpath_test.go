package ops

import (
	"testing"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func TestFusedAffineScalarFastPath(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{-1, 0, 1, 2})
	scale := tensor.NewDense[float32](tensor.Shape{1}, []float32{2})
	bias := tensor.NewDense[float32](tensor.Shape{1}, []float32{3})
	out, err := opFusedAffine(&ir.Node{}, []tensor.Tensor{x, scale, bias})
	if err != nil {
		t.Fatal(err)
	}
	got := out[0].(*tensor.Dense[float32]).Data()
	want := []float32{1, 3, 5, 7}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out=%v want=%v", got, want)
		}
	}
}

func TestConv1x1ParallelMatchesSerial(t *testing.T) {
	xData := make([]float32, 16*64*64)
	wData := make([]float32, 16*16)
	for i := range xData {
		xData[i] = float32(i%17-8) / 17
	}
	for i := range wData {
		wData[i] = float32(i%11-5) / 11
	}
	x := tensor.NewDense[float32](tensor.Shape{1, 16, 64, 64}, xData)
	w := tensor.NewDense[float32](tensor.Shape{16, 16, 1, 1}, wData)
	node := &ir.Node{Attrs: map[string]ir.AttrValue{"kernel_shape": ir.AttrInts{Value: []int64{1, 1}}}}
	parallel := DefaultKernelConfig()
	parallel.MaxThreads = 4
	activeConvConfig = parallel
	got, err := conv2d(x, w, (*tensor.Dense[float32])(nil), node)
	if err != nil {
		t.Fatal(err)
	}
	serial := DefaultKernelConfig()
	serial.UseParallelConv = false
	activeConvConfig = serial
	want, err := conv2d(x, w, (*tensor.Dense[float32])(nil), node)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range want.Data() {
		if got.Data()[i] != v {
			t.Fatalf("index %d: parallel=%v serial=%v", i, got.Data()[i], v)
		}
	}
}

// ストリップ融合 conv (float32) が汎用パス (float64) と一致することを確認する。
// 値を小さな整数に限定しているため float32/float64 の結果は厳密に一致する。
func TestConvStripsMatchGenericPath(t *testing.T) {
	oldTarget := convStripTargetFloats
	convStripTargetFloats = 64 // 極小にしてストリップ分割を強制
	oldCfg := activeConvConfig
	parallel := DefaultKernelConfig()
	parallel.MaxThreads = 4
	activeConvConfig = parallel
	defer func() {
		convStripTargetFloats = oldTarget
		activeConvConfig = oldCfg
	}()

	cases := []struct {
		name                    string
		N, C, H, W, OC, KH, KW  int
		stride, pad, dil, group int
		bias                    bool
	}{
		{"3x3s1p1", 1, 3, 13, 11, 5, 3, 3, 1, 1, 1, 1, true},
		{"3x3s2p1", 1, 4, 16, 15, 8, 3, 3, 2, 1, 1, 1, false},
		{"5x5dil2", 1, 2, 20, 18, 3, 5, 5, 1, 2, 2, 1, true},
		{"group2", 1, 4, 12, 12, 6, 3, 3, 1, 1, 1, 2, true},
		{"1x1pad0", 2, 3, 9, 9, 4, 1, 1, 1, 0, 1, 1, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			xN := tc.N * tc.C * tc.H * tc.W
			wN := tc.OC * (tc.C / tc.group) * tc.KH * tc.KW
			x32 := make([]float32, xN)
			x64 := make([]float64, xN)
			for i := range x32 {
				v := float64(i%7 - 3)
				x32[i], x64[i] = float32(v), v
			}
			w32 := make([]float32, wN)
			w64 := make([]float64, wN)
			for i := range w32 {
				v := float64(i%5 - 2)
				w32[i], w64[i] = float32(v), v
			}
			var b32 *tensor.Dense[float32]
			var b64 *tensor.Dense[float64]
			if tc.bias {
				bv32 := make([]float32, tc.OC)
				bv64 := make([]float64, tc.OC)
				for i := range bv32 {
					v := float64(i%3 - 1)
					bv32[i], bv64[i] = float32(v), v
				}
				b32 = tensor.NewDense[float32](tensor.Shape{tc.OC}, bv32)
				b64 = tensor.NewDense[float64](tensor.Shape{tc.OC}, bv64)
			}
			node := &ir.Node{Attrs: map[string]ir.AttrValue{
				"kernel_shape": ir.AttrInts{Value: []int64{int64(tc.KH), int64(tc.KW)}},
				"strides":      ir.AttrInts{Value: []int64{int64(tc.stride), int64(tc.stride)}},
				"pads":         ir.AttrInts{Value: []int64{int64(tc.pad), int64(tc.pad), int64(tc.pad), int64(tc.pad)}},
				"dilations":    ir.AttrInts{Value: []int64{int64(tc.dil), int64(tc.dil)}},
				"group":        ir.AttrInt{Value: int64(tc.group)},
			}}
			xShape := tensor.Shape{tc.N, tc.C, tc.H, tc.W}
			wShape := tensor.Shape{tc.OC, tc.C / tc.group, tc.KH, tc.KW}
			got, err := conv2d(tensor.NewDense[float32](xShape, x32), tensor.NewDense[float32](wShape, w32), b32, node)
			if err != nil {
				t.Fatal(err)
			}
			want, err := conv2d(tensor.NewDense[float64](xShape, x64), tensor.NewDense[float64](wShape, w64), b64, node)
			if err != nil {
				t.Fatal(err)
			}
			for i, wv := range want.Data() {
				if got.Data()[i] != float32(wv) {
					t.Fatalf("index %d: got=%v want=%v", i, got.Data()[i], wv)
				}
			}
		})
	}
}

func TestMaxPool2x2Stride1FastPath(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 3, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	node := &ir.Node{Attrs: map[string]ir.AttrValue{"kernel_shape": ir.AttrInts{Value: []int64{2, 2}}, "strides": ir.AttrInts{Value: []int64{1, 1}}}}
	activePoolConfig = DefaultKernelConfig()
	got, err := maxPool2d(x, node)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{5, 6, 8, 9}
	for i := range want {
		if got.Data()[i] != want[i] {
			t.Fatalf("out=%v want=%v", got.Data(), want)
		}
	}
}
