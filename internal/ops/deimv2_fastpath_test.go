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
