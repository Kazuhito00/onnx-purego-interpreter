package ops

import (
	"math"
	"testing"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func makeNode(opType string, attrs map[string]ir.AttrValue) *ir.Node {
	if attrs == nil {
		attrs = make(map[string]ir.AttrValue)
	}
	return &ir.Node{OpType: opType, Attrs: attrs}
}

func assertClose(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch %d vs %d", name, len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Errorf("%s[%d]: got %f, want %f (diff %f)", name, i, got[i], want[i], diff)
		}
	}
}

func TestAdd(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{1, 2, 3, 4})
	b := tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{10, 20, 30, 40})
	out, err := opAdd(makeNode("Add", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	assertClose(t, "Add", result.Data(), []float32{11, 22, 33, 44}, 1e-6)
}

func TestAddBroadcast(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := tensor.NewDense[float32](tensor.Shape{3}, []float32{10, 20, 30})
	out, err := opAdd(makeNode("Add", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	assertClose(t, "AddBroadcast", result.Data(), []float32{11, 22, 33, 14, 25, 36}, 1e-6)
}

func TestMatMul2D(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := tensor.NewDense[float32](tensor.Shape{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	out, err := opMatMul(makeNode("MatMul", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	// [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
	// [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
	assertClose(t, "MatMul", result.Data(), []float32{22, 28, 49, 64}, 1e-6)
}

func TestGemm(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := tensor.NewDense[float32](tensor.Shape{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	c := tensor.NewDense[float32](tensor.Shape{2}, []float32{100, 200})
	node := makeNode("Gemm", map[string]ir.AttrValue{
		"alpha":  ir.AttrFloat{Value: 1.0},
		"beta":   ir.AttrFloat{Value: 1.0},
		"transA": ir.AttrInt{Value: 0},
		"transB": ir.AttrInt{Value: 0},
	})
	out, err := opGemm(node, []tensor.Tensor{a, b, c})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	assertClose(t, "Gemm", result.Data(), []float32{122, 228, 149, 264}, 1e-6)
}

func TestRelu(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{4}, []float32{-2, -1, 0, 1})
	out, err := opRelu(makeNode("Relu", nil), []tensor.Tensor{a})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	assertClose(t, "Relu", result.Data(), []float32{0, 0, 0, 1}, 1e-6)
}

func TestSoftmax(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{1, 3}, []float32{1, 2, 3})
	out, err := opSoftmax(makeNode("Softmax", map[string]ir.AttrValue{
		"axis": ir.AttrInt{Value: -1},
	}), []tensor.Tensor{a})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	data := result.Data()
	// Check sums to 1
	sum := data[0] + data[1] + data[2]
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax should sum to 1, got %f", sum)
	}
	// Check order preserved
	if data[0] >= data[1] || data[1] >= data[2] {
		t.Errorf("softmax order wrong: %v", data)
	}
}

func TestReshape(t *testing.T) {
	data := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	shape := tensor.NewDense[int64](tensor.Shape{2}, []int64{3, 2})
	out, err := opReshape(makeNode("Reshape", nil), []tensor.Tensor{data, shape})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	if !result.Shape().Equal(tensor.Shape{3, 2}) {
		t.Errorf("expected [3,2], got %v", result.Shape())
	}
}

func TestTranspose(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	out, err := opTranspose(makeNode("Transpose", map[string]ir.AttrValue{
		"perm": ir.AttrInts{Value: []int64{1, 0}},
	}), []tensor.Tensor{a})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	if !result.Shape().Equal(tensor.Shape{3, 2}) {
		t.Errorf("expected [3,2], got %v", result.Shape())
	}
	// Transposed: [[1,4],[2,5],[3,6]]
	assertClose(t, "Transpose", result.Data(), []float32{1, 4, 2, 5, 3, 6}, 1e-6)
}

func TestConv2D(t *testing.T) {
	// Simple 1x1x3x3 input, 1x1x2x2 kernel
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	w := tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	node := makeNode("Conv", map[string]ir.AttrValue{
		"strides": ir.AttrInts{Value: []int64{1, 1}},
		"pads":    ir.AttrInts{Value: []int64{0, 0, 0, 0}},
	})
	out, err := opConv(node, []tensor.Tensor{x, w})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	if !result.Shape().Equal(tensor.Shape{1, 1, 2, 2}) {
		t.Errorf("expected [1,1,2,2], got %v", result.Shape())
	}
	// [1+5, 2+6, 4+8, 5+9] = [6, 8, 12, 14]
	assertClose(t, "Conv", result.Data(), []float32{6, 8, 12, 14}, 1e-6)
}

func TestMaxPool(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 4, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	})
	node := makeNode("MaxPool", map[string]ir.AttrValue{
		"kernel_shape": ir.AttrInts{Value: []int64{2, 2}},
		"strides":      ir.AttrInts{Value: []int64{2, 2}},
	})
	out, err := opMaxPool(node, []tensor.Tensor{x})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	if !result.Shape().Equal(tensor.Shape{1, 1, 2, 2}) {
		t.Errorf("expected [1,1,2,2], got %v", result.Shape())
	}
	assertClose(t, "MaxPool", result.Data(), []float32{6, 8, 14, 16}, 1e-6)
}

func TestBatchNorm(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 2, 1, 1}, []float32{1, 2})
	scale := tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 1})
	bias := tensor.NewDense[float32](tensor.Shape{2}, []float32{0, 0})
	mean := tensor.NewDense[float32](tensor.Shape{2}, []float32{0, 0})
	variance := tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 1})
	out, err := opBatchNormalization(
		makeNode("BatchNormalization", nil),
		[]tensor.Tensor{x, scale, bias, mean, variance},
	)
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	// With mean=0, var=1, scale=1, bias=0: output ~= input (normalized)
	d := result.Data()
	if math.Abs(float64(d[0]-1.0)) > 0.01 || math.Abs(float64(d[1]-2.0)) > 0.01 {
		t.Errorf("BatchNorm unexpected output: %v", d)
	}
}

func TestGather(t *testing.T) {
	data := tensor.NewDense[float32](tensor.Shape{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	indices := tensor.NewDense[int64](tensor.Shape{2}, []int64{0, 2})
	out, err := opGather(makeNode("Gather", nil), []tensor.Tensor{data, indices})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	if !result.Shape().Equal(tensor.Shape{2, 2}) {
		t.Errorf("expected [2,2], got %v", result.Shape())
	}
	assertClose(t, "Gather", result.Data(), []float32{1, 2, 5, 6}, 1e-6)
}

func TestFlatten(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2, 3, 4}, make([]float32, 24))
	out, err := opFlatten(makeNode("Flatten", map[string]ir.AttrValue{
		"axis": ir.AttrInt{Value: 1},
	}), []tensor.Tensor{a})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	if !result.Shape().Equal(tensor.Shape{2, 12}) {
		t.Errorf("expected [2,12], got %v", result.Shape())
	}
}

func TestGlobalAveragePool(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	out, err := opGlobalAveragePool(makeNode("GlobalAveragePool", nil), []tensor.Tensor{x})
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[float32])
	if !result.Shape().Equal(tensor.Shape{1, 1, 1, 1}) {
		t.Errorf("expected [1,1,1,1], got %v", result.Shape())
	}
	assertClose(t, "GlobalAvgPool", result.Data(), []float32{2.5}, 1e-6)
}
