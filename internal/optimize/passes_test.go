package optimize

import (
	"testing"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
)

func TestFuseConvAffineScalar(t *testing.T) {
	conv := &ir.Node{OpType: "FusedConv", Inputs: []string{"x", "w"}, Outputs: []string{"conv"}, Attrs: map[string]ir.AttrValue{"activation": ir.AttrString{Value: "relu"}}}
	affine := &ir.Node{OpType: "FusedAffine", Inputs: []string{"conv", "scale", "bias"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{conv, affine}, Initializers: map[string]*ir.Initializer{
		"scale": {DType: ir.DataTypeFloat, Shape: ir.Shape{1}, FloatData: []float32{2}},
		"bias":  {DType: ir.DataTypeFloat, Shape: ir.Shape{1}, FloatData: []float32{3}},
	}}
	fuseConvAffine(g)
	if len(g.Nodes) != 1 || conv.Outputs[0] != "y" {
		t.Fatalf("affine not fused: %#v", g.Nodes)
	}
	if got := conv.GetAttrFloat("post_scale", 0); got != 2 {
		t.Fatalf("post_scale=%v", got)
	}
	if got := conv.GetAttrFloat("post_bias", 0); got != 3 {
		t.Fatalf("post_bias=%v", got)
	}
}

func TestFusePadConvStaticZeroPad(t *testing.T) {
	pad := &ir.Node{OpType: "Pad", Inputs: []string{"x", "pads"}, Outputs: []string{"padded"}, Attrs: map[string]ir.AttrValue{"mode": ir.AttrString{Value: "constant"}}}
	conv := &ir.Node{OpType: "Conv", Inputs: []string{"padded", "w"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{"pads": ir.AttrInts{Value: []int64{1, 1, 1, 1}}}}
	g := &ir.Graph{Nodes: []*ir.Node{pad, conv}, Initializers: map[string]*ir.Initializer{
		"pads": {DType: ir.DataTypeInt64, Shape: ir.Shape{8}, Int64Data: []int64{0, 0, 2, 3, 0, 0, 4, 5}},
	}}
	fusePadConv(g)
	if len(g.Nodes) != 1 || conv.Inputs[0] != "x" {
		t.Fatalf("pad not fused: %#v", g.Nodes)
	}
	want := []int64{3, 4, 5, 6}
	got := conv.GetAttrInts("pads", nil)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pads=%v want=%v", got, want)
		}
	}
}

// 逆数形 RMSNorm: Mul(Mul(x, Div(1, Sqrt(Add(ReduceMean(Pow(x,2)), eps)))), scale)
func rmsReciprocalGraph(axis int64) (*ir.Graph, *ir.Node) {
	pow := &ir.Node{OpType: "Pow", Inputs: []string{"x", "two"}, Outputs: []string{"pow"}, Attrs: map[string]ir.AttrValue{}}
	mean := &ir.Node{OpType: "ReduceMean", Inputs: []string{"pow"}, Outputs: []string{"mean"}, Attrs: map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{axis}}}}
	add := &ir.Node{OpType: "Add", Inputs: []string{"mean", "eps"}, Outputs: []string{"var_eps"}, Attrs: map[string]ir.AttrValue{}}
	sqrt := &ir.Node{OpType: "Sqrt", Inputs: []string{"var_eps"}, Outputs: []string{"std"}, Attrs: map[string]ir.AttrValue{}}
	div := &ir.Node{OpType: "Div", Inputs: []string{"one", "std"}, Outputs: []string{"recip"}, Attrs: map[string]ir.AttrValue{}}
	mul := &ir.Node{OpType: "Mul", Inputs: []string{"x", "recip"}, Outputs: []string{"normed"}, Attrs: map[string]ir.AttrValue{}}
	final := &ir.Node{OpType: "Mul", Inputs: []string{"normed", "scale"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{pow, mean, add, sqrt, div, mul, final}, Initializers: map[string]*ir.Initializer{
		"two":   {DType: ir.DataTypeFloat, Shape: ir.Shape{}, FloatData: []float32{2}},
		"one":   {DType: ir.DataTypeFloat, Shape: ir.Shape{}, FloatData: []float32{1}},
		"eps":   {DType: ir.DataTypeFloat, Shape: ir.Shape{}, FloatData: []float32{1e-5}},
		"scale": {DType: ir.DataTypeFloat, Shape: ir.Shape{4}, FloatData: []float32{1, 1, 1, 1}},
	}}
	return g, final
}

func TestFuseRMSNormReciprocalForm(t *testing.T) {
	g, final := rmsReciprocalGraph(-1)
	fuseRMSNormalization(g)
	if len(g.Nodes) != 1 || final.OpType != "RMSNormalization" {
		t.Fatalf("not fused: %#v", g.Nodes)
	}
	if final.Inputs[0] != "x" || final.Inputs[1] != "scale" {
		t.Fatalf("inputs=%v", final.Inputs)
	}
	if got := final.GetAttrInt("axis", 0); got != -1 {
		t.Fatalf("axis=%v", got)
	}
	if got := final.GetAttrFloat("epsilon", 0); got != 1e-5 {
		t.Fatalf("epsilon=%v", got)
	}
}

func TestFuseRMSNormRejectsNonLastAxis(t *testing.T) {
	// カーネルは [axis, rank) を正規化するため、最終軸 (-1) 以外は融合しない
	g, final := rmsReciprocalGraph(1)
	fuseRMSNormalization(g)
	if len(g.Nodes) != 7 || final.OpType != "Mul" {
		t.Fatalf("non-last axis must not be fused: %#v", g.Nodes)
	}
}

func TestFuseRMSNormDivForm(t *testing.T) {
	// 変種1: Mul(Div(x, Sqrt(Add(ReduceMean(Pow(x,2)), eps))), scale)
	pow := &ir.Node{OpType: "Pow", Inputs: []string{"x", "two"}, Outputs: []string{"pow"}, Attrs: map[string]ir.AttrValue{}}
	mean := &ir.Node{OpType: "ReduceMean", Inputs: []string{"pow"}, Outputs: []string{"mean"}, Attrs: map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{-1}}}}
	add := &ir.Node{OpType: "Add", Inputs: []string{"mean", "eps"}, Outputs: []string{"var_eps"}, Attrs: map[string]ir.AttrValue{}}
	sqrt := &ir.Node{OpType: "Sqrt", Inputs: []string{"var_eps"}, Outputs: []string{"std"}, Attrs: map[string]ir.AttrValue{}}
	div := &ir.Node{OpType: "Div", Inputs: []string{"x", "std"}, Outputs: []string{"normed"}, Attrs: map[string]ir.AttrValue{}}
	final := &ir.Node{OpType: "Mul", Inputs: []string{"normed", "scale"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{pow, mean, add, sqrt, div, final}, Initializers: map[string]*ir.Initializer{
		"two":   {DType: ir.DataTypeFloat, Shape: ir.Shape{}, FloatData: []float32{2}},
		"eps":   {DType: ir.DataTypeFloat, Shape: ir.Shape{}, FloatData: []float32{1e-6}},
		"scale": {DType: ir.DataTypeFloat, Shape: ir.Shape{4}, FloatData: []float32{1, 1, 1, 1}},
	}}
	fuseRMSNormalization(g)
	if len(g.Nodes) != 1 || final.OpType != "RMSNormalization" {
		t.Fatalf("not fused: %#v", g.Nodes)
	}
}

func TestFuseLayerNormalization(t *testing.T) {
	mean := &ir.Node{OpType: "ReduceMean", Inputs: []string{"x"}, Outputs: []string{"mean"}, Attrs: map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{-1}}}}
	sub := &ir.Node{OpType: "Sub", Inputs: []string{"x", "mean"}, Outputs: []string{"centered"}, Attrs: map[string]ir.AttrValue{}}
	pow := &ir.Node{OpType: "Pow", Inputs: []string{"centered", "two"}, Outputs: []string{"pow"}, Attrs: map[string]ir.AttrValue{}}
	variance := &ir.Node{OpType: "ReduceMean", Inputs: []string{"pow"}, Outputs: []string{"variance"}, Attrs: map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{-1}}}}
	add := &ir.Node{OpType: "Add", Inputs: []string{"variance", "eps"}, Outputs: []string{"var_eps"}, Attrs: map[string]ir.AttrValue{}}
	sqrt := &ir.Node{OpType: "Sqrt", Inputs: []string{"var_eps"}, Outputs: []string{"std"}, Attrs: map[string]ir.AttrValue{}}
	div := &ir.Node{OpType: "Div", Inputs: []string{"centered", "std"}, Outputs: []string{"normed"}, Attrs: map[string]ir.AttrValue{}}
	mul := &ir.Node{OpType: "Mul", Inputs: []string{"normed", "scale"}, Outputs: []string{"scaled"}, Attrs: map[string]ir.AttrValue{}}
	final := &ir.Node{OpType: "Add", Inputs: []string{"scaled", "bias"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{mean, sub, pow, variance, add, sqrt, div, mul, final}, Initializers: map[string]*ir.Initializer{
		"two":   {DType: ir.DataTypeFloat, Shape: ir.Shape{}, FloatData: []float32{2}},
		"eps":   {DType: ir.DataTypeFloat, Shape: ir.Shape{}, FloatData: []float32{1e-5}},
		"scale": {DType: ir.DataTypeFloat, Shape: ir.Shape{4}, FloatData: []float32{1, 1, 1, 1}},
		"bias":  {DType: ir.DataTypeFloat, Shape: ir.Shape{4}, FloatData: []float32{0, 0, 0, 0}},
	}}
	fuseLayerNormalization(g)
	if len(g.Nodes) != 1 || final.OpType != "LayerNormalization" {
		t.Fatalf("not fused: %#v", g.Nodes)
	}
	if final.Inputs[0] != "x" || final.Inputs[1] != "scale" || final.Inputs[2] != "bias" {
		t.Fatalf("inputs=%v", final.Inputs)
	}
}

func hardSwishGraph(alpha float32) (*ir.Graph, *ir.Node) {
	hs := &ir.Node{OpType: "HardSigmoid", Inputs: []string{"x"}, Outputs: []string{"hs"}, Attrs: map[string]ir.AttrValue{
		"alpha": ir.AttrFloat{Value: alpha}, "beta": ir.AttrFloat{Value: 0.5}}}
	mul := &ir.Node{OpType: "Mul", Inputs: []string{"x", "hs"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{hs, mul}, Initializers: map[string]*ir.Initializer{}}
	return g, mul
}

func TestFuseHardSwish(t *testing.T) {
	g, mul := hardSwishGraph(1.0 / 6.0)
	fuseHardSwish(g)
	if len(g.Nodes) != 1 || mul.OpType != "HardSwish" || mul.Inputs[0] != "x" {
		t.Fatalf("not fused: %#v", g.Nodes)
	}
}

func TestFuseHardSwishRejectsWrongAlpha(t *testing.T) {
	// ONNX デフォルトの alpha=0.2 は HardSwish (alpha=1/6) と一致しないため融合しない
	g, mul := hardSwishGraph(0.2)
	fuseHardSwish(g)
	if len(g.Nodes) != 2 || mul.OpType != "Mul" {
		t.Fatalf("must not fuse: %#v", g.Nodes)
	}
}

func TestFuseConvHardSwish(t *testing.T) {
	conv := &ir.Node{OpType: "Conv", Inputs: []string{"x", "w"}, Outputs: []string{"c"}, Attrs: map[string]ir.AttrValue{}}
	hsw := &ir.Node{OpType: "HardSwish", Inputs: []string{"c"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{conv, hsw}, Initializers: map[string]*ir.Initializer{}}
	fuseConvActivation(g)
	if len(g.Nodes) != 1 || conv.OpType != "FusedConv" {
		t.Fatalf("not fused: %#v", g.Nodes)
	}
	if got := conv.GetAttrString("activation", ""); got != "hardswish" {
		t.Fatalf("activation=%q", got)
	}
}

func TestFoldConstantsAdd(t *testing.T) {
	add := &ir.Node{OpType: "Add", Inputs: []string{"a", "b"}, Outputs: []string{"c"}, Attrs: map[string]ir.AttrValue{}}
	use := &ir.Node{OpType: "Relu", Inputs: []string{"c"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{add, use}, Initializers: map[string]*ir.Initializer{
		"a": {DType: ir.DataTypeFloat, Shape: ir.Shape{2}, FloatData: []float32{1, 2}},
		"b": {DType: ir.DataTypeFloat, Shape: ir.Shape{2}, FloatData: []float32{10, 20}},
	}}
	foldConstants(g)
	if len(g.Nodes) != 1 {
		t.Fatalf("Add not folded: %#v", g.Nodes)
	}
	folded := g.Initializers["c"]
	if folded == nil || len(folded.FloatData) != 2 || folded.FloatData[0] != 11 || folded.FloatData[1] != 22 {
		t.Fatalf("folded=%#v", folded)
	}
}

func TestSimplifyTransposesPreservesGraphOutputName(t *testing.T) {
	// グラフ出力を生成する恒等 Transpose は除去しない(出力名が変わるため)
	t1 := &ir.Node{OpType: "Transpose", Inputs: []string{"x"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{"perm": ir.AttrInts{Value: []int64{0, 1}}}}
	g := &ir.Graph{Nodes: []*ir.Node{t1}, Initializers: map[string]*ir.Initializer{},
		Outputs: []ir.TensorSpec{{Name: "y"}}}
	simplifyTransposes(g)
	if len(g.Nodes) != 1 || g.Outputs[0].Name != "y" {
		t.Fatalf("graph output renamed: nodes=%d out=%q", len(g.Nodes), g.Outputs[0].Name)
	}
}

func TestSimplifyTransposesComposesToIdentity(t *testing.T) {
	t1 := &ir.Node{OpType: "Transpose", Inputs: []string{"x"}, Outputs: []string{"t1"}, Attrs: map[string]ir.AttrValue{"perm": ir.AttrInts{Value: []int64{1, 0}}}}
	t2 := &ir.Node{OpType: "Transpose", Inputs: []string{"t1"}, Outputs: []string{"t2"}, Attrs: map[string]ir.AttrValue{"perm": ir.AttrInts{Value: []int64{1, 0}}}}
	use := &ir.Node{OpType: "Relu", Inputs: []string{"t2"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{t1, t2, use}, Initializers: map[string]*ir.Initializer{}}
	simplifyTransposes(g)
	if len(g.Nodes) != 1 || use.Inputs[0] != "x" {
		t.Fatalf("transposes not eliminated: %#v", g.Nodes)
	}
}

func TestFusePadConvRejectsSharedOutput(t *testing.T) {
	pad := &ir.Node{OpType: "Pad", Inputs: []string{"x", "pads"}, Outputs: []string{"padded"}, Attrs: map[string]ir.AttrValue{}}
	conv := &ir.Node{OpType: "Conv", Inputs: []string{"padded", "w"}, Outputs: []string{"y"}, Attrs: map[string]ir.AttrValue{}}
	other := &ir.Node{OpType: "MaxPool", Inputs: []string{"padded"}, Outputs: []string{"z"}, Attrs: map[string]ir.AttrValue{}}
	g := &ir.Graph{Nodes: []*ir.Node{pad, conv, other}, Initializers: map[string]*ir.Initializer{"pads": {DType: ir.DataTypeInt64, Int64Data: []int64{0, 0, 1, 1, 0, 0, 1, 1}}}}
	fusePadConv(g)
	if len(g.Nodes) != 3 || conv.Inputs[0] != "padded" {
		t.Fatal("shared Pad must not be fused")
	}
}
