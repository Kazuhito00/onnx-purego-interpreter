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
