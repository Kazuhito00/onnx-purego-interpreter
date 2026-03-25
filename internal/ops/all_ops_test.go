package ops

import (
	"math"
	"sort"
	"testing"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

var unitTestedOps = map[string]struct{}{
	"Abs":                   {},
	"Acos":                  {},
	"Acosh":                 {},
	"Asin":                  {},
	"Asinh":                 {},
	"Atan":                  {},
	"Atanh":                 {},
	"Add":                   {},
	"Attention":              {},
	"And":                   {},
	"ArgMax":                {},
	"ArgMin":                {},
	"AveragePool":           {},
	"BatchNormalization":    {},
	"BitShift":              {},
	"BitwiseAnd":            {},
	"BitwiseNot":            {},
	"BitwiseOr":             {},
	"BitwiseXor":            {},
	"Cast":                  {},
	"CastLike":              {},
	"Ceil":                  {},
	"Celu":                  {},
	"Compress":              {},
	"Cosh":                  {},
	"Clip":                  {},
	"Concat":                {},
	"Constant":              {},
	"ConstantOfShape":       {},
	"Conv":                  {},
	"ConvTranspose":         {},
	"Cos":                   {},
	"CumSum":                {},
	"DeformConv":            {},
	"DepthToSpace":          {},
	"DequantizeLinear":      {},
	"Det":                   {},
	"DFT":                   {},
	"Div":                   {},
	"DynamicQuantizeLinear": {},
	"Dropout":               {},
	"Einsum":                {},
	"Elu":                   {},
	"Equal":                 {},
	"Erf":                   {},
	"Exp":                   {},
	"Expand":                {},
	"EyeLike":               {},
	"FastGELU":              {},
	"Flatten":               {},
	"Floor":                 {},
	"FusedAffine":           {},
	"FusedConv":             {},
	"FusedMatMul":           {},
	"Gather":                {},
	"GatherElements":        {},
	"GatherND":              {},
	"Gelu":                  {},
	"Gemm":                  {},
	"GlobalAveragePool":     {},
	"GlobalMaxPool":         {},
	"GroupNormalization":     {},
	"Greater":               {},
	"GreaterOrEqual":        {},
	"GRU":                   {},
	"GridSample":            {},
	"HardSigmoid":           {},
	"Hardmax":               {},
	"HardSwish":             {},
	"Identity":              {},
	"If":                    {},
	"InstanceNormalization": {},
	"IsInf":                 {},
	"IsNaN":                 {},
	"LpNormalization":       {},
	"LRN":                   {},
	"LayerNormalization":    {},
	"LeakyRelu":             {},
	"Less":                  {},
	"LessOrEqual":           {},
	"LogSoftmax":            {},
	"Loop":                  {},
	"LSTM":                  {},
	"Log":                   {},
	"MatMul":                {},
	"Max":                   {},
	"MaxPool":               {},
	"MaxRoiPool":            {},
	"Mean":                  {},
	"MeanVarianceNormalization": {},
	"MelWeightMatrix":       {},
	"Min":                   {},
	"Mish":                  {},
	"Mod":                   {},
	"Mul":                   {},
	"Neg":                   {},
	"NonMaxSuppression":     {},
	"NonZero":               {},
	"Not":                   {},
	"OneHot":                {},
	"Or":                    {},
	"Pad":                   {},
	"Pow":                   {},
	"PRelu":                 {},
	"QLinearConv":           {},
	"QLinearMatMul":         {},
	"QuantizeLinear":        {},
	"Range":                 {},
	"Reciprocal":            {},
	"ReduceL1":              {},
	"ReduceL2":              {},
	"ReduceMax":             {},
	"ReduceLogSum":          {},
	"ReduceLogSumExp":       {},
	"ReduceMean":            {},
	"ReduceMin":             {},
	"ReduceProd":            {},
	"ReduceSum":             {},
	"ReduceSumSquare":       {},
	"Relu":                  {},
	"Reshape":               {},
	"Resize":                {},
	"RMSNormalization":      {},
	"SimplifiedLayerNormalization": {},
	"RNN":                   {},
	"RoiAlign":              {},
	"RotaryEmbedding":       {},
	"Round":                 {},
	"Scan":                  {},
	"ScatterElements":       {},
	"Shrink":                {},
	"ScatterND":             {},
	"Selu":                  {},
	"Shape":                 {},
	"Sigmoid":               {},
	"Sign":                  {},
	"Sin":                   {},
	"Sinh":                  {},
	"Softplus":              {},
	"Softsign":              {},
	"SpaceToDepth":          {},
	"STFT":                  {},
	"Size":                  {},
	"Slice":                 {},
	"Softmax":               {},
	"Split":                 {},
	"Sqrt":                  {},
	"Sub":                   {},
	"Sum":                   {},
	"Squeeze":               {},
	"Swish":                 {},
	"Tile":                  {},
	"Tan":                   {},
	"Tanh":                  {},
	"ThresholdedRelu":       {},
	"TopK":                  {},
	"Transpose":             {},
	"Trilu":                 {},
	"Unique":                {},
	"Unsqueeze":             {},
	"Upsample":              {},
	"Where":                 {},
	"Xor":                   {},
}

func assertInt64Data(t *testing.T, name string, got []int64, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch %d vs %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: got %d want %d", name, i, got[i], want[i])
		}
	}
}

func assertUint8Data(t *testing.T, name string, got []uint8, want []uint8) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch %d vs %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: got %d want %d", name, i, got[i], want[i])
		}
	}
}

func floatTensorAttr(shape ir.Shape, data []float32) ir.AttrTensor {
	return ir.AttrTensor{Value: &ir.Initializer{
		Name:      "const",
		DType:     ir.DataTypeFloat,
		Shape:     shape,
		FloatData: data,
	}}
}

func int64TensorAttr(shape ir.Shape, data []int64) ir.AttrTensor {
	return ir.AttrTensor{Value: &ir.Initializer{
		Name:      "const",
		DType:     ir.DataTypeInt64,
		Shape:     shape,
		Int64Data: data,
	}}
}

func TestRegisterAllHasUnitCoverage(t *testing.T) {
	r := NewRegistry()
	RegisterAll(r)
	var missing []string
	for name := range r.ops {
		if _, ok := unitTestedOps[name]; !ok {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Fatalf("missing unit coverage for ops: %v", missing)
	}
}

func TestArithmeticAndUnaryOps(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2}, []float32{4, 9})
	b := tensor.NewDense[float32](tensor.Shape{2}, []float32{2, 3})

	out, err := opSub(makeNode("Sub", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Sub", out[0].(*tensor.Dense[float32]).Data(), []float32{2, 6}, 1e-6)

	out, err = opMul(makeNode("Mul", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Mul", out[0].(*tensor.Dense[float32]).Data(), []float32{8, 27}, 1e-6)

	out, err = opDiv(makeNode("Div", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Div", out[0].(*tensor.Dense[float32]).Data(), []float32{2, 3}, 1e-6)

	modA := tensor.NewDense[int64](tensor.Shape{2}, []int64{7, 10})
	modB := tensor.NewDense[int64](tensor.Shape{2}, []int64{3, 4})
	out, err = opMod(makeNode("Mod", nil), []tensor.Tensor{modA, modB})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "Mod", out[0].(*tensor.Dense[int64]).Data(), []int64{1, 2})

	out, err = opLeakyRelu(makeNode("LeakyRelu", map[string]ir.AttrValue{"alpha": ir.AttrFloat{Value: 0.1}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{3}, []float32{-2, 0, 3}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "LeakyRelu", out[0].(*tensor.Dense[float32]).Data(), []float32{-0.2, 0, 3}, 1e-6)

	out, err = opElu(makeNode("Elu", map[string]ir.AttrValue{"alpha": ir.AttrFloat{Value: 1.0}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{-1, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	gotElu := out[0].(*tensor.Dense[float32]).Data()
	if math.Abs(float64(gotElu[0]-float32(math.Exp(-1)+-1))) > 1e-5 || math.Abs(float64(gotElu[1]-1)) > 1e-6 {
		t.Fatalf("Elu unexpected: %v", gotElu)
	}

	for name, fn := range map[string]func(*ir.Node, []tensor.Tensor) ([]tensor.Tensor, error){
		"Sigmoid": opSigmoid,
		"HardSigmoid": func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
			return opHardSigmoid(makeNode("HardSigmoid", map[string]ir.AttrValue{
				"alpha": ir.AttrFloat{Value: 0.2},
				"beta":  ir.AttrFloat{Value: 0.5},
			}), inputs)
		},
		"HardSwish": opHardSwish,
		"Tanh":      opTanh,
		"Log":       opLog,
		"Abs":       opAbs,
		"Sin":       opSin,
		"Cos":       opCos,
		"Sqrt":      opSqrt,
		"Neg":       opNeg,
		"Erf":       opErf,
		"Floor":     opFloor,
		"Exp":       opExp,
	} {
		if _, err := fn(makeNode(name, nil), []tensor.Tensor{
			tensor.NewDense[float32](tensor.Shape{2}, []float32{0.5, 1.5}),
		}); err != nil {
			t.Fatalf("%s failed: %v", name, err)
		}
	}

	out, err = opPow(makeNode("Pow", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{2, 3}),
		tensor.NewDense[float32](tensor.Shape{2}, []float32{3, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Pow", out[0].(*tensor.Dense[float32]).Data(), []float32{8, 9}, 1e-6)

	out, err = opReciprocal(makeNode("Reciprocal", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{2, 4}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Reciprocal", out[0].(*tensor.Dense[float32]).Data(), []float32{0.5, 0.25}, 1e-6)
}

func TestComparisonAndCastOps(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 3})
	b := tensor.NewDense[float32](tensor.Shape{2}, []float32{2, 3})

	out, err := opEqual(makeNode("Equal", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertUint8Data(t, "Equal", out[0].(*tensor.Dense[uint8]).Data(), []uint8{0, 1})

	out, err = opGreater(makeNode("Greater", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertUint8Data(t, "Greater", out[0].(*tensor.Dense[uint8]).Data(), []uint8{0, 0})

	out, err = opLess(makeNode("Less", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertUint8Data(t, "Less", out[0].(*tensor.Dense[uint8]).Data(), []uint8{1, 0})

	out, err = opAnd(makeNode("And", nil), []tensor.Tensor{
		tensor.NewDense[uint8](tensor.Shape{2}, []uint8{1, 0}),
		tensor.NewDense[uint8](tensor.Shape{2}, []uint8{1, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertUint8Data(t, "And", out[0].(*tensor.Dense[uint8]).Data(), []uint8{1, 0})

	out, err = opMin(makeNode("Min", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Min", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 3}, 1e-6)

	out, err = opMax(makeNode("Max", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Max", out[0].(*tensor.Dense[float32]).Data(), []float32{2, 3}, 1e-6)

	out, err = opCast(makeNode("Cast", map[string]ir.AttrValue{"to": ir.AttrInt{Value: 7}}), []tensor.Tensor{a})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "Cast", out[0].(*tensor.Dense[int64]).Data(), []int64{1, 3})

	out, err = opWhere(makeNode("Where", nil), []tensor.Tensor{
		tensor.NewDense[uint8](tensor.Shape{2}, []uint8{1, 0}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{10, 20}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{30, 40}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "Where", out[0].(*tensor.Dense[int64]).Data(), []int64{10, 40})
}

func TestShapeTensorOps(t *testing.T) {
	out, err := opShape(makeNode("Shape", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 3}, make([]float32, 6)),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "Shape", out[0].(*tensor.Dense[int64]).Data(), []int64{2, 3})

	out, err = opSqueeze(makeNode("Squeeze", map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{0}}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 2, 1}, []float32{1, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{2, 1}) {
		t.Fatalf("Squeeze shape: %v", out[0].Shape())
	}

	out, err = opUnsqueeze(makeNode("Unsqueeze", map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{1}}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 3}, make([]float32, 6)),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{2, 1, 3}) {
		t.Fatalf("Unsqueeze shape: %v", out[0].Shape())
	}

	out, err = opConcat(makeNode("Concat", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 0}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 2}, []float32{1, 2}),
		tensor.NewDense[float32](tensor.Shape{1, 2}, []float32{3, 4}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Concat", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2, 3, 4}, 1e-6)

	out, err = opSplit(makeNode("Split", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 1}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 4}, []float32{1, 2, 3, 4}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{2, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || !out[0].Shape().Equal(tensor.Shape{1, 2}) || !out[1].Shape().Equal(tensor.Shape{1, 2}) {
		t.Fatalf("Split unexpected outputs")
	}

	out, err = opExpand(makeNode("Expand", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 2}, []float32{1, 2}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{2, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Expand", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2, 1, 2}, 1e-6)

	out, err = opSize(makeNode("Size", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 3}, make([]float32, 6)),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "Size", out[0].(*tensor.Dense[int64]).Data(), []int64{6})
}

func TestIndexingAndReductionOps(t *testing.T) {
	out, err := opTopK(makeNode("TopK", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 1}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 4}, []float32{1, 4, 3, 2}),
		tensor.NewDense[int64](tensor.Shape{}, []int64{2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "TopK.Values", out[0].(*tensor.Dense[float32]).Data(), []float32{4, 3}, 1e-6)
	assertInt64Data(t, "TopK.Indices", out[1].(*tensor.Dense[int64]).Data(), []int64{1, 2})

	for name, fn := range map[string]func(*ir.Node, []tensor.Tensor) ([]tensor.Tensor, error){
		"ReduceMax":       opReduceMax,
		"ReduceProd":      opReduceProd,
		"ReduceSum":       opReduceSum,
		"ReduceMean":      opReduceMean,
		"ReduceSumSquare": opReduceSumSquare,
	} {
		out, err := fn(makeNode(name, map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{1}}, "keepdims": ir.AttrInt{Value: 0}}), []tensor.Tensor{
			tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{1, 2, 3, 4}),
		})
		if err != nil {
			t.Fatalf("%s failed: %v", name, err)
		}
		if !out[0].Shape().Equal(tensor.Shape{2}) {
			t.Fatalf("%s shape: %v", name, out[0].Shape())
		}
	}

	out, err = opGatherElements(makeNode("GatherElements", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 1}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{1, 2, 3, 4}),
		tensor.NewDense[int64](tensor.Shape{2, 2}, []int64{1, 0, 0, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "GatherElements", out[0].(*tensor.Dense[float32]).Data(), []float32{2, 1, 3, 4}, 1e-6)

	out, err = opGatherND(makeNode("GatherND", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{1, 2, 3, 4}),
		tensor.NewDense[int64](tensor.Shape{2, 2}, []int64{0, 0, 1, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "GatherND", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 4}, 1e-6)

	out, err = opScatterND(makeNode("ScatterND", nil), []tensor.Tensor{
		tensor.NewDense[int64](tensor.Shape{3}, []int64{0, 0, 0}),
		tensor.NewDense[int64](tensor.Shape{2, 1}, []int64{0, 2}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{5, 7}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "ScatterND", out[0].(*tensor.Dense[int64]).Data(), []int64{5, 0, 7})

	out, err = opArgMax(makeNode("ArgMax", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 1}, "keepdims": ir.AttrInt{Value: 0}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{1, 3, 5, 4}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "ArgMax", out[0].(*tensor.Dense[int64]).Data(), []int64{1, 0})

	out, err = opCumSum(makeNode("CumSum", map[string]ir.AttrValue{"exclusive": ir.AttrInt{Value: 0}}), []tensor.Tensor{
		tensor.NewDense[int32](tensor.Shape{3}, []int32{1, 2, 3}),
		tensor.NewDense[int32](tensor.Shape{}, []int32{0}),
	})
	if err != nil {
		t.Fatal(err)
	}
	gotCum := out[0].(*tensor.Dense[int32]).Data()
	if gotCum[0] != 1 || gotCum[1] != 3 || gotCum[2] != 6 {
		t.Fatalf("CumSum unexpected: %v", gotCum)
	}
}

func TestSamplingAndPaddingOps(t *testing.T) {
	src := tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 2, 3, 4})

	out, err := opSlice(makeNode("Slice", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		tensor.NewDense[int64](tensor.Shape{1}, []int64{0}),
		tensor.NewDense[int64](tensor.Shape{1}, []int64{1}),
		tensor.NewDense[int64](tensor.Shape{1}, []int64{0}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Slice", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2, 3}, 1e-6)

	out, err = opPad(makeNode("Pad", nil), []tensor.Tensor{
		tensor.NewDense[int64](tensor.Shape{2}, []int64{1, 2}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{1, 1}),
	})
	_ = out
	_ = err

	out, err = opPad(makeNode("Pad", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 2}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{1, 1}),
		tensor.NewDense[float32](tensor.Shape{}, []float32{0}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Pad", out[0].(*tensor.Dense[float32]).Data(), []float32{0, 1, 2, 0}, 1e-6)

	out, err = opTile(makeNode("Tile", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 2}),
		tensor.NewDense[int64](tensor.Shape{1}, []int64{2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Tile", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2, 1, 2}, 1e-6)

	out, err = opResize(makeNode("Resize", map[string]ir.AttrValue{"mode": ir.AttrString{Value: "nearest"}}), []tensor.Tensor{
		src,
		nil,
		tensor.NewDense[float32](tensor.Shape{4}, []float32{1, 1, 2, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 4, 4}) {
		t.Fatalf("Resize shape: %v", out[0].Shape())
	}

	out, err = opUpsample(makeNode("Upsample", map[string]ir.AttrValue{"mode": ir.AttrString{Value: "nearest"}}), []tensor.Tensor{
		src,
		tensor.NewDense[float32](tensor.Shape{4}, []float32{1, 1, 2, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 4, 4}) {
		t.Fatalf("Upsample shape: %v", out[0].Shape())
	}

	grid := tensor.NewDense[float32](tensor.Shape{1, 1, 1, 2}, []float32{0, 0})
	out, err = opGridSample(makeNode("GridSample", nil), []tensor.Tensor{src, grid})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 1, 1}) {
		t.Fatalf("GridSample shape: %v", out[0].Shape())
	}
}

func TestNormConvPoolFusedAndDataOps(t *testing.T) {
	constNode := makeNode("Constant", map[string]ir.AttrValue{
		"value": floatTensorAttr(ir.Shape{2}, []float32{1, 2}),
	})
	out, err := opConstant(constNode, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Constant", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2}, 1e-6)

	out, err = opIdentity(makeNode("Identity", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Identity", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2}, 1e-6)

	out, err = opDropout(makeNode("Dropout", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Dropout", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2}, 1e-6)

	out, err = opClip(makeNode("Clip", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{3}, []float32{-1, 0.5, 2}),
		tensor.NewDense[float32](tensor.Shape{}, []float32{0}),
		tensor.NewDense[float32](tensor.Shape{}, []float32{1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Clip", out[0].(*tensor.Dense[float32]).Data(), []float32{0, 0.5, 1}, 1e-6)

	x := tensor.NewDense[float32](tensor.Shape{1, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	w := tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 0, 0, 1})
	out, err = opFusedConv(makeNode("FusedConv", map[string]ir.AttrValue{
		"strides":    ir.AttrInts{Value: []int64{1, 1}},
		"pads":       ir.AttrInts{Value: []int64{0, 0, 0, 0}},
		"activation": ir.AttrString{Value: "relu"},
	}), []tensor.Tensor{x, w})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 2, 2}) {
		t.Fatalf("FusedConv shape: %v", out[0].Shape())
	}

	out, err = opConvTranspose(makeNode("ConvTranspose", map[string]ir.AttrValue{
		"strides": ir.AttrInts{Value: []int64{1, 1}},
	}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 2, 3, 4}),
		tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 0, 0, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 3, 3}) {
		t.Fatalf("ConvTranspose shape: %v", out[0].Shape())
	}

	out, err = opAveragePool(makeNode("AveragePool", map[string]ir.AttrValue{
		"kernel_shape": ir.AttrInts{Value: []int64{2, 2}},
		"strides":      ir.AttrInts{Value: []int64{2, 2}},
	}), []tensor.Tensor{x})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "AveragePool", out[0].(*tensor.Dense[float32]).Data(), []float32{3}, 1e-6)

	for name, fn := range map[string]func(*ir.Node, []tensor.Tensor) ([]tensor.Tensor, error){
		"LRN": opLRN,
	} {
		if _, err := fn(makeNode(name, map[string]ir.AttrValue{"size": ir.AttrInt{Value: 1}}), []tensor.Tensor{
			tensor.NewDense[float32](tensor.Shape{1, 1, 1, 1}, []float32{2}),
		}); err != nil {
			t.Fatalf("%s failed: %v", name, err)
		}
	}

	out, err = opInstanceNormalization(makeNode("InstanceNormalization", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 2, 3, 4}),
		tensor.NewDense[float32](tensor.Shape{1}, []float32{1}),
		tensor.NewDense[float32](tensor.Shape{1}, []float32{0}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 2, 2}) {
		t.Fatalf("InstanceNormalization shape: %v", out[0].Shape())
	}

	out, err = opLayerNormalization(makeNode("LayerNormalization", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{1, 2, 3, 4}),
		tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 1}),
		tensor.NewDense[float32](tensor.Shape{2}, []float32{0, 0}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{2, 2}) {
		t.Fatalf("LayerNormalization shape: %v", out[0].Shape())
	}
}

func TestGatherRangeControlFusedAndLSTM(t *testing.T) {
	out, err := opGather(makeNode("Gather", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 0}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{3, 2}, []float32{1, 2, 3, 4, 5, 6}),
		tensor.NewDense[int64](tensor.Shape{2}, []int64{0, 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Gather", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2, 5, 6}, 1e-6)

	out, err = opRange(makeNode("Range", nil), []tensor.Tensor{
		tensor.NewDense[int64](tensor.Shape{}, []int64{1}),
		tensor.NewDense[int64](tensor.Shape{}, []int64{5}),
		tensor.NewDense[int64](tensor.Shape{}, []int64{2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "Range", out[0].(*tensor.Dense[int64]).Data(), []int64{1, 3})

	out, err = opNot(makeNode("Not", nil), []tensor.Tensor{
		tensor.NewDense[uint8](tensor.Shape{3}, []uint8{1, 0, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertUint8Data(t, "Not", out[0].(*tensor.Dense[uint8]).Data(), []uint8{0, 1, 0})

	out, err = opOneHot(makeNode("OneHot", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: -1}}), []tensor.Tensor{
		tensor.NewDense[int64](tensor.Shape{2}, []int64{0, 2}),
		tensor.NewDense[int64](tensor.Shape{}, []int64{3}),
		tensor.NewDense[float32](tensor.Shape{2}, []float32{0, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "OneHot", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 0, 0, 0, 0, 1}, 1e-6)

	if _, err := opIf(makeNode("If", nil), nil); err == nil {
		t.Fatal("If should return runtime-handled error")
	}

	out, err = opConstantOfShape(makeNode("ConstantOfShape", map[string]ir.AttrValue{
		"value": int64TensorAttr(ir.Shape{1}, []int64{7}),
	}), []tensor.Tensor{
		tensor.NewDense[int64](tensor.Shape{2}, []int64{2, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertInt64Data(t, "ConstantOfShape", out[0].(*tensor.Dense[int64]).Data(), []int64{7, 7})

	out, err = opFastGELU(makeNode("FastGELU", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{-1, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{2}) {
		t.Fatalf("FastGELU shape: %v", out[0].Shape())
	}

	out, err = opFusedMatMul(makeNode("FusedMatMul", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 2}, []float32{1, 2}),
		tensor.NewDense[float32](tensor.Shape{2, 1}, []float32{3, 4}),
		tensor.NewDense[float32](tensor.Shape{1}, []float32{5}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "FusedMatMul", out[0].(*tensor.Dense[float32]).Data(), []float32{16}, 1e-6)

	out, err = opFusedAffine(makeNode("FusedAffine", nil), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 2}),
		tensor.NewDense[float32](tensor.Shape{2}, []float32{2, 3}),
		tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "FusedAffine", out[0].(*tensor.Dense[float32]).Data(), []float32{3, 7}, 1e-6)

	out, err = opLSTM(makeNode("LSTM", map[string]ir.AttrValue{"hidden_size": ir.AttrInt{Value: 1}}), []tensor.Tensor{
		tensor.NewDense[float32](tensor.Shape{1, 1, 1}, []float32{1}),
		tensor.NewDense[float32](tensor.Shape{1, 4, 1}, []float32{1, 1, 1, 1}),
		tensor.NewDense[float32](tensor.Shape{1, 4, 1}, []float32{0, 0, 0, 0}),
		tensor.NewDense[float32](tensor.Shape{1, 8}, make([]float32, 8)),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 || !out[0].Shape().Equal(tensor.Shape{1, 1, 1, 1}) {
		t.Fatalf("LSTM outputs unexpected")
	}
}

func TestGRU(t *testing.T) {
	// X[seq=1, batch=1, input=2], W[1, 6, 2], R[1, 6, 2], B[1, 12]
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 2}, []float32{0.5, -0.3})
	w := tensor.NewDense[float32](tensor.Shape{1, 6, 2}, make([]float32, 12)) // zeros
	r := tensor.NewDense[float32](tensor.Shape{1, 6, 2}, make([]float32, 12))
	b := tensor.NewDense[float32](tensor.Shape{1, 12}, make([]float32, 12))
	node := makeNode("GRU", map[string]ir.AttrValue{
		"hidden_size": ir.AttrInt{Value: 2},
	})
	out, err := opGRU(node, []tensor.Tensor{x, w, r, b, nil, nil})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(out))
	}
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 1, 2}) {
		t.Fatalf("Y shape: got %v, want [1,1,1,2]", out[0].Shape())
	}
}

func TestPRelu(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{4}, []float32{-2, -1, 0, 1})
	slope := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.25})
	out, err := opPRelu(makeNode("PRelu", nil), []tensor.Tensor{x, slope})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "PRelu", out[0].(*tensor.Dense[float32]).Data(), []float32{-0.5, -0.25, 0, 1}, 1e-6)
}

func TestGreaterOrEqual(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{3}, []float32{1, 2, 3})
	b := tensor.NewDense[float32](tensor.Shape{3}, []float32{2, 2, 1})
	out, err := opGreaterOrEqual(makeNode("GreaterOrEqual", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	got := out[0].(*tensor.Dense[uint8]).Data()
	want := []uint8{0, 1, 1}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("GreaterOrEqual[%d]: got %d want %d", i, got[i], want[i])
		}
	}
}

func TestNonZero(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{0, 1, 0, 2, 0, 3})
	out, err := opNonZero(makeNode("NonZero", nil), []tensor.Tensor{x})
	if err != nil {
		t.Fatal(err)
	}
	// Non-zero at: (0,1), (1,0), (1,2) → output shape [2, 3]
	result := out[0].(*tensor.Dense[int64])
	if !result.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("NonZero shape: got %v, want [2,3]", result.Shape())
	}
}

func TestOr(t *testing.T) {
	a := tensor.NewDense[uint8](tensor.Shape{3}, []uint8{0, 1, 0})
	b := tensor.NewDense[uint8](tensor.Shape{3}, []uint8{0, 0, 1})
	out, err := opOr(makeNode("Or", nil), []tensor.Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	got := out[0].(*tensor.Dense[uint8]).Data()
	want := []uint8{0, 1, 1}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("Or[%d]: got %d want %d", i, got[i], want[i])
		}
	}
}

func TestNonMaxSuppression(t *testing.T) {
	// 1 batch, 4 boxes, 1 class
	boxes := tensor.NewDense[float32](tensor.Shape{1, 4, 4}, []float32{
		0, 0, 1, 1, // box0
		0, 0, 1, 1, // box1 (same as box0 → suppressed)
		0, 0, 0.5, 0.5, // box2 (partial overlap)
		2, 2, 3, 3, // box3 (no overlap)
	})
	scores := tensor.NewDense[float32](tensor.Shape{1, 1, 4}, []float32{0.9, 0.8, 0.7, 0.6})
	maxBoxes := tensor.NewDenseScalar[int64](10)
	iouThresh := tensor.NewDenseScalar[float32](0.5)
	scoreThresh := tensor.NewDenseScalar[float32](0.0)

	out, err := opNonMaxSuppression(
		makeNode("NonMaxSuppression", nil),
		[]tensor.Tensor{boxes, scores, maxBoxes, iouThresh, scoreThresh},
	)
	if err != nil {
		t.Fatal(err)
	}
	result := out[0].(*tensor.Dense[int64])
	// box0 (0.9) selected, box1 (0.8) suppressed by box0, box2 (0.7) selected, box3 (0.6) selected
	// → 3 selections
	if result.Shape()[0] != 3 {
		t.Fatalf("NMS: expected 3 selected, got %d (shape=%v data=%v)", result.Shape()[0], result.Shape(), result.Data())
	}
}

func TestMathExtraOps(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{1.3, -2.7, 0.5})

	// Ceil
	out, err := opCeil(makeNode("Ceil", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	assertClose(t, "Ceil", out[0].(*tensor.Dense[float32]).Data(), []float32{2, -2, 1}, 1e-6)

	// Round (banker's rounding)
	out, err = opRound(makeNode("Round", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	assertClose(t, "Round", out[0].(*tensor.Dense[float32]).Data(), []float32{1, -3, 0}, 1e-6)

	// Sign
	out, err = opSign(makeNode("Sign", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	assertClose(t, "Sign", out[0].(*tensor.Dense[float32]).Data(), []float32{1, -1, 1}, 1e-6)

	// Tan
	y := tensor.NewDense[float32](tensor.Shape{1}, []float32{0})
	out, err = opTan(makeNode("Tan", nil), []tensor.Tensor{y})
	if err != nil { t.Fatal(err) }
	assertClose(t, "Tan", out[0].(*tensor.Dense[float32]).Data(), []float32{0}, 1e-6)

	// Sinh/Cosh
	out, err = opSinh(makeNode("Sinh", nil), []tensor.Tensor{y})
	if err != nil { t.Fatal(err) }
	assertClose(t, "Sinh", out[0].(*tensor.Dense[float32]).Data(), []float32{0}, 1e-6)
	out, err = opCosh(makeNode("Cosh", nil), []tensor.Tensor{y})
	if err != nil { t.Fatal(err) }
	assertClose(t, "Cosh", out[0].(*tensor.Dense[float32]).Data(), []float32{1}, 1e-6)

	// Asin/Acos/Atan
	z := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.5})
	out, err = opAsin(makeNode("Asin", nil), []tensor.Tensor{z})
	if err != nil { t.Fatal(err) }
	if len(out[0].(*tensor.Dense[float32]).Data()) != 1 { t.Fatal("Asin: wrong length") }
	out, err = opAcos(makeNode("Acos", nil), []tensor.Tensor{z})
	if err != nil { t.Fatal(err) }
	out, err = opAtan(makeNode("Atan", nil), []tensor.Tensor{z})
	if err != nil { t.Fatal(err) }
}

func TestComparisonExtraOps(t *testing.T) {
	a := tensor.NewDense[float32](tensor.Shape{3}, []float32{1, 2, 3})
	b := tensor.NewDense[float32](tensor.Shape{3}, []float32{2, 2, 1})
	out, err := opLessOrEqual(makeNode("LessOrEqual", nil), []tensor.Tensor{a, b})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[uint8]).Data()
	want := []uint8{1, 1, 0}
	for i := range got { if got[i] != want[i] { t.Fatalf("LessOrEqual[%d]: got %d want %d", i, got[i], want[i]) } }
}

func TestActivationExtraOps(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{-1, 0, 1})

	out, _ := opGelu(makeNode("Gelu", nil), []tensor.Tensor{x})
	if out[0].Len() != 3 { t.Fatal("Gelu: wrong length") }

	out, _ = opMish(makeNode("Mish", nil), []tensor.Tensor{x})
	if out[0].Len() != 3 { t.Fatal("Mish: wrong length") }

	out, _ = opSelu(makeNode("Selu", nil), []tensor.Tensor{x})
	if out[0].Len() != 3 { t.Fatal("Selu: wrong length") }

	out, _ = opSoftplus(makeNode("Softplus", nil), []tensor.Tensor{x})
	d := out[0].(*tensor.Dense[float32]).Data()
	for _, v := range d { if v < 0 { t.Fatal("Softplus: negative value") } }

	out, _ = opSoftsign(makeNode("Softsign", nil), []tensor.Tensor{x})
	if out[0].Len() != 3 { t.Fatal("Softsign: wrong length") }

	sm := tensor.NewDense[float32](tensor.Shape{1, 3}, []float32{1, 2, 3})
	out, _ = opLogSoftmax(makeNode("LogSoftmax", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: -1}}), []tensor.Tensor{sm})
	ld := out[0].(*tensor.Dense[float32]).Data()
	for _, v := range ld { if v > 0 { t.Fatal("LogSoftmax: positive value") } }
}

func TestPoolingExtraOps(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 3, 2, 4})
	out, err := opGlobalMaxPool(makeNode("GlobalMaxPool", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	assertClose(t, "GlobalMaxPool", out[0].(*tensor.Dense[float32]).Data(), []float32{4}, 1e-6)
}

func TestShapeExtraOps(t *testing.T) {
	// DepthToSpace: [1, 4, 1, 1] bs=2 → [1, 1, 2, 2]
	x := tensor.NewDense[float32](tensor.Shape{1, 4, 1, 1}, []float32{1, 2, 3, 4})
	out, err := opDepthToSpace(makeNode("DepthToSpace", map[string]ir.AttrValue{"blocksize": ir.AttrInt{Value: 2}}), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if !out[0].Shape().Equal(tensor.Shape{1, 1, 2, 2}) { t.Fatalf("DepthToSpace: shape %v", out[0].Shape()) }

	// SpaceToDepth: [1, 1, 2, 2] bs=2 → [1, 4, 1, 1]
	y := tensor.NewDense[float32](tensor.Shape{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	out, err = opSpaceToDepth(makeNode("SpaceToDepth", map[string]ir.AttrValue{"blocksize": ir.AttrInt{Value: 2}}), []tensor.Tensor{y})
	if err != nil { t.Fatal(err) }
	if !out[0].Shape().Equal(tensor.Shape{1, 4, 1, 1}) { t.Fatalf("SpaceToDepth: shape %v", out[0].Shape()) }
}

func TestReductionExtraOps(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{3, 1, 2, 6, 4, 5})

	out, _ := opReduceMin(makeNode("ReduceMin", map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{1}}, "keepdims": ir.AttrInt{Value: 0}}), []tensor.Tensor{x})
	assertClose(t, "ReduceMin", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 4}, 1e-6)

	out, _ = opReduceL1(makeNode("ReduceL1", map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{1}}, "keepdims": ir.AttrInt{Value: 0}}), []tensor.Tensor{x})
	assertClose(t, "ReduceL1", out[0].(*tensor.Dense[float32]).Data(), []float32{6, 15}, 1e-6)

	out, _ = opReduceL2(makeNode("ReduceL2", map[string]ir.AttrValue{"axes": ir.AttrInts{Value: []int64{1}}, "keepdims": ir.AttrInt{Value: 0}}), []tensor.Tensor{x})
	if out[0].Len() != 2 { t.Fatal("ReduceL2: wrong length") }

	a := tensor.NewDense[float32](tensor.Shape{2}, []float32{1, 2})
	b := tensor.NewDense[float32](tensor.Shape{2}, []float32{3, 4})
	out, _ = opSum(makeNode("Sum", nil), []tensor.Tensor{a, b})
	assertClose(t, "Sum", out[0].(*tensor.Dense[float32]).Data(), []float32{4, 6}, 1e-6)

	out, _ = opMean(makeNode("Mean", nil), []tensor.Tensor{a, b})
	assertClose(t, "Mean", out[0].(*tensor.Dense[float32]).Data(), []float32{2, 3}, 1e-6)
}

func TestIndexingExtraOps(t *testing.T) {
	data := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	indices := tensor.NewDense[int64](tensor.Shape{2, 1}, []int64{1, 0})
	updates := tensor.NewDense[float32](tensor.Shape{2, 1}, []float32{99, 88})
	out, err := opScatterElements(makeNode("ScatterElements", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 1}}), []tensor.Tensor{data, indices, updates})
	if err != nil { t.Fatal(err) }
	// row0: [1, 99, 3], row1: [88, 5, 6]
	assertClose(t, "ScatterElements", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 99, 3, 88, 5, 6}, 1e-6)
}

func TestTrilu(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	// Upper triangle
	out, _ := opTrilu(makeNode("Trilu", map[string]ir.AttrValue{"upper": ir.AttrInt{Value: 1}}), []tensor.Tensor{x, nil})
	assertClose(t, "Trilu_upper", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 2, 3, 0, 5, 6, 0, 0, 9}, 1e-6)
	// Lower triangle
	out, _ = opTrilu(makeNode("Trilu", map[string]ir.AttrValue{"upper": ir.AttrInt{Value: 0}}), []tensor.Tensor{x, nil})
	assertClose(t, "Trilu_lower", out[0].(*tensor.Dense[float32]).Data(), []float32{1, 0, 0, 4, 5, 0, 7, 8, 9}, 1e-6)
}

func TestNormExtraOps(t *testing.T) {
	// GroupNorm: [1, 4, 1, 1] with 2 groups
	x := tensor.NewDense[float32](tensor.Shape{1, 4, 1, 1}, []float32{1, 2, 3, 4})
	scale := tensor.NewDense[float32](tensor.Shape{4}, []float32{1, 1, 1, 1})
	bias := tensor.NewDense[float32](tensor.Shape{4}, []float32{0, 0, 0, 0})
	out, err := opGroupNormalization(makeNode("GroupNormalization", map[string]ir.AttrValue{"num_groups": ir.AttrInt{Value: 2}}), []tensor.Tensor{x, scale, bias})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 4 { t.Fatal("GroupNorm: wrong length") }

	// RMSNorm: [1, 3]
	y := tensor.NewDense[float32](tensor.Shape{1, 3}, []float32{1, 2, 3})
	s := tensor.NewDense[float32](tensor.Shape{3}, []float32{1, 1, 1})
	out, err = opRMSNormalization(makeNode("RMSNormalization", nil), []tensor.Tensor{y, s})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 3 { t.Fatal("RMSNorm: wrong length") }
}

func TestEinsum(t *testing.T) {
	// Matrix multiply: ij,jk->ik
	a := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := tensor.NewDense[float32](tensor.Shape{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	out, err := opEinsum(makeNode("Einsum", map[string]ir.AttrValue{"equation": ir.AttrString{Value: "ij,jk->ik"}}), []tensor.Tensor{a, b})
	if err != nil { t.Fatal(err) }
	assertClose(t, "Einsum_matmul", out[0].(*tensor.Dense[float32]).Data(), []float32{22, 28, 49, 64}, 1e-5)
}

func TestRNNOp(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 2}, []float32{0.5, -0.3})
	w := tensor.NewDense[float32](tensor.Shape{1, 1, 2}, []float32{0.1, 0.2})
	r := tensor.NewDense[float32](tensor.Shape{1, 1, 1}, []float32{0.3})
	out, err := opRNN(makeNode("RNN", map[string]ir.AttrValue{"hidden_size": ir.AttrInt{Value: 1}}), []tensor.Tensor{x, w, r, nil, nil, nil})
	if err != nil { t.Fatal(err) }
	if len(out) != 2 { t.Fatalf("RNN: expected 2 outputs, got %d", len(out)) }
}

func TestDequantizeLinear(t *testing.T) {
	x := tensor.NewDense[uint8](tensor.Shape{3}, []uint8{0, 128, 255})
	scale := tensor.NewDenseScalar[float32](0.01)
	zp := tensor.NewDenseScalar[uint8](128)
	out, err := opDequantizeLinear(makeNode("DequantizeLinear", nil), []tensor.Tensor{x, scale, zp})
	if err != nil { t.Fatal(err) }
	d := out[0].(*tensor.Dense[float32]).Data()
	assertClose(t, "Dequant", d, []float32{-1.28, 0, 1.27}, 1e-5)
}

func TestQuantizeLinear(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{0, 0.5, 1.0})
	scale := tensor.NewDenseScalar[float32](0.01)
	zp := tensor.NewDenseScalar[uint8](0)
	out, err := opQuantizeLinear(makeNode("QuantizeLinear", nil), []tensor.Tensor{x, scale, zp})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 3 { t.Fatal("QuantizeLinear: wrong length") }
}

// Loop and Scan stubs return error (actual execution is handled by engine runtime via LoopControl/ScanControl)
func TestLoopScanStubs(t *testing.T) {
	dummy := tensor.NewDense[float32](tensor.Shape{1}, []float32{0})
	stubs := []struct{ name string; fn OpFunc }{
		{"Loop", opLoop}, {"Scan", opScan},
	}
	for _, s := range stubs {
		_, err := s.fn(makeNode(s.name, nil), []tensor.Tensor{dummy})
		if err == nil { t.Errorf("%s: expected error (stub dispatched by engine runtime)", s.name) }
	}
}

func TestArgMin(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{3, 1, 2, 6, 4, 5})
	out, err := opArgMin(makeNode("ArgMin", map[string]ir.AttrValue{
		"axis": ir.AttrInt{Value: 1}, "keepdims": ir.AttrInt{Value: 0},
	}), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[int64]).Data()
	want := []int64{1, 1}
	assertInt64Data(t, "ArgMin", got, want)
}

func TestIsNaN(t *testing.T) {
	nan := float32(math.NaN())
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{1.0, nan, 0.0})
	out, err := opIsNaN(makeNode("IsNaN", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[uint8]).Data()
	if got[0] != 0 || got[1] != 1 || got[2] != 0 { t.Fatalf("IsNaN: got %v", got) }
}

func TestIsInf(t *testing.T) {
	inf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{inf, ninf, 0.0})
	out, err := opIsInf(makeNode("IsInf", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[uint8]).Data()
	if got[0] != 1 || got[1] != 1 || got[2] != 0 { t.Fatalf("IsInf: got %v", got) }
}

func TestCastLike(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2}, []float32{1.5, 2.5})
	target := tensor.NewDense[int64](tensor.Shape{1}, []int64{0})
	out, err := opCastLike(makeNode("CastLike", nil), []tensor.Tensor{x, target})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[int64]).Data()
	if got[0] != 1 || got[1] != 2 { t.Fatalf("CastLike: got %v", got) }
}

func TestAttention(t *testing.T) {
	// B=1, S=1, E=2, numHeads=1
	input := tensor.NewDense[float32](tensor.Shape{1, 1, 2}, []float32{1, 0})
	weights := tensor.NewDense[float32](tensor.Shape{2, 6}, []float32{
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
	})
	bias := tensor.NewDense[float32](tensor.Shape{6}, []float32{0, 0, 0, 0, 0, 0})
	out, err := opAttention(makeNode("Attention", map[string]ir.AttrValue{
		"num_heads": ir.AttrInt{Value: 1},
	}), []tensor.Tensor{input, weights, bias})
	if err != nil { t.Fatal(err) }
	if out[0].Shape()[0] != 1 || out[0].Shape()[1] != 1 || out[0].Shape()[2] != 2 {
		t.Fatalf("Attention: wrong shape %v", out[0].Shape())
	}
}

func TestRotaryEmbedding(t *testing.T) {
	// B=1, S=1, N=1, H=4
	input := tensor.NewDense[float32](tensor.Shape{1, 1, 1, 4}, []float32{1, 2, 3, 4})
	posIds := tensor.NewDense[int64](tensor.Shape{1, 1}, []int64{0})
	cosCache := tensor.NewDense[float32](tensor.Shape{1, 2}, []float32{1, 1})
	sinCache := tensor.NewDense[float32](tensor.Shape{1, 2}, []float32{0, 0})
	out, err := opRotaryEmbedding(makeNode("RotaryEmbedding", nil),
		[]tensor.Tensor{input, posIds, cosCache, sinCache})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[float32]).Data()
	// cos=1, sin=0 → identity
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 { t.Fatalf("RotaryEmbedding: got %v", got) }
	}
}

func TestQLinearMatMul(t *testing.T) {
	a := tensor.NewDense[uint8](tensor.Shape{2, 2}, []uint8{128, 128, 128, 128})
	aScale := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.01})
	aZp := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{128})
	b := tensor.NewDense[uint8](tensor.Shape{2, 2}, []uint8{128, 128, 128, 128})
	bScale := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.01})
	bZp := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{128})
	yScale := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.01})
	yZp := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{128})
	out, err := opQLinearMatMul(makeNode("QLinearMatMul", nil),
		[]tensor.Tensor{a, aScale, aZp, b, bScale, bZp, yScale, yZp})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 4 { t.Fatalf("QLinearMatMul: wrong len %d", out[0].Len()) }
}

func TestQLinearConv(t *testing.T) {
	// 1x1x3x3 input, 1x1x1x1 kernel → should work
	x := tensor.NewDense[uint8](tensor.Shape{1, 1, 3, 3}, make([]uint8, 9))
	xScale := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.01})
	xZp := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{0})
	w := tensor.NewDense[uint8](tensor.Shape{1, 1, 1, 1}, []uint8{128})
	wScale := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.01})
	wZp := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{0})
	yScale := tensor.NewDense[float32](tensor.Shape{1}, []float32{0.01})
	yZp := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{0})
	out, err := opQLinearConv(makeNode("QLinearConv", nil),
		[]tensor.Tensor{x, xScale, xZp, w, wScale, wZp, yScale, yZp, nil})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 9 { t.Fatalf("QLinearConv: wrong len %d", out[0].Len()) }
}

func TestAcoshAsinhAtanh(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2}, []float32{2, 3})
	out, err := opAcosh(makeNode("Acosh", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 2 { t.Fatal("Acosh: wrong len") }

	x2 := tensor.NewDense[float32](tensor.Shape{2}, []float32{0.5, -0.5})
	out, err = opAsinh(makeNode("Asinh", nil), []tensor.Tensor{x2})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 2 { t.Fatal("Asinh: wrong len") }

	x3 := tensor.NewDense[float32](tensor.Shape{2}, []float32{0.5, -0.5})
	out, err = opAtanh(makeNode("Atanh", nil), []tensor.Tensor{x3})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 2 { t.Fatal("Atanh: wrong len") }
}

func TestCelu(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{-1, 0, 1})
	out, err := opCelu(makeNode("Celu", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 3 { t.Fatal("Celu: wrong len") }
}

func TestHardmax(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 3}, []float32{1, 3, 2})
	out, err := opHardmax(makeNode("Hardmax", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: -1}}), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[float32]).Data()
	if got[0] != 0 || got[1] != 1 || got[2] != 0 { t.Fatalf("Hardmax: got %v", got) }
}

func TestShrink(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{-2, 0, 2})
	out, err := opShrink(makeNode("Shrink", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 3 { t.Fatal("Shrink: wrong len") }
}

func TestThresholdedRelu(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{-1, 0.5, 2})
	out, err := opThresholdedRelu(makeNode("ThresholdedRelu", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[float32]).Data()
	if got[0] != 0 || got[1] != 0 || got[2] != 2 { t.Fatalf("ThresholdedRelu: got %v", got) }
}

func TestSwish(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{-1, 0, 1})
	out, err := opSwish(makeNode("Swish", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 3 { t.Fatal("Swish: wrong len") }
}

func TestXor(t *testing.T) {
	a := tensor.NewDense[uint8](tensor.Shape{3}, []uint8{1, 0, 1})
	b := tensor.NewDense[uint8](tensor.Shape{3}, []uint8{1, 1, 0})
	out, err := opXor(makeNode("Xor", nil), []tensor.Tensor{a, b})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[uint8]).Data()
	if got[0] != 0 || got[1] != 1 || got[2] != 1 { t.Fatalf("Xor: got %v", got) }
}

func TestReduceLogSum(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	out, err := opReduceLogSum(makeNode("ReduceLogSum", map[string]ir.AttrValue{
		"axes": ir.AttrInts{Value: []int64{1}}, "keepdims": ir.AttrInt{Value: 0},
	}), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 2 { t.Fatal("ReduceLogSum: wrong len") }
}

func TestReduceLogSumExp(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2, 3}, []float32{0, 0, 0, 0, 0, 0})
	out, err := opReduceLogSumExp(makeNode("ReduceLogSumExp", map[string]ir.AttrValue{
		"axes": ir.AttrInts{Value: []int64{1}}, "keepdims": ir.AttrInt{Value: 0},
	}), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 2 { t.Fatal("ReduceLogSumExp: wrong len") }
}

func TestEyeLike(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3, 3}, make([]float32, 9))
	out, err := opEyeLike(makeNode("EyeLike", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[float32]).Data()
	if got[0] != 1 || got[4] != 1 || got[8] != 1 || got[1] != 0 { t.Fatalf("EyeLike: got %v", got) }
}

func TestCompress(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{1, 2, 3})
	cond := tensor.NewDense[uint8](tensor.Shape{3}, []uint8{1, 0, 1})
	out, err := opCompress(makeNode("Compress", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: -1}}), []tensor.Tensor{x, cond})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[float32]).Data()
	if len(got) != 2 || got[0] != 1 || got[1] != 3 { t.Fatalf("Compress: got %v", got) }
}

func TestUnique(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{5}, []float32{1, 2, 1, 3, 2})
	out, err := opUnique(makeNode("Unique", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 3 { t.Fatalf("Unique: expected 3 unique, got %d", out[0].Len()) }
}

func TestDet(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{2, 2}, []float32{1, 2, 3, 4})
	out, err := opDet(makeNode("Det", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	got := out[0].(*tensor.Dense[float32]).Data()
	if math.Abs(float64(got[0])-(-2)) > 0.01 { t.Fatalf("Det: got %v, want -2", got[0]) }
}

func TestDynamicQuantizeLinear(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{0, 0.5, 1.0})
	out, err := opDynamicQuantizeLinear(makeNode("DynamicQuantizeLinear", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if len(out) != 3 { t.Fatal("DynamicQuantizeLinear: expected 3 outputs") }
}

func TestBitwiseOps(t *testing.T) {
	a := tensor.NewDense[uint8](tensor.Shape{2}, []uint8{0xFF, 0x0F})
	b := tensor.NewDense[uint8](tensor.Shape{2}, []uint8{0x0F, 0xF0})
	out, _ := opBitwiseAnd(makeNode("BitwiseAnd", nil), []tensor.Tensor{a, b})
	if out[0].(*tensor.Dense[uint8]).Data()[0] != 0x0F { t.Fatal("BitwiseAnd failed") }
	out, _ = opBitwiseOr(makeNode("BitwiseOr", nil), []tensor.Tensor{a, b})
	if out[0].(*tensor.Dense[uint8]).Data()[0] != 0xFF { t.Fatal("BitwiseOr failed") }
	out, _ = opBitwiseXor(makeNode("BitwiseXor", nil), []tensor.Tensor{a, b})
	if out[0].(*tensor.Dense[uint8]).Data()[0] != 0xF0 { t.Fatal("BitwiseXor failed") }
	out, _ = opBitwiseNot(makeNode("BitwiseNot", nil), []tensor.Tensor{tensor.NewDense[uint8](tensor.Shape{1}, []uint8{0x0F})})
	if out[0].(*tensor.Dense[uint8]).Data()[0] != 0xF0 { t.Fatal("BitwiseNot failed") }
}

func TestBitShift(t *testing.T) {
	a := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{4})
	b := tensor.NewDense[uint8](tensor.Shape{1}, []uint8{2})
	out, _ := opBitShift(makeNode("BitShift", map[string]ir.AttrValue{"direction": ir.AttrString{Value: "LEFT"}}), []tensor.Tensor{a, b})
	if out[0].(*tensor.Dense[uint8]).Data()[0] != 16 { t.Fatal("BitShift LEFT failed") }
}

func TestLpNormalization(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{3}, []float32{3, 4, 0})
	out, err := opLpNormalization(makeNode("LpNormalization", map[string]ir.AttrValue{"axis": ir.AttrInt{Value: 0}, "p": ir.AttrInt{Value: 2}}), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 3 { t.Fatal("LpNormalization: wrong len") }
}

func TestMeanVarianceNormalization(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 2, 1, 1}, []float32{1, 2})
	out, err := opMeanVarianceNormalization(makeNode("MeanVarianceNormalization", nil), []tensor.Tensor{x})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 2 { t.Fatal("MeanVarianceNormalization: wrong len") }
}

func TestMaxRoiPool(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 4, 4}, make([]float32, 16))
	x.Data()[0] = 5
	rois := tensor.NewDense[float32](tensor.Shape{1, 5}, []float32{0, 0, 0, 3, 3})
	out, err := opMaxRoiPool(makeNode("MaxRoiPool", map[string]ir.AttrValue{
		"pooled_shape": ir.AttrInts{Value: []int64{1, 1}},
	}), []tensor.Tensor{x, rois})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 1 { t.Fatal("MaxRoiPool: wrong len") }
}

func TestDFT(t *testing.T) {
	// Simple 4-point DFT: [1, 0, 0, 0] → [1,1,1,1] real, [0,0,0,0] imag
	x := tensor.NewDense[float32](tensor.Shape{1, 4}, []float32{1, 0, 0, 0})
	out, err := opDFT(makeNode("DFT", nil), []tensor.Tensor{x, nil, nil})
	if err != nil { t.Fatal(err) }
	if out[0].Shape()[len(out[0].Shape())-1] != 2 { t.Fatal("DFT: last dim should be 2 (complex)") }
}

func TestSTFT(t *testing.T) {
	signal := tensor.NewDense[float32](tensor.Shape{1, 8, 1}, make([]float32, 8))
	signal.Data()[0] = 1
	frameStep := tensor.NewDense[int64](tensor.Shape{}, []int64{2})
	window := tensor.NewDense[float32](tensor.Shape{4}, []float32{1, 1, 1, 1})
	out, err := opSTFT(makeNode("STFT", nil), []tensor.Tensor{signal, frameStep, window, nil})
	if err != nil { t.Fatal(err) }
	if out[0].Shape().NDim() != 4 { t.Fatal("STFT: expected 4D output") }
}

func TestMelWeightMatrix(t *testing.T) {
	numMelBins := tensor.NewDense[int64](tensor.Shape{}, []int64{4})
	dftLen := tensor.NewDense[int64](tensor.Shape{}, []int64{8})
	sampleRate := tensor.NewDense[int64](tensor.Shape{}, []int64{8000})
	lowerHz := tensor.NewDense[float32](tensor.Shape{}, []float32{0})
	upperHz := tensor.NewDense[float32](tensor.Shape{}, []float32{4000})
	out, err := opMelWeightMatrix(makeNode("MelWeightMatrix", nil), []tensor.Tensor{numMelBins, dftLen, sampleRate, lowerHz, upperHz})
	if err != nil { t.Fatal(err) }
	if out[0].Shape()[1] != 4 { t.Fatalf("MelWeightMatrix: wrong cols %d", out[0].Shape()[1]) }
}

func TestRoiAlignImpl(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 4, 4}, make([]float32, 16))
	for i := range x.Data() { x.Data()[i] = float32(i) }
	rois := tensor.NewDense[float32](tensor.Shape{1, 4}, []float32{0, 0, 3, 3})
	batchIdx := tensor.NewDense[int64](tensor.Shape{1}, []int64{0})
	out, err := opRoiAlign(makeNode("RoiAlign", map[string]ir.AttrValue{
		"output_height": ir.AttrInt{Value: 2}, "output_width": ir.AttrInt{Value: 2},
	}), []tensor.Tensor{x, rois, batchIdx})
	if err != nil { t.Fatal(err) }
	if out[0].Shape()[2] != 2 || out[0].Shape()[3] != 2 { t.Fatal("RoiAlign: wrong shape") }
}

func TestDeformConvImpl(t *testing.T) {
	x := tensor.NewDense[float32](tensor.Shape{1, 1, 3, 3}, make([]float32, 9))
	for i := range x.Data() { x.Data()[i] = 1 }
	w := tensor.NewDense[float32](tensor.Shape{1, 1, 1, 1}, []float32{1})
	offset := tensor.NewDense[float32](tensor.Shape{1, 2, 3, 3}, make([]float32, 18)) // zero offsets
	out, err := opDeformConv(makeNode("DeformConv", nil), []tensor.Tensor{x, w, offset})
	if err != nil { t.Fatal(err) }
	if out[0].Len() != 9 { t.Fatalf("DeformConv: wrong len %d", out[0].Len()) }
}
