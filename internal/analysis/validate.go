package analysis

import (
	"fmt"
	"slices"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
)

type valueInfo struct {
	dtype ir.DataType
	shape ir.Shape
}

// ValidateGraph performs structural and semantic validation on the canonical graph.
func ValidateGraph(g *ir.Graph) error {
	return validateGraph(g, nil)
}

func validateGraph(g *ir.Graph, outer map[string]valueInfo) error {
	if g == nil {
		return fmt.Errorf("analysis: nil graph")
	}

	known := make(map[string]valueInfo, len(g.Inputs)+len(g.Initializers)+len(g.Nodes))
	for name, info := range outer {
		known[name] = info
	}
	produced := make(map[string]struct{}, len(g.Nodes))

	for _, inp := range g.Inputs {
		if inp.Name == "" {
			return fmt.Errorf("analysis: graph input has empty name")
		}
		if _, exists := known[inp.Name]; exists {
			return fmt.Errorf("analysis: duplicate graph input %q", inp.Name)
		}
		known[inp.Name] = valueInfo{dtype: inp.DType, shape: cloneShape(inp.Shape)}
	}
	for name, init := range g.Initializers {
		if name == "" {
			return fmt.Errorf("analysis: initializer has empty name")
		}
		known[name] = valueInfo{dtype: init.DType, shape: cloneShape(init.Shape)}
	}

	for _, node := range g.Nodes {
		if node.OpType == "" {
			return fmt.Errorf("analysis: node %q has empty op type", node.Name)
		}
		if err := validateInputCount(node); err != nil {
			return err
		}
		inputs := make([]valueInfo, len(node.Inputs))
		for i, input := range node.Inputs {
			if input == "" {
				continue
			}
			info, ok := known[input]
			if !ok {
				return fmt.Errorf("analysis: input %q for node %s (%s) is undefined", input, node.Name, node.OpType)
			}
			inputs[i] = info
		}
		if err := validateSemantics(g, node, inputs); err != nil {
			return err
		}
		outputs, err := inferOutputs(g, node, inputs)
		if err != nil {
			return err
		}
		for i, output := range node.Outputs {
			if output == "" {
				continue
			}
			if _, exists := produced[output]; exists {
				return fmt.Errorf("analysis: value %q produced multiple times", output)
			}
			produced[output] = struct{}{}
			if i < len(outputs) {
				known[output] = outputs[i]
			} else {
				known[output] = valueInfo{}
			}
		}

		for attrName, attr := range node.Attrs {
			ag, ok := attr.(*ir.AttrGraph)
			if !ok || ag == nil || ag.Value == nil {
				continue
			}
			if err := validateGraph(ag.Value, known); err != nil {
				return fmt.Errorf("analysis: subgraph %q for node %s (%s): %w", attrName, node.Name, node.OpType, err)
			}
		}
	}

	for _, out := range g.Outputs {
		if out.Name == "" {
			return fmt.Errorf("analysis: graph output has empty name")
		}
		info, ok := known[out.Name]
		if !ok {
			return fmt.Errorf("analysis: graph output %q is undefined", out.Name)
		}
		if out.DType != ir.DataTypeUndefined && info.dtype != ir.DataTypeUndefined && out.DType != info.dtype {
			return fmt.Errorf("analysis: graph output %q dtype mismatch: declared=%s inferred=%s", out.Name, out.DType, info.dtype)
		}
		if len(out.Shape) > 0 && len(info.shape) > 0 && len(out.Shape) == len(info.shape) && isStaticShape(out.Shape) && isStaticShape(info.shape) && !compatibleShape(out.Shape, info.shape) {
			return fmt.Errorf("analysis: graph output %q shape mismatch: declared=%v inferred=%v", out.Name, out.Shape, info.shape)
		}
	}

	return nil
}

func validateInputCount(node *ir.Node) error {
	count := len(node.Inputs)
	nonEmpty := 0
	for _, input := range node.Inputs {
		if input != "" {
			nonEmpty++
		}
	}
	switch node.OpType {
	case "Add", "Sub", "Mul", "Div", "MatMul", "Gather", "Reshape", "Gemm":
		if count < 2 {
			return fmt.Errorf("analysis: node %s (%s) expects at least 2 inputs", node.Name, node.OpType)
		}
	case "Concat":
		if nonEmpty < 1 {
			return fmt.Errorf("analysis: node %s (%s) expects at least 1 non-empty input", node.Name, node.OpType)
		}
	case "Relu", "Sigmoid", "Tanh", "Softmax", "Identity", "Dropout", "Transpose", "Flatten", "Shape", "Size", "Not", "Cast", "Squeeze", "Unsqueeze", "Conv", "MaxPool", "AveragePool", "GlobalAveragePool", "ConstantOfShape":
		if count < 1 {
			return fmt.Errorf("analysis: node %s (%s) expects at least 1 input", node.Name, node.OpType)
		}
	case "BatchNormalization":
		if count < 5 {
			return fmt.Errorf("analysis: node %s (%s) expects 5 inputs", node.Name, node.OpType)
		}
	case "If":
		if count < 1 {
			return fmt.Errorf("analysis: node %s (%s) expects a condition input", node.Name, node.OpType)
		}
	}
	return nil
}

func validateSemantics(g *ir.Graph, node *ir.Node, inputs []valueInfo) error {
	switch node.OpType {
	case "Add", "Sub", "Mul", "Div":
		if err := requireSameDType(node, inputs[0], inputs[1]); err != nil {
			return err
		}
		if _, ok := broadcastShapes(inputs[0].shape, inputs[1].shape); !ok {
			return fmt.Errorf("analysis: node %s (%s) has incompatible broadcast shapes %v and %v", node.Name, node.OpType, inputs[0].shape, inputs[1].shape)
		}
	case "MatMul":
	case "Gemm":
	case "Concat":
	case "Conv":
	case "BatchNormalization":
		if err := requireSameDType(node, inputs[0], inputs[1]); err != nil {
			return err
		}
	case "If":
		if inputs[0].dtype != ir.DataTypeUndefined && inputs[0].dtype != ir.DataTypeBool && inputs[0].dtype != ir.DataTypeUint8 {
			return fmt.Errorf("analysis: node %s (%s) expects bool condition, got %s", node.Name, node.OpType, inputs[0].dtype)
		}
	}
	return nil
}

func inferOutputs(g *ir.Graph, node *ir.Node, inputs []valueInfo) ([]valueInfo, error) {
	switch node.OpType {
	case "Add", "Sub", "Mul", "Div":
		shape, _ := broadcastShapes(inputs[0].shape, inputs[1].shape)
		return []valueInfo{{dtype: preferDType(inputs[0], inputs[1]), shape: shape}}, nil
	case "Max", "Min", "Greater", "Less", "And":
		shape, _ := broadcastShapes(inputs[0].shape, inputs[1].shape)
		dtype := preferDType(inputs[0], inputs[1])
		if node.OpType == "Greater" || node.OpType == "Less" || node.OpType == "And" {
			dtype = ir.DataTypeBool
		}
		return []valueInfo{{dtype: dtype, shape: shape}}, nil
	case "Relu", "Sigmoid", "Tanh", "Softmax", "Identity", "Dropout", "FastGELU", "BatchNormalization", "Not":
		return []valueInfo{{dtype: inputs[0].dtype, shape: cloneShape(inputs[0].shape)}}, nil
	case "MatMul":
		dtype, shape, err := inferMatMul(inputs[0], inputs[1])
		if err != nil {
			return []valueInfo{{dtype: firstKnownDType(inputs)}}, nil
		}
		return []valueInfo{{dtype: dtype, shape: shape}}, nil
	case "Gemm":
		dtype, shape, err := inferGemm(node, inputs)
		if err != nil {
			return []valueInfo{{dtype: firstKnownDType(inputs)}}, nil
		}
		return []valueInfo{{dtype: dtype, shape: shape}}, nil
	case "Reshape":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferReshape(g, node, inputs)}}, nil
	case "Transpose":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferTranspose(node, inputs[0].shape)}}, nil
	case "Flatten":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferFlatten(node, inputs[0].shape)}}, nil
	case "Squeeze":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferSqueeze(g, node, inputs[0].shape)}}, nil
	case "Unsqueeze":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferUnsqueeze(g, node, inputs[0].shape)}}, nil
	case "Concat":
		shape, err := inferConcat(node, inputs)
		if err != nil {
			return []valueInfo{{dtype: firstKnownDType(inputs)}}, nil
		}
		return []valueInfo{{dtype: firstKnownDType(inputs), shape: shape}}, nil
	case "Gather":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferGather(node, inputs)}}, nil
	case "Slice":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferSlice(g, node, inputs)}}, nil
	case "Shape":
		if len(inputs[0].shape) == 0 {
			return []valueInfo{{dtype: ir.DataTypeInt64}}, nil
		}
		return []valueInfo{{dtype: ir.DataTypeInt64, shape: ir.Shape{int64(len(inputs[0].shape))}}}, nil
	case "Size":
		return []valueInfo{{dtype: ir.DataTypeInt64, shape: ir.Shape{}}}, nil
	case "Cast":
		return []valueInfo{{dtype: ir.DataType(node.GetAttrInt("to", int64(inputs[0].dtype))), shape: cloneShape(inputs[0].shape)}}, nil
	case "Conv":
		dtype, shape, err := inferConv(node, inputs)
		if err != nil {
			return []valueInfo{{dtype: firstKnownDType(inputs)}}, nil
		}
		return []valueInfo{{dtype: dtype, shape: shape}}, nil
	case "MaxPool", "AveragePool", "GlobalAveragePool":
		return []valueInfo{{dtype: inputs[0].dtype, shape: inferPool(node, inputs[0].shape)}}, nil
	case "Constant":
		if t := node.GetAttrTensor("value"); t != nil {
			return []valueInfo{{dtype: t.DType, shape: cloneShape(t.Shape)}}, nil
		}
		if _, ok := node.Attrs["value_float"]; ok {
			return []valueInfo{{dtype: ir.DataTypeFloat, shape: ir.Shape{}}}, nil
		}
		if _, ok := node.Attrs["value_int"]; ok {
			return []valueInfo{{dtype: ir.DataTypeInt64, shape: ir.Shape{}}}, nil
		}
	case "ConstantOfShape":
		return []valueInfo{{dtype: inferConstantOfShapeDType(node), shape: nil}}, nil
	case "If":
		return inferIfOutputs(node), nil
	}
	return make([]valueInfo, len(node.Outputs)), nil
}

func inferIfOutputs(node *ir.Node) []valueInfo {
	for _, attrName := range []string{"then_branch", "else_branch"} {
		ag, ok := node.Attrs[attrName].(*ir.AttrGraph)
		if ok && ag != nil && ag.Value != nil {
			out := make([]valueInfo, len(ag.Value.Outputs))
			for i, spec := range ag.Value.Outputs {
				out[i] = valueInfo{dtype: spec.DType, shape: cloneShape(spec.Shape)}
			}
			return out
		}
	}
	return make([]valueInfo, len(node.Outputs))
}

func inferConstantOfShapeDType(node *ir.Node) ir.DataType {
	if t := node.GetAttrTensor("value"); t != nil {
		return t.DType
	}
	return ir.DataTypeFloat
}

func inferPool(node *ir.Node, in ir.Shape) ir.Shape {
	if len(in) < 3 {
		return nil
	}
	out := cloneShape(in)
	if node.OpType == "GlobalAveragePool" {
		for i := 2; i < len(out); i++ {
			out[i] = 1
		}
		return out
	}
	kernel := node.GetAttrInts("kernel_shape", nil)
	strides := expandInts(node.GetAttrInts("strides", nil), len(kernel), 1)
	pads := expandInts(node.GetAttrInts("pads", nil), len(kernel)*2, 0)
	for i := 0; i < len(kernel) && 2+i < len(out); i++ {
		if out[2+i] <= 0 {
			out[2+i] = -1
			continue
		}
		out[2+i] = (out[2+i]+pads[i]+pads[i+len(kernel)]-kernel[i])/strides[i] + 1
	}
	return out
}

func inferConv(node *ir.Node, inputs []valueInfo) (ir.DataType, ir.Shape, error) {
	dtype := preferDType(inputs[0], inputs[1])
	if len(inputs[0].shape) == 0 || len(inputs[1].shape) == 0 {
		return dtype, nil, nil
	}
	xShape := inputs[0].shape
	wShape := inputs[1].shape
	if len(xShape) != len(wShape) {
		return dtype, nil, fmt.Errorf("conv rank mismatch: x=%v w=%v", xShape, wShape)
	}
	if len(xShape) < 3 {
		return dtype, nil, fmt.Errorf("conv expects rank >= 3")
	}
	if xShape[1] > 0 && wShape[1] > 0 {
		group := node.GetAttrInt("group", 1)
		if xShape[1] != wShape[1]*group {
			return dtype, nil, fmt.Errorf("conv channel mismatch: x.c=%d weight.c=%d group=%d", xShape[1], wShape[1], group)
		}
	}
	out := cloneShape(xShape)
	out[1] = wShape[0]
	kernel := wShape[2:]
	strides := expandInts(node.GetAttrInts("strides", nil), len(kernel), 1)
	dilations := expandInts(node.GetAttrInts("dilations", nil), len(kernel), 1)
	pads := expandInts(node.GetAttrInts("pads", nil), len(kernel)*2, 0)
	for i := 0; i < len(kernel); i++ {
		if xShape[2+i] <= 0 || kernel[i] <= 0 {
			out[2+i] = -1
			continue
		}
		effective := dilations[i]*(kernel[i]-1) + 1
		out[2+i] = (xShape[2+i]+pads[i]+pads[i+len(kernel)]-effective)/strides[i] + 1
	}
	return dtype, out, nil
}

func inferGather(node *ir.Node, inputs []valueInfo) ir.Shape {
	if len(inputs[0].shape) == 0 || len(inputs[1].shape) == 0 {
		return nil
	}
	axis := normalizeAxis(int(node.GetAttrInt("axis", 0)), len(inputs[0].shape))
	out := make(ir.Shape, 0, len(inputs[0].shape)-1+len(inputs[1].shape))
	out = append(out, inputs[0].shape[:axis]...)
	out = append(out, inputs[1].shape...)
	out = append(out, inputs[0].shape[axis+1:]...)
	return out
}

func inferSlice(g *ir.Graph, node *ir.Node, inputs []valueInfo) ir.Shape {
	if len(inputs) == 0 || len(inputs[0].shape) == 0 {
		return nil
	}
	starts := node.GetAttrInts("starts", nil)
	ends := node.GetAttrInts("ends", nil)
	axes := node.GetAttrInts("axes", nil)
	steps := []int64(nil)
	if len(node.Inputs) > 1 {
		if v, ok := constInt64s(g, node.Inputs[1]); ok {
			starts = v
		}
	}
	if len(node.Inputs) > 2 {
		if v, ok := constInt64s(g, node.Inputs[2]); ok {
			ends = v
		}
	}
	if len(node.Inputs) > 3 {
		if v, ok := constInt64s(g, node.Inputs[3]); ok {
			axes = v
		}
	}
	if len(node.Inputs) > 4 {
		if v, ok := constInt64s(g, node.Inputs[4]); ok {
			steps = v
		}
	}
	if len(starts) == 0 || len(ends) == 0 {
		return nil
	}

	out := cloneShape(inputs[0].shape)
	if len(axes) == 0 {
		axes = make([]int64, len(starts))
		for i := range starts {
			axes[i] = int64(i)
		}
	}
	if len(steps) == 0 {
		steps = make([]int64, len(starts))
		for i := range starts {
			steps[i] = 1
		}
	}

	for i, axis := range axes {
		d := normalizeAxis(int(axis), len(out))
		dim := out[d]
		if dim <= 0 {
			out[d] = -1
			continue
		}
		st := steps[i]
		if st == 0 {
			st = 1
		}
		rawStart := starts[i]
		rawEnd := ends[i]
		s := rawStart
		e := rawEnd
		if st > 0 {
			if s < 0 {
				s += dim
			}
			if e < 0 {
				e += dim
			}
			if rawEnd > 2000000000 {
				e = dim
			}
			if s < 0 {
				s = 0
			}
			if s > dim {
				s = dim
			}
			if e < 0 {
				e = 0
			}
			if e > dim {
				e = dim
			}
			if e <= s {
				out[d] = 0
				continue
			}
			out[d] = (e - s + st - 1) / st
			continue
		}

		if s < 0 {
			s += dim
		}
		if e < 0 {
			e += dim
		}
		if rawEnd < -2000000000 {
			e = -1
		}
		if s < 0 {
			s = dim - 1
		}
		if s >= dim {
			s = dim - 1
		}
		if e < -1 {
			e = -1
		}
		if e >= dim {
			e = dim - 1
		}
		if s <= e {
			out[d] = 0
			continue
		}
		step := -st
		out[d] = (s - e + step - 1) / step
	}

	return out
}

func inferConcat(node *ir.Node, inputs []valueInfo) (ir.Shape, error) {
	filtered := make([]valueInfo, 0, len(inputs))
	for _, in := range inputs {
		if len(in.shape) > 0 || in.dtype != ir.DataTypeUndefined {
			filtered = append(filtered, in)
		}
	}
	base := firstKnownShape(filtered)
	if len(base) == 0 {
		return nil, nil
	}
	axis := normalizeAxis(int(node.GetAttrInt("axis", 0)), len(base))
	out := cloneShape(base)
	out[axis] = 0
	for _, in := range filtered {
		if len(in.shape) == 0 {
			return nil, nil
		}
		if len(in.shape) != len(base) {
			return nil, fmt.Errorf("concat rank mismatch: %v vs %v", base, in.shape)
		}
		for i := range base {
			if i == axis {
				continue
			}
			if !dimCompatible(base[i], in.shape[i]) {
				return nil, fmt.Errorf("concat dim mismatch at axis %d: %v vs %v", i, base, in.shape)
			}
		}
		if out[axis] >= 0 && in.shape[axis] >= 0 {
			out[axis] += in.shape[axis]
		} else {
			out[axis] = -1
		}
	}
	return out, nil
}

func inferFlatten(node *ir.Node, in ir.Shape) ir.Shape {
	if len(in) == 0 {
		return nil
	}
	axis := normalizeAxis(int(node.GetAttrInt("axis", 1)), len(in))
	left := product(in[:axis])
	right := product(in[axis:])
	return ir.Shape{left, right}
}

func inferTranspose(node *ir.Node, in ir.Shape) ir.Shape {
	if len(in) == 0 {
		return nil
	}
	perm := node.GetAttrInts("perm", nil)
	if len(perm) == 0 {
		out := cloneShape(in)
		slices.Reverse(out)
		return out
	}
	out := make(ir.Shape, len(perm))
	for i, p := range perm {
		if int(p) < len(in) {
			out[i] = in[p]
		}
	}
	return out
}

func inferReshape(g *ir.Graph, node *ir.Node, inputs []valueInfo) ir.Shape {
	if len(node.Inputs) < 2 {
		return nil
	}
	dims, ok := constInt64s(g, node.Inputs[1])
	if !ok {
		return nil
	}
	out := make(ir.Shape, len(dims))
	copy(out, dims)
	if len(inputs[0].shape) == 0 {
		return out
	}
	total := product(inputs[0].shape)
	known := int64(1)
	negOne := -1
	for i, d := range out {
		switch d {
		case 0:
			if i < len(inputs[0].shape) {
				out[i] = inputs[0].shape[i]
				d = out[i]
			}
		case -1:
			negOne = i
			continue
		}
		if d > 0 {
			known *= d
		}
	}
	if negOne >= 0 && total > 0 && known > 0 {
		out[negOne] = total / known
	}
	return out
}

func inferSqueeze(g *ir.Graph, node *ir.Node, in ir.Shape) ir.Shape {
	if len(in) == 0 {
		return nil
	}
	axes := node.GetAttrInts("axes", nil)
	if len(axes) == 0 && len(node.Inputs) > 1 {
		axes, _ = constInt64s(g, node.Inputs[1])
	}
	if len(axes) == 0 {
		out := make(ir.Shape, 0, len(in))
		for _, d := range in {
			if d != 1 {
				out = append(out, d)
			}
		}
		return out
	}
	remove := make(map[int]bool, len(axes))
	for _, axis := range axes {
		remove[normalizeAxis(int(axis), len(in))] = true
	}
	out := make(ir.Shape, 0, len(in))
	for i, d := range in {
		if !remove[i] {
			out = append(out, d)
		}
	}
	return out
}

func inferUnsqueeze(g *ir.Graph, node *ir.Node, in ir.Shape) ir.Shape {
	axes := node.GetAttrInts("axes", nil)
	if len(axes) == 0 && len(node.Inputs) > 1 {
		axes, _ = constInt64s(g, node.Inputs[1])
	}
	if len(axes) == 0 {
		return nil
	}
	out := cloneShape(in)
	norm := make([]int, len(axes))
	for i, axis := range axes {
		norm[i] = int(axis)
	}
	slices.Sort(norm)
	for _, axis := range norm {
		if axis < 0 {
			axis += len(out) + 1
		}
		if axis < 0 || axis > len(out) {
			return nil
		}
		out = append(out[:axis], append(ir.Shape{1}, out[axis:]...)...)
	}
	return out
}

func inferMatMul(a, b valueInfo) (ir.DataType, ir.Shape, error) {
	dtype := preferDType(a, b)
	if len(a.shape) == 0 || len(b.shape) == 0 {
		return dtype, nil, nil
	}
	if len(a.shape) == 1 && len(b.shape) == 1 {
		if !dimCompatible(a.shape[0], b.shape[0]) {
			return dtype, nil, fmt.Errorf("matmul vector mismatch: %v vs %v", a.shape, b.shape)
		}
		return dtype, ir.Shape{}, nil
	}
	if len(a.shape) == 1 {
		a.shape = append(ir.Shape{1}, a.shape...)
	}
	if len(b.shape) == 1 {
		b.shape = append(b.shape, 1)
	}
	if !dimCompatible(a.shape[len(a.shape)-1], b.shape[len(b.shape)-2]) {
		return dtype, nil, fmt.Errorf("matmul inner dim mismatch: %v vs %v", a.shape, b.shape)
	}
	batch, ok := broadcastShapes(a.shape[:len(a.shape)-2], b.shape[:len(b.shape)-2])
	if !ok {
		return dtype, nil, fmt.Errorf("matmul batch broadcast mismatch: %v vs %v", a.shape, b.shape)
	}
	out := append(cloneShape(batch), a.shape[len(a.shape)-2], b.shape[len(b.shape)-1])
	if len(out) == 2 && len(a.shape) == 2 && len(b.shape) == 1 {
		out = ir.Shape{out[0]}
	}
	return dtype, out, nil
}

func inferGemm(node *ir.Node, inputs []valueInfo) (ir.DataType, ir.Shape, error) {
	a, b := inputs[0], inputs[1]
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return preferDType(a, b), nil, nil
	}
	m, kA := a.shape[0], a.shape[1]
	kB, n := b.shape[0], b.shape[1]
	if node.GetAttrInt("transA", 0) != 0 {
		m, kA = a.shape[1], a.shape[0]
	}
	if node.GetAttrInt("transB", 0) != 0 {
		kB, n = b.shape[1], b.shape[0]
	}
	if !dimCompatible(kA, kB) {
		return preferDType(a, b), nil, fmt.Errorf("gemm inner dim mismatch: %v vs %v", a.shape, b.shape)
	}
	return preferDType(a, b), ir.Shape{m, n}, nil
}

func constInt64s(g *ir.Graph, name string) ([]int64, bool) {
	init := g.Initializers[name]
	if init == nil {
		return nil, false
	}
	data, err := materialize.Int64(init)
	if err != nil {
		return nil, false
	}
	return data, true
}

func firstKnownShape(inputs []valueInfo) ir.Shape {
	for _, in := range inputs {
		if len(in.shape) > 0 {
			return in.shape
		}
	}
	return nil
}

func firstKnownDType(inputs []valueInfo) ir.DataType {
	for _, in := range inputs {
		if in.dtype != ir.DataTypeUndefined {
			return in.dtype
		}
	}
	return ir.DataTypeUndefined
}

func requireSameDType(node *ir.Node, a, b valueInfo) error {
	if a.dtype != ir.DataTypeUndefined && b.dtype != ir.DataTypeUndefined && a.dtype != b.dtype {
		return fmt.Errorf("analysis: node %s (%s) dtype mismatch: %s vs %s", node.Name, node.OpType, a.dtype, b.dtype)
	}
	return nil
}

func preferDType(a, b valueInfo) ir.DataType {
	if a.dtype != ir.DataTypeUndefined {
		return a.dtype
	}
	return b.dtype
}

func cloneShape(shape ir.Shape) ir.Shape {
	if len(shape) == 0 {
		return nil
	}
	out := make(ir.Shape, len(shape))
	copy(out, shape)
	return out
}

func compatibleShape(a, b ir.Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !dimCompatible(a[i], b[i]) {
			return false
		}
	}
	return true
}

func dimCompatible(a, b int64) bool {
	return a <= 0 || b <= 0 || a == b
}

func isStaticShape(shape ir.Shape) bool {
	for _, d := range shape {
		if d <= 0 {
			return false
		}
	}
	return true
}

func hasDynamicDim(s ir.Shape) bool {
	for _, d := range s {
		if d <= 0 {
			return true
		}
	}
	return false
}

func broadcastShapes(a, b ir.Shape) (ir.Shape, bool) {
	if len(a) == 0 {
		return cloneShape(b), true
	}
	if len(b) == 0 {
		return cloneShape(a), true
	}
	// If either shape contains dynamic dims, skip strict broadcast check
	if hasDynamicDim(a) || hasDynamicDim(b) {
		n := max(len(a), len(b))
		out := make(ir.Shape, n)
		for i := 0; i < n; i++ {
			ad, bd := int64(1), int64(1)
			if j := len(a) - 1 - i; j >= 0 { ad = a[j] }
			if j := len(b) - 1 - i; j >= 0 { bd = b[j] }
			if ad > 0 && bd > 0 && ad != 1 && bd != 1 && ad == bd {
				out[n-1-i] = ad
			} else if ad > 0 && ad != 1 {
				out[n-1-i] = ad
			} else if bd > 0 && bd != 1 {
				out[n-1-i] = bd
			} else {
				out[n-1-i] = -1
			}
		}
		return out, true
	}
	n := max(len(a), len(b))
	out := make(ir.Shape, n)
	for i := 0; i < n; i++ {
		ad := int64(1)
		bd := int64(1)
		if j := len(a) - 1 - i; j >= 0 {
			ad = a[j]
		}
		if j := len(b) - 1 - i; j >= 0 {
			bd = b[j]
		}
		switch {
		case ad == bd:
			out[n-1-i] = ad
		case ad == 1:
			out[n-1-i] = bd
		case bd == 1:
			out[n-1-i] = ad
		case ad <= 0 || bd <= 0:
			// Dynamic dimension: unknown size, treat as compatible
			if ad > 0 {
				out[n-1-i] = ad
			} else if bd > 0 {
				out[n-1-i] = bd
			} else {
				out[n-1-i] = -1
			}
		default:
			return nil, false
		}
	}
	return out, true
}

func expandInts(values []int64, n int, def int64) []int64 {
	if len(values) == 0 {
		out := make([]int64, n)
		for i := range out {
			out[i] = def
		}
		return out
	}
	out := make([]int64, n)
	copy(out, values)
	for i := len(values); i < n; i++ {
		out[i] = def
	}
	return out
}

func normalizeAxis(axis, rank int) int {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 {
		return 0
	}
	if axis >= rank {
		return rank - 1
	}
	return axis
}

func product(shape ir.Shape) int64 {
	if len(shape) == 0 {
		return 1
	}
	prod := int64(1)
	for _, d := range shape {
		if d <= 0 {
			return -1
		}
		prod *= d
	}
	return prod
}

// opsetMinVersion defines the minimum opset version required for each operator.
// Operators not listed here have no minimum version constraint.
var opsetMinVersion = map[string]int64{
	"Mish":               18,
	"GroupNormalization":  18,
	"Gelu":               20,
	"GridSample":         16,
	"Trilu":              14,
	"Einsum":             12,
	"DynamicQuantizeLinear": 11,
}

// ValidateOpsetCompatibility checks that each node's opset version is
// compatible with known opset-sensitive behaviors. Returns warnings (not errors)
// for potential issues.
func ValidateOpsetCompatibility(g *ir.Graph) []string {
	var warnings []string
	for _, node := range g.Nodes {
		v := node.OpsetVersion
		if v == 0 {
			continue // no opset info available
		}

		// Check minimum version requirements
		if minVer, ok := opsetMinVersion[node.OpType]; ok {
			if v < minVer {
				warnings = append(warnings, fmt.Sprintf(
					"node %q (%s): opset %d < minimum %d for this operator",
					node.Name, node.OpType, v, minVer))
			}
		}

		// Check opset-sensitive semantics
		switch node.OpType {
		case "Softmax", "LogSoftmax":
			// opset < 13: default axis=1 (flatten), >= 13: default axis=-1
			if v < 13 {
				if _, ok := node.Attrs["axis"]; !ok {
					warnings = append(warnings, fmt.Sprintf(
						"node %q (%s): opset %d < 13, default axis=1 (flatten mode)",
						node.Name, node.OpType, v))
				}
			}
		case "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceSum",
			"ReduceSumSquare", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp":
			// opset < 18: axes from attribute; >= 18: axes from input
			if v >= 18 {
				if _, ok := node.Attrs["axes"]; ok {
					warnings = append(warnings, fmt.Sprintf(
						"node %q (%s): opset %d >= 18 but axes specified as attribute (should be input)",
						node.Name, node.OpType, v))
				}
			}
		case "Resize":
			// opset 10: no coordinate_transformation_mode
			if v <= 10 {
				if _, ok := node.Attrs["coordinate_transformation_mode"]; ok {
					warnings = append(warnings, fmt.Sprintf(
						"node %q (Resize): opset %d <= 10 but coordinate_transformation_mode specified",
						node.Name, v))
				}
			}
		case "Squeeze", "Unsqueeze":
			// opset < 13: axes from attribute; >= 13: axes from input
			if v >= 13 {
				if _, ok := node.Attrs["axes"]; ok {
					warnings = append(warnings, fmt.Sprintf(
						"node %q (%s): opset %d >= 13 but axes specified as attribute (should be input)",
						node.Name, node.OpType, v))
				}
			}
		}
	}
	return warnings
}
