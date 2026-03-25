package optimize

import (
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
)

// materializeConstants converts Constant op nodes into graph initializers.
// This enables downstream passes (BN fusion, GELU fusion, etc.) to recognize
// constant values that were encoded as Constant ops rather than initializers.
func materializeConstants(g *ir.Graph) {
	var toRemove []*ir.Node
	for _, n := range g.Nodes {
		if n.OpType != "Constant" || len(n.Outputs) == 0 {
			continue
		}
		outName := n.Outputs[0]

		// value (tensor attribute)
		if t := n.GetAttrTensor("value"); t != nil {
			t.Name = outName
			g.Initializers[outName] = t
			toRemove = append(toRemove, n)
			continue
		}
		// value_float
		if v, ok := n.Attrs["value_float"]; ok {
			if af, ok := v.(ir.AttrFloat); ok {
				g.Initializers[outName] = &ir.Initializer{
					Name: outName, DType: ir.DataTypeFloat, Shape: ir.Shape{},
					RawData: f32Bytes([]float32{af.Value}),
				}
				toRemove = append(toRemove, n)
				continue
			}
		}
		// value_int
		if v, ok := n.Attrs["value_int"]; ok {
			if ai, ok := v.(ir.AttrInt); ok {
				g.Initializers[outName] = &ir.Initializer{
					Name: outName, DType: ir.DataTypeInt64, Shape: ir.Shape{},
					RawData: i64Bytes([]int64{ai.Value}),
				}
				toRemove = append(toRemove, n)
				continue
			}
		}
		// value_floats
		if v, ok := n.Attrs["value_floats"]; ok {
			if af, ok := v.(ir.AttrFloats); ok {
				data := make([]float32, len(af.Value))
				copy(data, af.Value)
				g.Initializers[outName] = &ir.Initializer{
					Name: outName, DType: ir.DataTypeFloat, Shape: ir.Shape{int64(len(data))},
					RawData: f32Bytes(data),
				}
				toRemove = append(toRemove, n)
				continue
			}
		}
		// value_ints
		if v, ok := n.Attrs["value_ints"]; ok {
			if ai, ok := v.(ir.AttrInts); ok {
				data := make([]int64, len(ai.Value))
				copy(data, ai.Value)
				g.Initializers[outName] = &ir.Initializer{
					Name: outName, DType: ir.DataTypeInt64, Shape: ir.Shape{int64(len(data))},
					RawData: i64Bytes(data),
				}
				toRemove = append(toRemove, n)
				continue
			}
		}
	}
	removeNodes(g, toRemove)
}

func f32Bytes(data []float32) []byte {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		buf[i*4] = byte(bits); buf[i*4+1] = byte(bits >> 8); buf[i*4+2] = byte(bits >> 16); buf[i*4+3] = byte(bits >> 24)
	}
	return buf
}

func i64Bytes(data []int64) []byte {
	buf := make([]byte, len(data)*8)
	for i, v := range data {
		u := uint64(v)
		for j := 0; j < 8; j++ { buf[i*8+j] = byte(u >> (j * 8)) }
	}
	return buf
}

// Optimize applies inference-time graph optimizations in the recommended order.

// eliminateDropout removes Dropout nodes (inference-time passthrough).
func eliminateDropout(g *ir.Graph) {
	replacePassthrough(g, "Dropout")
}

// eliminateIdentity removes Identity nodes.
func eliminateIdentity(g *ir.Graph) {
	replacePassthrough(g, "Identity")
}

// replacePassthrough removes nodes that just pass input[0] to output[0].
func replacePassthrough(g *ir.Graph, opType string) {
	for {
		changed := false
		for i, n := range g.Nodes {
			if n.OpType != opType || len(n.Inputs) == 0 || len(n.Outputs) == 0 {
				continue
			}
			inName := n.Inputs[0]
			outName := n.Outputs[0]
			// Rewire all consumers
			for _, other := range g.Nodes {
				for j, inp := range other.Inputs {
					if inp == outName {
						other.Inputs[j] = inName
					}
				}
			}
			// Rewire graph outputs
			for j, out := range g.Outputs {
				if out.Name == outName {
					g.Outputs[j].Name = inName
				}
			}
			// Remove ir.Node
			g.Nodes = append(g.Nodes[:i], g.Nodes[i+1:]...)
			changed = true
			break
		}
		if !changed {
			break
		}
	}
}

// fuseConvBatchNorm folds BatchNormalization into preceding Conv weights.
// Formula: Wf[oc] = W[oc] * (scale[oc] / sqrt(var[oc] + eps))
//
//	bf[oc] = (b[oc] - mean[oc]) * (scale[oc] / sqrt(var[oc] + eps)) + beta[oc]
func fuseConvBatchNorm(g *ir.Graph) {
	// Build output ↁEir.Node map
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			producer[out] = n
		}
	}

	// Build output ↁEconsumer count
	useCount := make(map[string]int)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	var toRemove []*ir.Node
	for _, bn := range g.Nodes {
		if bn.OpType != "BatchNormalization" || len(bn.Inputs) < 5 {
			continue
		}

		convOut := bn.Inputs[0]
		conv, ok := producer[convOut]
		if !ok || conv.OpType != "Conv" {
			continue
		}

		// Conv output must only feed this BN
		if useCount[convOut] != 1 {
			continue
		}

		// Get BN parameters from initializers
		scaleInit := g.Initializers[bn.Inputs[1]]
		betaInit := g.Initializers[bn.Inputs[2]]
		meanInit := g.Initializers[bn.Inputs[3]]
		varInit := g.Initializers[bn.Inputs[4]]
		if scaleInit == nil || betaInit == nil || meanInit == nil || varInit == nil {
			continue
		}

		// Get Conv weight
		if len(conv.Inputs) < 2 {
			continue
		}
		wInit := g.Initializers[conv.Inputs[1]]
		if wInit == nil || wInit.DType != ir.DataTypeFloat {
			continue
		}

		// Materialize all as float32
		wData, err := materialize.Float32(wInit)
		if err != nil {
			continue
		}
		scaleData, _ := materialize.Float32(scaleInit)
		betaData, _ := materialize.Float32(betaInit)
		meanData, _ := materialize.Float32(meanInit)
		varData, _ := materialize.Float32(varInit)
		if scaleData == nil || betaData == nil || meanData == nil || varData == nil {
			continue
		}

		eps := float64(bn.GetAttrFloat("epsilon", 1e-5))
		oc := int(wInit.Shape[0])
		channelSize := len(wData) / oc

		// Get or create Conv bias
		var biasData []float32
		hasBias := len(conv.Inputs) >= 3 && conv.Inputs[2] != ""
		if hasBias {
			biasInit := g.Initializers[conv.Inputs[2]]
			if biasInit != nil {
				biasData, _ = materialize.Float32(biasInit)
			}
		}
		if biasData == nil {
			biasData = make([]float32, oc)
		}

		// Fuse: Wf[oc] = W[oc] * alpha, bf[oc] = (b[oc] - mean) * alpha + beta
		newW := make([]float32, len(wData))
		newBias := make([]float32, oc)
		for c := 0; c < oc; c++ {
			alpha := float64(scaleData[c]) / math.Sqrt(float64(varData[c])+eps)
			alphaF := float32(alpha)
			for j := 0; j < channelSize; j++ {
				newW[c*channelSize+j] = wData[c*channelSize+j] * alphaF
			}
			newBias[c] = float32((float64(biasData[c])-float64(meanData[c]))*alpha + float64(betaData[c]))
		}

		// Update weight ir.Initializer
		wInit.FloatData = newW
		wInit.RawData = nil

		// Create or update bias ir.Initializer
		biasName := conv.Inputs[1] + "_fused_bias"
		if hasBias && conv.Inputs[2] != "" {
			biasName = conv.Inputs[2]
		}
		g.Initializers[biasName] = &ir.Initializer{
			Name:      biasName,
			DType:     ir.DataTypeFloat,
			Shape:     ir.Shape{int64(oc)},
			FloatData: newBias,
		}

		// Ensure Conv has bias input
		if !hasBias {
			conv.Inputs = append(conv.Inputs, biasName)
		} else {
			conv.Inputs[2] = biasName
		}

		// Rewire: BN consumers now use Conv output
		bnOut := bn.Outputs[0]
		for _, other := range g.Nodes {
			for j, inp := range other.Inputs {
				if inp == bnOut {
					other.Inputs[j] = convOut
				}
			}
		}
		for j, out := range g.Outputs {
			if out.Name == bnOut {
				g.Outputs[j].Name = convOut
			}
		}

		toRemove = append(toRemove, bn)
	}

	removeNodes(g, toRemove)
}

// fuseConvAddBias folds Conv ↁEAdd(constant per-channel) into Conv bias.
func fuseConvAddBias(g *ir.Graph) {
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			producer[out] = n
		}
	}
	useCount := make(map[string]int)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	var toRemove []*ir.Node
	for _, add := range g.Nodes {
		if add.OpType != "Add" || len(add.Inputs) < 2 {
			continue
		}

		// Find which input comes from Conv and which is a constant bias
		var conv *ir.Node
		var convOut string
		var biasInit *ir.Initializer
		for idx := 0; idx < 2; idx++ {
			name := add.Inputs[idx]
			otherName := add.Inputs[1-idx]
			if p, ok := producer[name]; ok && (p.OpType == "Conv" || p.OpType == "FusedConv") {
				if useCount[name] == 1 {
					if init, ok := g.Initializers[otherName]; ok && init.DType == ir.DataTypeFloat {
						conv = p
						convOut = name
						biasInit = init
						break
					}
				}
			}
		}
		if conv == nil || biasInit == nil {
			continue
		}

		// Materialize add bias
		addBias, err := materialize.Float32(biasInit)
		if err != nil || len(addBias) == 0 {
			continue
		}

		// Get or create Conv bias
		hasBias := len(conv.Inputs) >= 3 && conv.Inputs[2] != ""
		if hasBias {
			existingInit := g.Initializers[conv.Inputs[2]]
			if existingInit != nil {
				existingBias, err := materialize.Float32(existingInit)
				if err == nil && len(existingBias) == len(addBias) {
					// Add to existing bias
					for i := range existingBias {
						existingBias[i] += addBias[i]
					}
					existingInit.FloatData = existingBias
					existingInit.RawData = nil
				}
			}
		} else {
			// Create new bias ir.Initializer
			biasName := convOut + "_fused_add_bias"
			g.Initializers[biasName] = &ir.Initializer{
				Name:      biasName,
				DType:     ir.DataTypeFloat,
				Shape:     biasInit.Shape,
				FloatData: addBias,
			}
			if len(conv.Inputs) < 3 {
				conv.Inputs = append(conv.Inputs, biasName)
			} else {
				conv.Inputs[2] = biasName
			}
		}

		// Rewire consumers
		addOut := add.Outputs[0]
		for _, other := range g.Nodes {
			for j, inp := range other.Inputs {
				if inp == addOut {
					other.Inputs[j] = convOut
				}
			}
		}
		for j, out := range g.Outputs {
			if out.Name == addOut {
				g.Outputs[j].Name = convOut
			}
		}
		toRemove = append(toRemove, add)
	}
	removeNodes(g, toRemove)
}

// fuseConvActivation merges Conv ↁERelu or Conv ↁEClip into a FusedConv.
func fuseConvActivation(g *ir.Graph) {
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			producer[out] = n
		}
	}
	useCount := make(map[string]int)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	var toRemove []*ir.Node
	for _, act := range g.Nodes {
		var activation string
		switch act.OpType {
		case "Relu":
			activation = "relu"
		case "Clip":
			activation = "clip"
		case "LeakyRelu":
			activation = "leakyrelu"
		default:
			continue
		}

		if len(act.Inputs) == 0 {
			continue
		}
		convOut := act.Inputs[0]
		conv, ok := producer[convOut]
		if !ok || (conv.OpType != "Conv" && conv.OpType != "FusedConv") {
			continue
		}
		// Conv output must only feed this activation
		if useCount[convOut] != 1 {
			continue
		}
		// Don't fuse if Conv already has an activation
		if _, exists := conv.Attrs["activation"]; exists {
			continue
		}

		conv.OpType = "FusedConv"
		conv.Attrs["activation"] = ir.AttrString{Value: activation}

		// For LeakyRelu, store alpha
		if activation == "leakyrelu" {
			conv.Attrs["leakyrelu_alpha"] = ir.AttrFloat{Value: act.GetAttrFloat("alpha", 0.01)}
		}
		// For Clip, store min/max from inputs or defaults
		if activation == "clip" {
			if len(act.Inputs) > 1 && act.Inputs[1] != "" {
				if init := g.Initializers[act.Inputs[1]]; init != nil {
					if data, err := materialize.Float32(init); err == nil && len(data) > 0 {
						conv.Attrs["clip_min"] = ir.AttrFloat{Value: data[0]}
					}
				}
			}
			if len(act.Inputs) > 2 && act.Inputs[2] != "" {
				if init := g.Initializers[act.Inputs[2]]; init != nil {
					if data, err := materialize.Float32(init); err == nil && len(data) > 0 {
						conv.Attrs["clip_max"] = ir.AttrFloat{Value: data[0]}
					}
				}
			}
		}

		// Rewire: Conv output takes activation's output name
		actOut := act.Outputs[0]
		conv.Outputs[0] = actOut

		toRemove = append(toRemove, act)
	}

	removeNodes(g, toRemove)
}

// fuseConvSiLU fuses Conv → Sigmoid → Mul(x, sig) into FusedConv with activation="silu".
// Pattern: Conv produces X, Sigmoid(X) produces S, Mul(X, S) produces Y.
// Conv output X must feed both Sigmoid and Mul (useCount == 2).
func fuseConvSiLU(g *ir.Graph) {
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			producer[out] = n
		}
	}
	useCount := make(map[string]int)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	var toRemove []*ir.Node
	for _, mul := range g.Nodes {
		if mul.OpType != "Mul" || len(mul.Inputs) < 2 {
			continue
		}
		// Find pattern: Mul(convOut, sigmoidOut) where sigmoid input == convOut
		for idx := 0; idx < 2; idx++ {
			sigOutName := mul.Inputs[idx]
			convOutName := mul.Inputs[1-idx]

			sig, ok := producer[sigOutName]
			if !ok || sig.OpType != "Sigmoid" || len(sig.Inputs) == 0 {
				continue
			}
			// Sigmoid's input must be the same as Mul's other input
			if sig.Inputs[0] != convOutName {
				continue
			}
			// Sigmoid output must only feed this Mul
			if useCount[sigOutName] != 1 {
				continue
			}
			// Conv/FusedConv output feeds Sigmoid and Mul (useCount == 2)
			conv, ok := producer[convOutName]
			if !ok || (conv.OpType != "Conv" && conv.OpType != "FusedConv") {
				continue
			}
			if useCount[convOutName] != 2 {
				continue
			}
			// Don't fuse if Conv already has an activation
			if _, exists := conv.Attrs["activation"]; exists {
				continue
			}

			// Fuse: Conv → SiLU
			conv.OpType = "FusedConv"
			conv.Attrs["activation"] = ir.AttrString{Value: "silu"}
			conv.Outputs[0] = mul.Outputs[0]

			toRemove = append(toRemove, sig, mul)
			break
		}
	}
	removeNodes(g, toRemove)
}

// eliminateDeadNodes removes nodes whose outputs are not consumed by anyone.
func eliminateDeadNodes(g *ir.Graph) {
	for {
		used := make(map[string]bool)
		for _, out := range g.Outputs {
			used[out.Name] = true
		}
		for _, n := range g.Nodes {
			for _, inp := range n.Inputs {
				if inp != "" {
					used[inp] = true
				}
			}
		}

		changed := false
		filtered := g.Nodes[:0]
		for _, n := range g.Nodes {
			keep := false
			for _, out := range n.Outputs {
				if used[out] {
					keep = true
					break
				}
			}
			if keep {
				filtered = append(filtered, n)
			} else {
				changed = true
			}
		}
		g.Nodes = filtered
		if !changed {
			break
		}
	}

	// Clean up unused initializers
	used := make(map[string]bool)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			used[inp] = true
		}
	}
	for name := range g.Initializers {
		if !used[name] {
			delete(g.Initializers, name)
		}
	}
}

func removeNodes(g *ir.Graph, nodes []*ir.Node) {
	if len(nodes) == 0 {
		return
	}
	remove := make(map[*ir.Node]bool)
	for _, n := range nodes {
		remove[n] = true
	}
	filtered := g.Nodes[:0]
	for _, n := range g.Nodes {
		if !remove[n] {
			filtered = append(filtered, n)
		}
	}
	g.Nodes = filtered
}

// fuseGELU replaces Div(x,sqrt2)->Erf->Add(1)->Mul(x)->Mul(0.5) with a single FastGELU op.
// isConstScalarApprox checks if a tensor name is a scalar initializer with approximate value.
// Constant ops are already converted to initializers by materializeConstants.
func isConstScalarApprox(g *ir.Graph, name string, target float32) bool {
	init, ok := g.Initializers[name]
	if !ok {
		return false
	}
	data, err := materialize.Float32(init)
	return err == nil && len(data) == 1 && data[0] > target-0.01 && data[0] < target+0.01
}

func fuseGELU(g *ir.Graph) {
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			producer[out] = n
		}
	}
	useCount := make(map[string]int)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	var toRemove []*ir.Node
	for _, lastMul := range g.Nodes {
		// Pattern: Mul(half, Mul(x, Add(1, Erf(Div(x, sqrt2)))))
		// or: Mul(Mul(x, Add(1, Erf(Div(x, sqrt2)))), half)
		if lastMul.OpType != "Mul" {
			continue
		}

		// Find the 0.5 constant and the inner Mul
		var innerMul *ir.Node
		for _, inp := range lastMul.Inputs {
			if isConstScalarApprox(g, inp, 0.5) {
				// Found 0.5 constant, other input is inner Mul
				for _, inp2 := range lastMul.Inputs {
					if inp2 != inp {
						if p, ok := producer[inp2]; ok && p.OpType == "Mul" {
							innerMul = p
						}
					}
				}
			}
		}
		if innerMul == nil || useCount[innerMul.Outputs[0]] != 1 {
			continue
		}

		// innerMul = Mul(x, Add(1, Erf(...)))
		var addNode *ir.Node
		var xName string
		for _, inp := range innerMul.Inputs {
			if p, ok := producer[inp]; ok && p.OpType == "Add" && useCount[inp] == 1 {
				addNode = p
				// The other input is x
				for _, inp2 := range innerMul.Inputs {
					if inp2 != inp {
						xName = inp2
					}
				}
			}
		}
		if addNode == nil || xName == "" {
			continue
		}

		// addNode = Add(1, Erf(...))
		var erfNode *ir.Node
		for _, inp := range addNode.Inputs {
			if p, ok := producer[inp]; ok && p.OpType == "Erf" && useCount[inp] == 1 {
				erfNode = p
			}
		}
		if erfNode == nil {
			continue
		}

		// erfNode = Erf(Div(x, sqrt2))
		divInput := erfNode.Inputs[0]
		divNode, ok := producer[divInput]
		if !ok || divNode.OpType != "Div" || useCount[divInput] != 1 {
			continue
		}

		// Verify divNode input[0] is the same x
		if divNode.Inputs[0] != xName {
			continue
		}

		// SUCCESS: Replace 5 nodes with FastGELU
		lastMul.OpType = "FastGELU"
		lastMul.Inputs = []string{xName}
		// Keep lastMul.Outputs as is

		toRemove = append(toRemove, innerMul, addNode, erfNode, divNode)
	}
	removeNodes(g, toRemove)
}

// fuseMatMulAddBias folds MatMul ↁEAdd(const bias) into FusedMatMul.
func fuseMatMulAddBias(g *ir.Graph) {
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			producer[out] = n
		}
	}
	useCount := make(map[string]int)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	var toRemove []*ir.Node
	for _, add := range g.Nodes {
		if add.OpType != "Add" || len(add.Inputs) < 2 {
			continue
		}

		var matmul *ir.Node
		var biasName string

		for idx := 0; idx < 2; idx++ {
			name := add.Inputs[idx]
			otherName := add.Inputs[1-idx]
			if p, ok := producer[name]; ok && p.OpType == "MatMul" {
				if useCount[name] == 1 {
					if _, isInit := g.Initializers[otherName]; isInit {
						matmul = p
						biasName = otherName
						break
					}
				}
			}
		}
		if matmul == nil {
			continue
		}

		// Convert to FusedMatMul with bias
		matmul.OpType = "FusedMatMul"
		matmul.Inputs = append(matmul.Inputs, biasName)

		// Rewire: Add consumers now use MatMul output
		addOut := add.Outputs[0]
		matmul.Outputs[0] = addOut

		toRemove = append(toRemove, add)
	}
	removeNodes(g, toRemove)
}

// fuseMulAddAffine merges Mul(X, scale_const) ↁEAdd(_, bias_const) into FusedAffine.
func fuseMulAddAffine(g *ir.Graph) {
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			producer[out] = n
		}
	}
	useCount := make(map[string]int)
	for _, n := range g.Nodes {
		for _, inp := range n.Inputs {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	var toRemove []*ir.Node
	for _, add := range g.Nodes {
		if add.OpType != "Add" || len(add.Inputs) < 2 {
			continue
		}
		for idx := 0; idx < 2; idx++ {
			name := add.Inputs[idx]
			otherName := add.Inputs[1-idx]
			if _, isInit := g.Initializers[otherName]; !isInit {
				continue
			}
			p, ok := producer[name]
			if !ok || p.OpType != "Mul" || useCount[name] != 1 {
				continue
			}
			var scaleName, mulXName string
			for _, mInp := range p.Inputs {
				if _, isInit := g.Initializers[mInp]; isInit {
					scaleName = mInp
				} else {
					mulXName = mInp
				}
			}
			if scaleName == "" || mulXName == "" {
				continue
			}

			addOut := add.Outputs[0]
			p.OpType = "FusedAffine"
			p.Inputs = []string{mulXName, scaleName, otherName}
			p.Outputs[0] = addOut
			toRemove = append(toRemove, add)
			break
		}
	}
	removeNodes(g, toRemove)
}
