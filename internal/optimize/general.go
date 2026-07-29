package optimize

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ops"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

const maxFoldElements = 1 << 20

var foldableOps = map[string]bool{
	"Add": true, "Sub": true, "Mul": true, "Div": true,
	"Cast": true, "Reshape": true, "Transpose": true,
	"Squeeze": true, "Unsqueeze": true, "Flatten": true,
	"Concat": true, "Shape": true, "Gather": true, "Slice": true,
}

// foldConstants evaluates small, side-effect-free constant subgraphs using the
// same kernels as runtime execution. Unsupported or invalid folds are skipped.
func foldConstants(g *ir.Graph) {
	reg := ops.NewRegistry()
	ops.RegisterAll(reg)
	for {
		changed := false
		for i, n := range g.Nodes {
			if n.Domain != "" || !foldableOps[n.OpType] || len(n.Outputs) != 1 {
				continue
			}
			inputs := make([]tensor.Tensor, len(n.Inputs))
			ready := true
			for j, name := range n.Inputs {
				if name == "" {
					continue
				}
				init := g.Initializers[name]
				if init == nil {
					ready = false
					break
				}
				t, err := materialize.Tensor(init)
				if err != nil || t.Len() > maxFoldElements {
					ready = false
					break
				}
				inputs[j] = t
			}
			if !ready {
				continue
			}
			fn := reg.Lookup(n.OpType)
			if fn == nil {
				continue
			}
			outputs, ok := safeConstEval(fn, n, inputs)
			if !ok || len(outputs) != 1 || outputs[0] == nil || outputs[0].Len() > maxFoldElements {
				continue
			}
			init, err := tensorInitializer(n.Outputs[0], outputs[0])
			if err != nil {
				continue
			}
			g.Initializers[n.Outputs[0]] = init
			g.Nodes = append(g.Nodes[:i], g.Nodes[i+1:]...)
			changed = true
			break
		}
		if !changed {
			return
		}
	}
}

func safeConstEval(fn ops.OpFunc, n *ir.Node, inputs []tensor.Tensor) (out []tensor.Tensor, ok bool) {
	defer func() {
		if recover() != nil {
			out = nil
			ok = false
		}
	}()
	out, err := fn(n, inputs)
	return out, err == nil
}

func tensorInitializer(name string, t tensor.Tensor) (*ir.Initializer, error) {
	shape := make(ir.Shape, len(t.Shape()))
	for i, d := range t.Shape() {
		shape[i] = int64(d)
	}
	init := &ir.Initializer{Name: name, Shape: shape}
	switch v := t.(type) {
	case *tensor.Dense[float32]:
		init.DType = ir.DataTypeFloat
		init.FloatData = append([]float32(nil), v.Data()...)
	case *tensor.Dense[float64]:
		init.DType = ir.DataTypeDouble
		init.DoubleData = append([]float64(nil), v.Data()...)
	case *tensor.Dense[int32]:
		init.DType = ir.DataTypeInt32
		init.Int32Data = append([]int32(nil), v.Data()...)
	case *tensor.Dense[int64]:
		init.DType = ir.DataTypeInt64
		init.Int64Data = append([]int64(nil), v.Data()...)
	default:
		return nil, fmt.Errorf("constant folding: unsupported result %T", t)
	}
	return init, nil
}

// simplifyTransposes removes identity transposes and composes adjacent ones.
func simplifyTransposes(g *ir.Graph) {
	for {
		producer := producerMap(g)
		uses := valueUseCounts(g)
		changed := false
		for _, n := range g.Nodes {
			if n.OpType != "Transpose" || len(n.Inputs) != 1 || len(n.Outputs) != 1 {
				continue
			}
			perm := n.GetAttrInts("perm", nil)
			if isIdentityPerm(perm) {
				replaceValue(g, n.Outputs[0], n.Inputs[0])
				removeNodes(g, []*ir.Node{n})
				changed = true
				break
			}
			prev := producer[n.Inputs[0]]
			if prev == nil || prev.OpType != "Transpose" || uses[n.Inputs[0]] != 1 {
				continue
			}
			p1, p2 := prev.GetAttrInts("perm", nil), perm
			if len(p1) == 0 || len(p1) != len(p2) {
				continue
			}
			composed := make([]int64, len(p1))
			valid := true
			for i, p := range p2 {
				if p < 0 || int(p) >= len(p1) {
					valid = false
					break
				}
				composed[i] = p1[p]
			}
			if !valid {
				continue
			}
			n.Inputs[0] = prev.Inputs[0]
			if isIdentityPerm(composed) {
				replaceValue(g, n.Outputs[0], n.Inputs[0])
				removeNodes(g, []*ir.Node{prev, n})
			} else {
				n.Attrs["perm"] = ir.AttrInts{Value: composed}
				removeNodes(g, []*ir.Node{prev})
			}
			changed = true
			break
		}
		if !changed {
			return
		}
	}
}

func isIdentityPerm(p []int64) bool {
	if len(p) == 0 {
		return false
	}
	for i, v := range p {
		if int64(i) != v {
			return false
		}
	}
	return true
}

func producerMap(g *ir.Graph) map[string]*ir.Node {
	m := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			m[out] = n
		}
	}
	return m
}

func valueUseCounts(g *ir.Graph) map[string]int {
	m := make(map[string]int)
	for _, n := range g.Nodes {
		for _, in := range n.Inputs {
			if in != "" {
				m[in]++
			}
		}
	}
	for _, out := range g.Outputs {
		m[out.Name]++
	}
	return m
}

func replaceValue(g *ir.Graph, old, replacement string) {
	for _, n := range g.Nodes {
		for i, in := range n.Inputs {
			if in == old {
				n.Inputs[i] = replacement
			}
		}
	}
	for i := range g.Outputs {
		if g.Outputs[i].Name == old {
			g.Outputs[i].Name = replacement
		}
	}
}
