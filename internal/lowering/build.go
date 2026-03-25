package lowering

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ops"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// BuildArtifacts holds the runtime-facing outputs of lowering preparation.
type BuildArtifacts struct {
	Order        []*ir.Node
	Initializers map[string]tensor.Tensor
	Plan         *Plan
	Arena        *Arena
	HasSubgraphs bool
}

// Prepare performs runtime-oriented lowering preparation from canonical IR.
func Prepare(g *ir.Graph, reg *ops.Registry) (*BuildArtifacts, error) {
	order, err := topoSort(g)
	if err != nil {
		return nil, err
	}

	inits := make(map[string]tensor.Tensor, len(g.Initializers))
	for name, init := range g.Initializers {
		t, err := materialize.Tensor(init)
		if err != nil {
			return nil, fmt.Errorf("materialize initializer %s: %w", name, err)
		}
		inits[name] = t
	}

	prepackConstWeights(g, inits)

	plan, err := Compile(g, order, reg, inits)
	if err != nil {
		return nil, err
	}

	return &BuildArtifacts{
		Order:        order,
		Initializers: inits,
		Plan:         plan,
		Arena:        NewArena(plan),
		HasSubgraphs: hasSubgraphs(g),
	}, nil
}

func hasSubgraphs(g *ir.Graph) bool {
	for _, node := range g.Nodes {
		if node.OpType == "If" || node.OpType == "Loop" || node.OpType == "Scan" {
			return true
		}
		for _, attr := range node.Attrs {
			if ag, ok := attr.(*ir.AttrGraph); ok && ag.Value != nil && hasSubgraphs(ag.Value) {
				return true
			}
		}
	}
	return false
}

func prepackConstWeights(g *ir.Graph, inits map[string]tensor.Tensor) {
	candidates := make(map[string]bool)
	for _, node := range g.Nodes {
		if (node.OpType == "MatMul" || node.OpType == "FusedMatMul") && len(node.Inputs) >= 2 {
			rhs := node.Inputs[1]
			if _, isInit := inits[rhs]; isInit {
				candidates[rhs] = true
			}
		}
	}

	for name := range candidates {
		t := inits[name]
		dt, ok := t.(*tensor.Dense[float32])
		if !ok {
			continue
		}
		shape := dt.Shape()
		if shape.NDim() != 2 {
			continue
		}
		k, n := shape[0], shape[1]
		if k < 16 || n < 8 {
			continue
		}
		packed := ops.PackMatMulB(dt.Data(), k, n)
		inits[name] = tensor.NewPackedF32(shape, k, n, packed.Data, dt.Data())
	}
}
