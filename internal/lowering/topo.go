package lowering

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
)

// TopoSort returns a topological execution order for a canonical graph.
func TopoSort(g *ir.Graph) ([]*ir.Node, error) {
	producer := make(map[string]*ir.Node)
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			if out != "" {
				producer[out] = n
			}
		}
	}

	type nodeInfo struct {
		node     *ir.Node
		inDegree int
	}
	nodeMap := make(map[*ir.Node]*nodeInfo)
	for _, n := range g.Nodes {
		ni := &nodeInfo{node: n}
		for _, inp := range nodeDependencies(n) {
			if inp == "" {
				continue
			}
			if p, isProduced := producer[inp]; isProduced && p != n {
				ni.inDegree++
			}
		}
		nodeMap[n] = ni
	}

	adj := make(map[*ir.Node][]*ir.Node)
	for _, n := range g.Nodes {
		for _, inp := range nodeDependencies(n) {
			if inp == "" {
				continue
			}
			if p, ok := producer[inp]; ok && p != n {
				adj[p] = append(adj[p], n)
			}
		}
	}

	var queue []*ir.Node
	for _, ni := range nodeMap {
		if ni.inDegree == 0 {
			queue = append(queue, ni.node)
		}
	}

	var sorted []*ir.Node
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		sorted = append(sorted, n)
		for _, dep := range adj[n] {
			ni := nodeMap[dep]
			ni.inDegree--
			if ni.inDegree == 0 {
				queue = append(queue, dep)
			}
		}
	}

	if len(sorted) != len(g.Nodes) {
		return nil, fmt.Errorf("lowering: graph has a cycle (%d sorted vs %d total nodes)", len(sorted), len(g.Nodes))
	}

	return sorted, nil
}

func topoSort(g *ir.Graph) ([]*ir.Node, error) {
	return TopoSort(g)
}

func nodeDependencies(n *ir.Node) []string {
	deps := append([]string(nil), n.Inputs...)
	seen := make(map[string]struct{}, len(deps))
	for _, dep := range deps {
		if dep != "" {
			seen[dep] = struct{}{}
		}
	}
	for _, dep := range subgraphFreeVars(n) {
		if dep == "" {
			continue
		}
		if _, ok := seen[dep]; ok {
			continue
		}
		seen[dep] = struct{}{}
		deps = append(deps, dep)
	}
	return deps
}

func subgraphFreeVars(n *ir.Node) []string {
	var deps []string
	seen := map[string]struct{}{}
	for _, attr := range n.Attrs {
		ag, ok := attr.(*ir.AttrGraph)
		if !ok || ag == nil || ag.Value == nil {
			continue
		}
		for _, dep := range graphFreeVars(ag.Value) {
			if _, ok := seen[dep]; ok {
				continue
			}
			seen[dep] = struct{}{}
			deps = append(deps, dep)
		}
	}
	return deps
}

func graphFreeVars(g *ir.Graph) []string {
	local := make(map[string]struct{}, len(g.Inputs)+len(g.Initializers)+len(g.Nodes))
	for _, in := range g.Inputs {
		local[in.Name] = struct{}{}
	}
	for name := range g.Initializers {
		local[name] = struct{}{}
	}
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			if out != "" {
				local[out] = struct{}{}
			}
		}
	}

	var deps []string
	seen := map[string]struct{}{}
	for _, n := range g.Nodes {
		for _, inp := range nodeDependencies(n) {
			if inp == "" {
				continue
			}
			if _, ok := local[inp]; ok {
				continue
			}
			if _, ok := seen[inp]; ok {
				continue
			}
			seen[inp] = struct{}{}
			deps = append(deps, inp)
		}
	}
	return deps
}
