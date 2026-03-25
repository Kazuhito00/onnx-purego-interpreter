package onnx

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/frontend"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
)

// FrontendDOT returns a Graphviz DOT view of the frontend ONNX-like graph.
// フロントエンド ONNX グラフの Graphviz DOT 文字列を返す。
func (s *Session) FrontendDOT() string {
	if s.frontendModel == nil {
		return "digraph frontend {}\n"
	}
	return frontendModelDOT(s.frontendModel)
}

// CanonicalDOT returns a Graphviz DOT view of the canonical IR graph.
// 正規化済み IR グラフの Graphviz DOT 文字列を返す。
func (s *Session) CanonicalDOT() string {
	if s.graph == nil {
		return "digraph canonical {}\n"
	}
	return canonicalGraphDOT(s.graph)
}

// PlanDOT returns a Graphviz DOT view of the lowered execution plan.
// lowering 済み実行プランの Graphviz DOT 文字列を返す。
func (s *Session) PlanDOT() string {
	if s.plan == nil {
		return "digraph plan {}\n"
	}

	var b strings.Builder
	b.WriteString("digraph plan {\n")
	b.WriteString("  rankdir=LR;\n")
	b.WriteString("  node [fontname=\"Consolas\"];\n")

	for _, in := range s.plan.InputSlots {
		fmt.Fprintf(&b, "  %q [shape=oval, label=%q];\n", fmt.Sprintf("slot_%d", in.Slot), fmt.Sprintf("slot %d\\ninput:%s", in.Slot, in.Name))
	}
	for _, init := range s.plan.InitSlots {
		fmt.Fprintf(&b, "  %q [shape=note, label=%q];\n", fmt.Sprintf("slot_%d", init.Slot), fmt.Sprintf("slot %d\\ninitializer", init.Slot))
	}
	for idx, inst := range s.plan.Instructions {
		nodeID := fmt.Sprintf("inst_%d", idx)
		label := inst.Node.OpType
		if inst.Node.Name != "" {
			label += "\\n" + inst.Node.Name
		}
		label += fmt.Sprintf("\\n#%d", idx)
		fmt.Fprintf(&b, "  %q [shape=box, label=%q];\n", nodeID, label)
		for _, slot := range inst.InputSlots {
			if slot < 0 {
				continue
			}
			fmt.Fprintf(&b, "  %q -> %q;\n", fmt.Sprintf("slot_%d", slot), nodeID)
		}
		for _, slot := range inst.OutputSlots {
			if slot < 0 {
				continue
			}
			fmt.Fprintf(&b, "  %q [shape=ellipse, label=%q];\n", fmt.Sprintf("slot_%d", slot), fmt.Sprintf("slot %d", slot))
			fmt.Fprintf(&b, "  %q -> %q;\n", nodeID, fmt.Sprintf("slot_%d", slot))
		}
	}
	for _, out := range s.plan.OutputSlots {
		fmt.Fprintf(&b, "  %q [shape=oval, label=%q];\n", fmt.Sprintf("output_%s", out.Name), fmt.Sprintf("output:%s", out.Name))
		fmt.Fprintf(&b, "  %q -> %q;\n", fmt.Sprintf("slot_%d", out.Slot), fmt.Sprintf("output_%s", out.Name))
	}

	b.WriteString("}\n")
	return b.String()
}

// FrontendJSON returns a JSON view of the frontend graph.
func (s *Session) FrontendJSON() ([]byte, error) {
	if s.frontendModel == nil {
		return json.Marshal(map[string]any{"graph": nil})
	}
	type node struct {
		Name    string   `json:"name"`
		OpType  string   `json:"op_type"`
		Domain  string   `json:"domain,omitempty"`
		Inputs  []string `json:"inputs"`
		Outputs []string `json:"outputs"`
	}
	payload := map[string]any{
		"name":    s.frontendModel.Graph.Name,
		"inputs":  s.frontendModel.Graph.Inputs,
		"outputs": s.frontendModel.Graph.Outputs,
		"nodes": func() []node {
			out := make([]node, 0, len(s.frontendModel.Graph.Nodes))
			for _, n := range s.frontendModel.Graph.Nodes {
				out = append(out, node{
					Name:    n.Name,
					OpType:  n.OpType,
					Domain:  n.Domain,
					Inputs:  append([]string(nil), n.Inputs...),
					Outputs: append([]string(nil), n.Outputs...),
				})
			}
			return out
		}(),
	}
	return json.MarshalIndent(payload, "", "  ")
}

// CanonicalJSON returns a JSON view of the canonical graph.
func (s *Session) CanonicalJSON() ([]byte, error) {
	if s.graph == nil {
		return json.Marshal(map[string]any{"graph": nil})
	}
	type node struct {
		Name         string   `json:"name"`
		OpType       string   `json:"op_type"`
		Domain       string   `json:"domain,omitempty"`
		OpsetVersion int64    `json:"opset_version"`
		Inputs       []string `json:"inputs"`
		Outputs      []string `json:"outputs"`
	}
	payload := map[string]any{
		"name":    s.graph.Name,
		"inputs":  s.graph.Inputs,
		"outputs": s.graph.Outputs,
		"initializers": func() []string {
			names := make([]string, 0, len(s.graph.Initializers))
			for name := range s.graph.Initializers {
				names = append(names, name)
			}
			sort.Strings(names)
			return names
		}(),
		"nodes": func() []node {
			out := make([]node, 0, len(s.graph.Nodes))
			for _, n := range s.graph.Nodes {
				out = append(out, node{
					Name:         n.Name,
					OpType:       n.OpType,
					Domain:       n.Domain,
					OpsetVersion: n.OpsetVersion,
					Inputs:       append([]string(nil), n.Inputs...),
					Outputs:      append([]string(nil), n.Outputs...),
				})
			}
			return out
		}(),
	}
	return json.MarshalIndent(payload, "", "  ")
}

// PlanJSON returns a JSON view of the lowered execution plan.
func (s *Session) PlanJSON() ([]byte, error) {
	if s.plan == nil {
		return json.Marshal(map[string]any{"plan": nil})
	}
	type inst struct {
		Index       int     `json:"index"`
		Name        string  `json:"name"`
		OpType      string  `json:"op_type"`
		InputSlots  []int16 `json:"input_slots"`
		OutputSlots []int16 `json:"output_slots"`
		NumInputs   int     `json:"num_inputs"`
		NumOutputs  int     `json:"num_outputs"`
		IsView      bool    `json:"is_view"`
	}
	payload := map[string]any{
		"slot_count": s.plan.SlotCount,
		"max_inputs": s.plan.MaxInputs,
		"instructions": func() []inst {
			out := make([]inst, 0, len(s.plan.Instructions))
			for i, in := range s.plan.Instructions {
				out = append(out, inst{
					Index:       i,
					Name:        in.Node.Name,
					OpType:      in.Node.OpType,
					InputSlots:  append([]int16(nil), in.InputSlots...),
					OutputSlots: append([]int16(nil), in.OutputSlots...),
					NumInputs:   in.NumInputs,
					NumOutputs:  in.NumOutputs,
					IsView:      in.IsView,
				})
			}
			return out
		}(),
		"inputs":  s.plan.InputSlots,
		"outputs": s.plan.OutputSlots,
	}
	return json.MarshalIndent(payload, "", "  ")
}

func frontendModelDOT(m *frontend.Model) string {
	var b strings.Builder
	b.WriteString("digraph frontend {\n")
	b.WriteString("  rankdir=LR;\n")
	b.WriteString("  node [shape=box, fontname=\"Consolas\"];\n")

	for _, inp := range m.Graph.Inputs {
		fmt.Fprintf(&b, "  %q [shape=oval, label=%q];\n", "input:"+inp.Name, labelWithShape(inp.Name, []int64(inp.Shape)))
	}
	for _, out := range m.Graph.Outputs {
		fmt.Fprintf(&b, "  %q [shape=oval, label=%q];\n", "output:"+out.Name, labelWithShape(out.Name, []int64(out.Shape)))
	}

	for i, n := range m.Graph.Nodes {
		nodeID := fmt.Sprintf("frontend_node_%d", i)
		fmt.Fprintf(&b, "  %q [label=%q];\n", nodeID, frontendNodeLabel(n))
		for _, in := range n.Inputs {
			if in == "" {
				continue
			}
			fmt.Fprintf(&b, "  %q -> %q;\n", "value:"+in, nodeID)
		}
		for _, out := range n.Outputs {
			if out == "" {
				continue
			}
			fmt.Fprintf(&b, "  %q -> %q;\n", nodeID, "value:"+out)
			fmt.Fprintf(&b, "  %q [shape=ellipse, label=%q];\n", "value:"+out, out)
		}
	}

	b.WriteString("}\n")
	return b.String()
}

func canonicalGraphDOT(g *ir.Graph) string {
	var b strings.Builder
	b.WriteString("digraph canonical {\n")
	b.WriteString("  rankdir=LR;\n")
	b.WriteString("  node [shape=box, fontname=\"Consolas\"];\n")

	for _, inp := range g.Inputs {
		fmt.Fprintf(&b, "  %q [shape=oval, label=%q];\n", "input:"+inp.Name, labelWithShape(inp.Name, inp.Shape))
	}
	for _, out := range g.Outputs {
		fmt.Fprintf(&b, "  %q [shape=oval, label=%q];\n", "output:"+out.Name, labelWithShape(out.Name, out.Shape))
	}

	names := make([]string, 0, len(g.Initializers))
	for name := range g.Initializers {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		init := g.Initializers[name]
		fmt.Fprintf(&b, "  %q [shape=note, label=%q];\n", "init:"+name, labelWithShape(name, init.Shape))
	}

	for i, n := range g.Nodes {
		nodeID := fmt.Sprintf("canonical_node_%d", i)
		fmt.Fprintf(&b, "  %q [label=%q];\n", nodeID, canonicalNodeLabel(n))
		for _, in := range n.Inputs {
			if in == "" {
				continue
			}
			switch {
			case g.Initializers[in] != nil:
				fmt.Fprintf(&b, "  %q -> %q;\n", "init:"+in, nodeID)
			default:
				fmt.Fprintf(&b, "  %q -> %q;\n", "value:"+in, nodeID)
			}
		}
		for _, out := range n.Outputs {
			if out == "" {
				continue
			}
			fmt.Fprintf(&b, "  %q -> %q;\n", nodeID, "value:"+out)
			fmt.Fprintf(&b, "  %q [shape=ellipse, label=%q];\n", "value:"+out, out)
		}
	}

	b.WriteString("}\n")
	return b.String()
}

func frontendNodeLabel(n *frontend.Node) string {
	if n.Name == "" {
		return n.OpType
	}
	return n.OpType + "\\n" + n.Name
}

func canonicalNodeLabel(n *ir.Node) string {
	if n.Name == "" {
		return n.OpType
	}
	return n.OpType + "\\n" + n.Name
}

func labelWithShape(name string, shape []int64) string {
	if len(shape) == 0 {
		return name
	}
	return fmt.Sprintf("%s\\n%v", name, shape)
}
