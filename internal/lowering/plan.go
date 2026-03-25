package lowering

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ops"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// Instruction is a compiled execution step with slot-based addressing.
type Instruction struct {
	OpFunc      ops.OpFunc
	Node        *ir.Node
	InputSlots  []int16
	OutputSlots []int16
	NumInputs   int
	NumOutputs  int
	IsView      bool
	IfControl   *IfControl
}

type CaptureEntry struct {
	Name string
	Slot int
}

type CompiledSubgraph struct {
	Plan     *Plan
	Arena    *Arena
	Captures []CaptureEntry
}

type IfControl struct {
	Then *CompiledSubgraph
	Else *CompiledSubgraph
}

type InitEntry struct {
	Slot int
	Val  tensor.Tensor
}

type InputEntry struct {
	Name string
	Slot int
}

type OutputEntry struct {
	Name string
	Slot int
}

// Plan is the lowered execution representation consumed by the runtime.
type Plan struct {
	Instructions []Instruction
	SlotCount    int
	InputSlotMap map[string]int
	SlotByName   map[string]int
	InitSlots    []InitEntry
	InputSlots   []InputEntry
	OutputSlots  []OutputEntry
	MaxInputs    int
}

// Arena stores reusable runtime buffers for a lowered plan.
type Arena struct {
	Slots    []tensor.Tensor
	InputBuf []tensor.Tensor
}

var viewOps = map[string]bool{
	"Identity": true,
	"Dropout":  true,
}

// Compile lowers the canonical graph into a runtime plan.
func Compile(g *ir.Graph, order []*ir.Node, reg *ops.Registry, inits map[string]tensor.Tensor) (*Plan, error) {
	nameToSlot := make(map[string]int)
	nextSlot := 0

	getSlot := func(name string) int {
		if name == "" {
			return -1
		}
		if id, ok := nameToSlot[name]; ok {
			return id
		}
		id := nextSlot
		nameToSlot[name] = id
		nextSlot++
		return id
	}

	var initSlots []InitEntry
	for name, t := range inits {
		slot := getSlot(name)
		initSlots = append(initSlots, InitEntry{Slot: slot, Val: t})
	}

	var inputSlots []InputEntry
	inputSlotMap := make(map[string]int, len(g.Inputs))
	for _, inp := range g.Inputs {
		slot := getSlot(inp.Name)
		inputSlots = append(inputSlots, InputEntry{Name: inp.Name, Slot: slot})
		inputSlotMap[inp.Name] = slot
	}

	maxInputs := 0
	instructions := make([]Instruction, 0, len(order))
	for _, node := range order {
		opKey := node.OpType
		if node.Domain != "" {
			opKey = node.Domain + "::" + node.OpType
		}
		opFunc := reg.Lookup(opKey)
		if opFunc == nil {
			return nil, fmt.Errorf("lowering: unsupported op %q (node %s)", opKey, node.Name)
		}

		var ifControl *IfControl
		if node.OpType == "If" {
			compiled, err := compileIfControl(node, reg)
			if err != nil {
				return nil, err
			}
			ifControl = compiled
		}

		iSlots := make([]int16, len(node.Inputs))
		for i, name := range node.Inputs {
			iSlots[i] = int16(getSlot(name))
		}
		oSlots := make([]int16, len(node.Outputs))
		for i, name := range node.Outputs {
			oSlots[i] = int16(getSlot(name))
		}

		instructions = append(instructions, Instruction{
			OpFunc:      opFunc,
			Node:        node,
			InputSlots:  iSlots,
			OutputSlots: oSlots,
			NumInputs:   len(node.Inputs),
			NumOutputs:  len(node.Outputs),
			IsView:      viewOps[node.OpType],
			IfControl:   ifControl,
		})
		if len(node.Inputs) > maxInputs {
			maxInputs = len(node.Inputs)
		}
	}

	var outputSlots []OutputEntry
	for _, out := range g.Outputs {
		slot := getSlot(out.Name)
		outputSlots = append(outputSlots, OutputEntry{Name: out.Name, Slot: slot})
	}

	return &Plan{
		Instructions: instructions,
		SlotCount:    nextSlot,
		InputSlotMap: inputSlotMap,
		SlotByName:   nameToSlot,
		InitSlots:    initSlots,
		InputSlots:   inputSlots,
		OutputSlots:  outputSlots,
		MaxInputs:    maxInputs,
	}, nil
}

func compileIfControl(node *ir.Node, reg *ops.Registry) (*IfControl, error) {
	thenPlan, err := compileSubgraph(node, "then_branch", reg)
	if err != nil {
		return nil, err
	}
	elsePlan, err := compileSubgraph(node, "else_branch", reg)
	if err != nil {
		return nil, err
	}
	return &IfControl{Then: thenPlan, Else: elsePlan}, nil
}

func compileSubgraph(node *ir.Node, attrName string, reg *ops.Registry) (*CompiledSubgraph, error) {
	attr, ok := node.Attrs[attrName]
	if !ok {
		return nil, fmt.Errorf("lowering: If node %s missing %s", node.Name, attrName)
	}
	ag, ok := attr.(*ir.AttrGraph)
	if !ok || ag == nil || ag.Value == nil {
		return nil, fmt.Errorf("lowering: If node %s has invalid %s", node.Name, attrName)
	}

	order, err := topoSort(ag.Value)
	if err != nil {
		return nil, err
	}

	inits := make(map[string]tensor.Tensor, len(ag.Value.Initializers))
	for name, init := range ag.Value.Initializers {
		t, err := materialize.Tensor(init)
		if err != nil {
			return nil, fmt.Errorf("materialize initializer %s: %w", name, err)
		}
		inits[name] = t
	}
	prepackConstWeights(ag.Value, inits)

	plan, err := Compile(ag.Value, order, reg, inits)
	if err != nil {
		return nil, err
	}

	freeVars := graphFreeVars(ag.Value)
	captures := make([]CaptureEntry, 0, len(freeVars))
	for _, name := range freeVars {
		slot, ok := plan.SlotByName[name]
		if !ok {
			continue
		}
		captures = append(captures, CaptureEntry{Name: name, Slot: slot})
	}

	return &CompiledSubgraph{
		Plan:     plan,
		Arena:    NewArena(plan),
		Captures: captures,
	}, nil
}

// NewArena allocates reusable runtime storage for a plan.
func NewArena(plan *Plan) *Arena {
	return &Arena{
		Slots:    make([]tensor.Tensor, plan.SlotCount),
		InputBuf: make([]tensor.Tensor, plan.MaxInputs),
	}
}

// Reset clears runtime storage between runs.
func (a *Arena) Reset() {
	for i := range a.Slots {
		a.Slots[i] = nil
	}
}
