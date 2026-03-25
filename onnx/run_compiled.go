package onnx

import (
	"fmt"
	"time"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/lowering"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func (s *Session) runCompiled(inputs map[string]tensor.Tensor) (map[string]tensor.Tensor, error) {
	plan := s.plan
	runInfo := RunInfo{
		TotalNodes: len(plan.Instructions),
		Compiled:   true,
		StartedAt:  time.Now(),
	}
	s.observers.onRunStart(runInfo)

	result, err := s.executeCompiledPlan(plan, s.arena, inputs, nil, nil, nil, true)
	if err != nil {
		s.observers.onRunFinish(runInfo, err)
		return nil, err
	}

	s.observers.onRunFinish(runInfo, nil)
	return result, nil
}

func (s *Session) runCompiledDebug(inputs map[string]tensor.Tensor) (map[string]tensor.Tensor, error) {
	values := make(map[string]tensor.Tensor)
	_, err := s.executeCompiledPlanDebug(s.plan, s.arena, inputs, nil, nil, nil, "", values)
	if err != nil {
		return nil, err
	}
	return values, nil
}

func (s *Session) executeCompiledPlan(plan *lowering.Plan, arena *lowering.Arena, inputs map[string]tensor.Tensor, captures []lowering.CaptureEntry, outerPlan *lowering.Plan, outerSlots []tensor.Tensor, observe bool) (map[string]tensor.Tensor, error) {
	arena.Reset()
	slots := arena.Slots

	for i := range plan.InitSlots {
		ie := &plan.InitSlots[i]
		slots[ie.Slot] = ie.Val
	}

	for _, cap := range captures {
		if outerPlan == nil || outerSlots == nil {
			continue
		}
		outerSlot, ok := outerPlan.SlotByName[cap.Name]
		if !ok || outerSlot < 0 || outerSlot >= len(outerSlots) {
			continue
		}
		slots[cap.Slot] = outerSlots[outerSlot]
	}

	for name, t := range inputs {
		if slot, ok := plan.InputSlotMap[name]; ok {
			slots[slot] = t
		}
	}

	if err := s.executeCompiledInstructions(plan, slots, arena.InputBuf, observe); err != nil {
		return nil, err
	}
	return collectPlanOutputs(plan, slots)
}

func (s *Session) executeCompiledPlanDebug(plan *lowering.Plan, arena *lowering.Arena, inputs map[string]tensor.Tensor, captures []lowering.CaptureEntry, outerPlan *lowering.Plan, outerSlots []tensor.Tensor, prefix string, values map[string]tensor.Tensor) (map[string]tensor.Tensor, error) {
	arena.Reset()
	slots := arena.Slots

	for i := range plan.InitSlots {
		ie := &plan.InitSlots[i]
		slots[ie.Slot] = ie.Val
	}
	for _, cap := range captures {
		if outerPlan == nil || outerSlots == nil {
			continue
		}
		outerSlot, ok := outerPlan.SlotByName[cap.Name]
		if !ok || outerSlot < 0 || outerSlot >= len(outerSlots) {
			continue
		}
		slots[cap.Slot] = outerSlots[outerSlot]
	}
	for name, t := range inputs {
		if slot, ok := plan.InputSlotMap[name]; ok {
			slots[slot] = t
		}
	}
	recordPlanValues(plan, slots, prefix, values)

	for idx := range plan.Instructions {
		inst := &plan.Instructions[idx]
		for i, slot := range inst.InputSlots {
			if slot < 0 {
				arena.InputBuf[i] = nil
			} else {
				arena.InputBuf[i] = slots[slot]
			}
		}
		outputs, err := s.executeCompiledInstructionDebug(inst, plan, slots, arena.InputBuf[:inst.NumInputs], prefix, values)
		if err != nil {
			return nil, fmt.Errorf("engine: executing %s (%s): %w", inst.Node.Name, inst.Node.OpType, err)
		}
		for i, slot := range inst.OutputSlots {
			if slot >= 0 && i < len(outputs) {
				slots[slot] = outputs[i]
			}
		}
		recordNamedOutputs(inst, outputs, prefix, values)
	}

	return collectPlanOutputs(plan, slots)
}

func (s *Session) executeCompiledInstructions(plan *lowering.Plan, slots []tensor.Tensor, inputBuf []tensor.Tensor, observe bool) error {
	for idx := range plan.Instructions {
		inst := &plan.Instructions[idx]
		for i, slot := range inst.InputSlots {
			if slot < 0 {
				inputBuf[i] = nil
			} else {
				inputBuf[i] = slots[slot]
			}
		}

		var outputs []tensor.Tensor
		var err error
		if observe {
			startExec := NodeExecution{Node: inst.Node, Index: idx, Total: len(plan.Instructions)}
			s.observers.onNodeStart(startExec)
			if len(s.observers) > 0 {
				allocBefore := getAllocBytes()
				t0 := time.Now()
				outputs, err = s.executeCompiledInstruction(inst, plan, slots, inputBuf[:inst.NumInputs])
				elapsed := time.Since(t0)
				allocAfter := getAllocBytes()
				finishExec := startExec
				finishExec.Elapsed = elapsed
				finishExec.AllocBytes = allocAfter - allocBefore
				s.observers.onNodeFinish(finishExec, err)
			} else {
				outputs, err = s.executeCompiledInstruction(inst, plan, slots, inputBuf[:inst.NumInputs])
			}
		} else {
			outputs, err = s.executeCompiledInstruction(inst, plan, slots, inputBuf[:inst.NumInputs])
		}
		if err != nil {
			return fmt.Errorf("engine: executing %s (%s): %w", inst.Node.Name, inst.Node.OpType, err)
		}

		for i, slot := range inst.OutputSlots {
			if slot >= 0 && i < len(outputs) {
				slots[slot] = outputs[i]
			}
		}
	}
	return nil
}

func (s *Session) executeCompiledInstruction(inst *lowering.Instruction, plan *lowering.Plan, slots []tensor.Tensor, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if inst.IfControl != nil {
		return s.executeCompiledIf(inst, plan, slots, inputs)
	}
	return inst.OpFunc(inst.Node, inputs)
}

func (s *Session) executeCompiledInstructionDebug(inst *lowering.Instruction, plan *lowering.Plan, slots []tensor.Tensor, inputs []tensor.Tensor, prefix string, values map[string]tensor.Tensor) ([]tensor.Tensor, error) {
	if inst.IfControl != nil {
		return s.executeCompiledIfDebug(inst, plan, slots, inputs, prefix, values)
	}
	return inst.OpFunc(inst.Node, inputs)
}

func (s *Session) executeCompiledIf(inst *lowering.Instruction, plan *lowering.Plan, slots []tensor.Tensor, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	branch, err := selectIfBranch(inst, inputs)
	if err != nil {
		return nil, err
	}

	result, err := s.executeCompiledPlan(branch.Plan, branch.Arena, nil, branch.Captures, plan, slots, false)
	if err != nil {
		return nil, err
	}
	return branchOutputs(branch.Plan, result), nil
}

func (s *Session) executeCompiledIfDebug(inst *lowering.Instruction, plan *lowering.Plan, slots []tensor.Tensor, inputs []tensor.Tensor, prefix string, values map[string]tensor.Tensor) ([]tensor.Tensor, error) {
	branch, branchName, err := selectIfBranchWithName(inst, inputs)
	if err != nil {
		return nil, err
	}

	branchPrefix := prefix + inst.Node.Name + "/" + branchName + "/"
	result, err := s.executeCompiledPlanDebug(branch.Plan, branch.Arena, nil, branch.Captures, plan, slots, branchPrefix, values)
	if err != nil {
		return nil, err
	}
	return branchOutputs(branch.Plan, result), nil
}

func selectIfBranch(inst *lowering.Instruction, inputs []tensor.Tensor) (*lowering.CompiledSubgraph, error) {
	branch, _, err := selectIfBranchWithName(inst, inputs)
	return branch, err
}

func selectIfBranchWithName(inst *lowering.Instruction, inputs []tensor.Tensor) (*lowering.CompiledSubgraph, string, error) {
	if len(inputs) == 0 || inputs[0] == nil {
		return nil, "", fmt.Errorf("engine: If node %s has no condition input", inst.Node.Name)
	}

	cond, err := tensorAsBool(inputs[0])
	if err != nil {
		return nil, "", fmt.Errorf("engine: If node %s: %w", inst.Node.Name, err)
	}

	branch := inst.IfControl.Else
	branchName := "else_branch"
	if cond {
		branch = inst.IfControl.Then
		branchName = "then_branch"
	}
	if branch == nil || branch.Plan == nil || branch.Arena == nil {
		return nil, "", fmt.Errorf("engine: If node %s has no compiled branch", inst.Node.Name)
	}
	return branch, branchName, nil
}

func branchOutputs(plan *lowering.Plan, result map[string]tensor.Tensor) []tensor.Tensor {
	outputs := make([]tensor.Tensor, 0, len(plan.OutputSlots))
	for _, out := range plan.OutputSlots {
		outputs = append(outputs, result[out.Name])
	}
	return outputs
}

func collectPlanOutputs(plan *lowering.Plan, slots []tensor.Tensor) (map[string]tensor.Tensor, error) {
	result := make(map[string]tensor.Tensor, len(plan.OutputSlots))
	for i := range plan.OutputSlots {
		oe := &plan.OutputSlots[i]
		t := slots[oe.Slot]
		if t == nil {
			return nil, fmt.Errorf("engine: output %q not found after execution", oe.Name)
		}
		result[oe.Name] = t
	}
	return result, nil
}

func recordPlanValues(plan *lowering.Plan, slots []tensor.Tensor, prefix string, values map[string]tensor.Tensor) {
	for name, slot := range plan.SlotByName {
		if slot < 0 || slot >= len(slots) {
			continue
		}
		if t := slots[slot]; t != nil {
			values[prefix+name] = t
		}
	}
}

func recordNamedOutputs(inst *lowering.Instruction, outputs []tensor.Tensor, prefix string, values map[string]tensor.Tensor) {
	for i, name := range inst.Node.Outputs {
		if name == "" || i >= len(outputs) || outputs[i] == nil {
			continue
		}
		values[prefix+name] = outputs[i]
	}
}

func tensorAsBool(t tensor.Tensor) (bool, error) {
	switch c := t.(type) {
	case *tensor.Dense[uint8]:
		return c.Len() > 0 && c.At(0) != 0, nil
	default:
		return false, fmt.Errorf("condition type %T is unsupported", t)
	}
}
