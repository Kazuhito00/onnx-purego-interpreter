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
	if inst.LoopControl != nil {
		return s.executeCompiledLoop(inst, plan, slots, inputs)
	}
	if inst.ScanControl != nil {
		return s.executeCompiledScan(inst, plan, slots, inputs)
	}
	return inst.OpFunc(inst.Node, inputs)
}

func (s *Session) executeCompiledInstructionDebug(inst *lowering.Instruction, plan *lowering.Plan, slots []tensor.Tensor, inputs []tensor.Tensor, prefix string, values map[string]tensor.Tensor) ([]tensor.Tensor, error) {
	if inst.IfControl != nil {
		return s.executeCompiledIfDebug(inst, plan, slots, inputs, prefix, values)
	}
	if inst.LoopControl != nil {
		return s.executeCompiledLoop(inst, plan, slots, inputs)
	}
	if inst.ScanControl != nil {
		return s.executeCompiledScan(inst, plan, slots, inputs)
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

// executeCompiledLoop implements the ONNX Loop operator.
// Loop(M, cond, v_initial...) with body(i, cond_in, v_prev...) → (cond_out, v_next..., scan_out...)
// Node outputs: v_final..., scan_outputs...
func (s *Session) executeCompiledLoop(inst *lowering.Instruction, plan *lowering.Plan, slots []tensor.Tensor, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	body := inst.LoopControl.Body
	bodyPlan := body.Plan

	// Parse Loop inputs: M (max iterations), cond, v_initial...
	var maxIter int64 = 1<<63 - 1 // default: practically infinite
	if len(inputs) > 0 && inputs[0] != nil {
		if mi, ok := inputs[0].(*tensor.Dense[int64]); ok && mi.Len() > 0 {
			maxIter = mi.At(0)
		}
	}

	keepGoing := true
	if len(inputs) > 1 && inputs[1] != nil {
		var err error
		keepGoing, err = tensorAsBool(inputs[1])
		if err != nil {
			return nil, fmt.Errorf("engine: Loop node %s: cond: %w", inst.Node.Name, err)
		}
	}

	// v_initial (carried state)
	numState := len(inputs) - 2
	if numState < 0 {
		numState = 0
	}
	state := make([]tensor.Tensor, numState)
	for i := 0; i < numState; i++ {
		if i+2 < len(inputs) {
			state[i] = inputs[i+2]
		}
	}

	// Body outputs: cond_out, v_next[0..numState-1], scan_out[0..]
	// Number of scan outputs = body outputs - 1 - numState
	numBodyOutputs := len(bodyPlan.OutputSlots)
	numScanOutputs := numBodyOutputs - 1 - numState
	if numScanOutputs < 0 {
		numScanOutputs = 0
	}

	// Accumulate scan outputs across iterations
	scanAccum := make([][]tensor.Tensor, numScanOutputs)
	for j := range scanAccum {
		scanAccum[j] = make([]tensor.Tensor, 0)
	}

	bodyInputNames := make([]string, 0, len(bodyPlan.InputSlots))
	for _, ie := range bodyPlan.InputSlots {
		bodyInputNames = append(bodyInputNames, ie.Name)
	}

	for i := int64(0); i < maxIter && keepGoing; i++ {
		// Build body inputs: (i, cond, v_prev...)
		bodyInputs := make(map[string]tensor.Tensor)
		if len(bodyInputNames) > 0 {
			bodyInputs[bodyInputNames[0]] = tensor.NewDense[int64](tensor.Shape{}, []int64{i})
		}
		if len(bodyInputNames) > 1 {
			condVal := uint8(0)
			if keepGoing {
				condVal = 1
			}
			bodyInputs[bodyInputNames[1]] = tensor.NewDense[uint8](tensor.Shape{}, []uint8{condVal})
		}
		for j := 0; j < numState && j+2 < len(bodyInputNames); j++ {
			bodyInputs[bodyInputNames[j+2]] = state[j]
		}

		result, err := s.executeCompiledPlan(bodyPlan, body.Arena, bodyInputs, body.Captures, plan, slots, false)
		if err != nil {
			return nil, fmt.Errorf("engine: Loop node %s iteration %d: %w", inst.Node.Name, i, err)
		}

		// Collect body outputs by output slot order
		bodyOuts := make([]tensor.Tensor, numBodyOutputs)
		for idx, oe := range bodyPlan.OutputSlots {
			bodyOuts[idx] = result[oe.Name]
		}

		// Update condition
		if bodyOuts[0] != nil {
			keepGoing, _ = tensorAsBool(bodyOuts[0])
		} else {
			keepGoing = false
		}

		// Update carried state
		for j := 0; j < numState; j++ {
			state[j] = bodyOuts[1+j]
		}

		// Accumulate scan outputs
		for j := 0; j < numScanOutputs; j++ {
			scanAccum[j] = append(scanAccum[j], bodyOuts[1+numState+j])
		}
	}

	// Build node outputs: v_final..., scan_outputs...
	outputs := make([]tensor.Tensor, 0, numState+numScanOutputs)
	outputs = append(outputs, state...)
	for j := 0; j < numScanOutputs; j++ {
		outputs = append(outputs, stackTensors(scanAccum[j]))
	}
	return outputs, nil
}

// executeCompiledScan implements the ONNX Scan operator.
// Scan(initial_state..., scan_inputs...) with body(state_in..., scan_in...) → (state_out..., scan_out...)
func (s *Session) executeCompiledScan(inst *lowering.Instruction, plan *lowering.Plan, slots []tensor.Tensor, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	body := inst.ScanControl.Body
	bodyPlan := body.Plan
	numScanInputs := inst.ScanControl.NumScanInputs

	numInputs := len(inputs)
	numState := numInputs - numScanInputs
	if numState < 0 {
		return nil, fmt.Errorf("engine: Scan node %s: numState < 0 (inputs=%d, scan_inputs=%d)", inst.Node.Name, numInputs, numScanInputs)
	}

	// State tensors
	state := make([]tensor.Tensor, numState)
	for i := 0; i < numState; i++ {
		state[i] = inputs[i]
	}

	// Scan input tensors: each is [sequence_length, ...]
	scanInputs := make([]tensor.Tensor, numScanInputs)
	for i := 0; i < numScanInputs; i++ {
		scanInputs[i] = inputs[numState+i]
	}

	// Determine sequence length from first scan input
	if numScanInputs == 0 {
		return nil, fmt.Errorf("engine: Scan node %s: no scan inputs", inst.Node.Name)
	}
	seqLen := scanInputs[0].Shape()[0]

	// Body outputs: state_out..., scan_out...
	numBodyOutputs := len(bodyPlan.OutputSlots)
	numScanOutputs := numBodyOutputs - numState
	if numScanOutputs < 0 {
		numScanOutputs = 0
	}

	scanAccum := make([][]tensor.Tensor, numScanOutputs)
	for j := range scanAccum {
		scanAccum[j] = make([]tensor.Tensor, 0, seqLen)
	}

	bodyInputNames := make([]string, 0, len(bodyPlan.InputSlots))
	for _, ie := range bodyPlan.InputSlots {
		bodyInputNames = append(bodyInputNames, ie.Name)
	}

	for i := 0; i < seqLen; i++ {
		bodyInputMap := make(map[string]tensor.Tensor)
		// State inputs
		for j := 0; j < numState && j < len(bodyInputNames); j++ {
			bodyInputMap[bodyInputNames[j]] = state[j]
		}
		// Scan inputs: slice along axis 0
		for j := 0; j < numScanInputs && numState+j < len(bodyInputNames); j++ {
			bodyInputMap[bodyInputNames[numState+j]] = sliceTensorAt(scanInputs[j], i)
		}

		result, err := s.executeCompiledPlan(bodyPlan, body.Arena, bodyInputMap, body.Captures, plan, slots, false)
		if err != nil {
			return nil, fmt.Errorf("engine: Scan node %s step %d: %w", inst.Node.Name, i, err)
		}

		bodyOuts := make([]tensor.Tensor, numBodyOutputs)
		for idx, oe := range bodyPlan.OutputSlots {
			bodyOuts[idx] = result[oe.Name]
		}

		// Update state
		for j := 0; j < numState; j++ {
			state[j] = bodyOuts[j]
		}
		// Accumulate scan outputs
		for j := 0; j < numScanOutputs; j++ {
			scanAccum[j] = append(scanAccum[j], bodyOuts[numState+j])
		}
	}

	// Build outputs: state_final..., scan_outputs...
	outputs := make([]tensor.Tensor, 0, numState+numScanOutputs)
	outputs = append(outputs, state...)
	for j := 0; j < numScanOutputs; j++ {
		outputs = append(outputs, stackTensors(scanAccum[j]))
	}
	return outputs, nil
}

// stackTensors concatenates tensors along a new leading axis (axis 0).
// All tensors must have the same shape. Returns nil for empty input.
func stackTensors(ts []tensor.Tensor) tensor.Tensor {
	if len(ts) == 0 {
		return nil
	}
	first := ts[0]
	if first == nil {
		return nil
	}

	elemShape := first.Shape()
	elemSize := 1
	for _, d := range elemShape {
		elemSize *= d
	}

	newShape := make(tensor.Shape, 1+len(elemShape))
	newShape[0] = len(ts)
	copy(newShape[1:], elemShape)

	switch f := first.(type) {
	case *tensor.Dense[float32]:
		data := make([]float32, len(ts)*elemSize)
		for i, t := range ts {
			copy(data[i*elemSize:], t.(*tensor.Dense[float32]).Data())
		}
		return tensor.NewDense[float32](newShape, data)
	case *tensor.Dense[int64]:
		data := make([]int64, len(ts)*elemSize)
		for i, t := range ts {
			copy(data[i*elemSize:], t.(*tensor.Dense[int64]).Data())
		}
		return tensor.NewDense[int64](newShape, data)
	case *tensor.Dense[int32]:
		data := make([]int32, len(ts)*elemSize)
		for i, t := range ts {
			copy(data[i*elemSize:], t.(*tensor.Dense[int32]).Data())
		}
		return tensor.NewDense[int32](newShape, data)
	case *tensor.Dense[uint8]:
		data := make([]uint8, len(ts)*elemSize)
		for i, t := range ts {
			copy(data[i*elemSize:], t.(*tensor.Dense[uint8]).Data())
		}
		return tensor.NewDense[uint8](newShape, data)
	case *tensor.Dense[float64]:
		data := make([]float64, len(ts)*elemSize)
		for i, t := range ts {
			copy(data[i*elemSize:], t.(*tensor.Dense[float64]).Data())
		}
		return tensor.NewDense[float64](newShape, data)
	default:
		_ = f
		return nil
	}
}

// sliceTensorAt extracts element at index i along axis 0.
func sliceTensorAt(t tensor.Tensor, i int) tensor.Tensor {
	shape := t.Shape()
	if len(shape) == 0 {
		return t
	}

	elemShape := make(tensor.Shape, len(shape)-1)
	copy(elemShape, shape[1:])
	elemSize := 1
	for _, d := range elemShape {
		elemSize *= d
	}
	start := i * elemSize

	switch dt := t.(type) {
	case *tensor.Dense[float32]:
		return tensor.NewDense[float32](elemShape, dt.Data()[start:start+elemSize])
	case *tensor.Dense[int64]:
		return tensor.NewDense[int64](elemShape, dt.Data()[start:start+elemSize])
	case *tensor.Dense[int32]:
		return tensor.NewDense[int32](elemShape, dt.Data()[start:start+elemSize])
	case *tensor.Dense[uint8]:
		return tensor.NewDense[uint8](elemShape, dt.Data()[start:start+elemSize])
	case *tensor.Dense[float64]:
		return tensor.NewDense[float64](elemShape, dt.Data()[start:start+elemSize])
	default:
		return t
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
