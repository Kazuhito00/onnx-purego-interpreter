package ir

import "github.com/Kazuhito00/onnx-purego-interpreter/internal/frontend"

// FromFrontend lowers the protobuf-free frontend model into the canonical IR.
func FromFrontend(m *frontend.Model) (*Graph, error) {
	g := &Graph{
		Name:         m.Graph.Name,
		Initializers: make(map[string]*Initializer, len(m.Graph.Initializers)),
	}

	for _, oi := range m.Opsets {
		g.Opsets = append(g.Opsets, OpsetImport{
			Domain:  oi.Domain,
			Version: oi.Version,
		})
	}
	for _, inp := range m.Graph.Inputs {
		g.Inputs = append(g.Inputs, TensorSpec{
			Name:  inp.Name,
			DType: DataType(inp.DType),
			Shape: Shape(inp.Shape),
		})
	}
	for _, out := range m.Graph.Outputs {
		g.Outputs = append(g.Outputs, TensorSpec{
			Name:  out.Name,
			DType: DataType(out.DType),
			Shape: Shape(out.Shape),
		})
	}
	for name, init := range m.Graph.Initializers {
		g.Initializers[name] = convertInitializer(init)
	}
	for _, node := range m.Graph.Nodes {
		cn := &Node{
			OpType:       node.OpType,
			Name:         node.Name,
			Domain:       node.Domain,
			OpsetVersion: g.OpsetVersion(node.Domain),
			Inputs:       append([]string(nil), node.Inputs...),
			Outputs:      append([]string(nil), node.Outputs...),
			Attrs:        make(map[string]AttrValue, len(node.Attrs)),
		}
		for name, attr := range node.Attrs {
			cn.Attrs[name] = convertAttrValue(attr)
		}
		g.Nodes = append(g.Nodes, cn)
	}

	return g, nil
}

func convertInitializer(init *frontend.Initializer) *Initializer {
	return &Initializer{
		Name:       init.Name,
		DType:      DataType(init.DType),
		Shape:      Shape(init.Shape),
		RawData:    append([]byte(nil), init.RawData...),
		FloatData:  append([]float32(nil), init.FloatData...),
		Int32Data:  append([]int32(nil), init.Int32Data...),
		Int64Data:  append([]int64(nil), init.Int64Data...),
		DoubleData: append([]float64(nil), init.DoubleData...),
	}
}

func convertAttrValue(attr frontend.AttrValue) AttrValue {
	switch a := attr.(type) {
	case frontend.AttrInt:
		return AttrInt{Value: a.Value}
	case frontend.AttrFloat:
		return AttrFloat{Value: a.Value}
	case frontend.AttrString:
		return AttrString{Value: a.Value}
	case frontend.AttrInts:
		return AttrInts{Value: append([]int64(nil), a.Value...)}
	case frontend.AttrFloats:
		return AttrFloats{Value: append([]float32(nil), a.Value...)}
	case frontend.AttrTensor:
		if a.Value == nil {
			return AttrTensor{}
		}
		return AttrTensor{Value: convertInitializer(a.Value)}
	case frontend.AttrGraph:
		if a.Value == nil {
			return &AttrGraph{}
		}
		sub := &frontend.Model{Graph: a.Value}
		subGraph, _ := FromFrontend(sub)
		return &AttrGraph{Value: subGraph}
	default:
		return AttrString{}
	}
}
