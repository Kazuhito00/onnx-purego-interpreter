package frontend

import (
	"fmt"

	onnxpb "github.com/Kazuhito00/onnx-purego-interpreter/internal/onnxpb"
)

// Build converts decoded ONNX protobuf structures into the frontend model.
func Build(model *onnxpb.ModelProto) (*Model, error) {
	if model.GetGraph() == nil {
		return nil, fmt.Errorf("frontend: model has no graph")
	}

	g := convertGraph(model.GetGraph())
	m := &Model{Graph: g}
	for _, oi := range model.GetOpsetImport() {
		m.Opsets = append(m.Opsets, OpsetImport{
			Domain:  oi.GetDomain(),
			Version: oi.GetVersion(),
		})
	}
	return m, nil
}

func convertGraph(pbGraph *onnxpb.GraphProto) *Graph {
	g := &Graph{
		Name:         pbGraph.GetName(),
		Initializers: make(map[string]*Initializer),
	}

	for _, vi := range pbGraph.GetInput() {
		g.Inputs = append(g.Inputs, convertValueInfo(vi))
	}
	for _, vi := range pbGraph.GetOutput() {
		g.Outputs = append(g.Outputs, convertValueInfo(vi))
	}
	for _, tp := range pbGraph.GetInitializer() {
		init := convertTensorProto(tp)
		g.Initializers[init.Name] = init
	}
	for _, pbn := range pbGraph.GetNode() {
		node := &Node{
			OpType:  pbn.GetOpType(),
			Name:    pbn.GetName(),
			Domain:  pbn.GetDomain(),
			Inputs:  append([]string(nil), pbn.GetInput()...),
			Outputs: append([]string(nil), pbn.GetOutput()...),
			Attrs:   make(map[string]AttrValue),
		}
		for _, attr := range pbn.GetAttribute() {
			node.Attrs[attr.GetName()] = convertAttr(attr)
		}
		g.Nodes = append(g.Nodes, node)
	}
	return g
}

func convertValueInfo(vi *onnxpb.ValueInfoProto) TensorSpec {
	spec := TensorSpec{Name: vi.GetName()}
	if vi.GetType() != nil {
		tt := vi.GetType().GetTensorType()
		if tt != nil {
			spec.DType = DataType(tt.GetElemType())
			if tt.GetShape() != nil {
				for _, dim := range tt.GetShape().GetDim() {
					if dim.GetDimValue() > 0 {
						spec.Shape = append(spec.Shape, dim.GetDimValue())
					} else {
						spec.Shape = append(spec.Shape, -1)
					}
				}
			}
		}
	}
	return spec
}

func convertTensorProto(tp *onnxpb.TensorProto) *Initializer {
	init := &Initializer{
		Name:       tp.GetName(),
		DType:      DataType(tp.GetDataType()),
		RawData:    tp.GetRawData(),
		FloatData:  tp.GetFloatData(),
		Int32Data:  tp.GetInt32Data(),
		Int64Data:  tp.GetInt64Data(),
		DoubleData: tp.GetDoubleData(),
	}
	for _, d := range tp.GetDims() {
		init.Shape = append(init.Shape, d)
	}
	return init
}

func convertAttr(attr *onnxpb.AttributeProto) AttrValue {
	switch attr.GetType() {
	case onnxpb.AttributeProto_INT:
		return AttrInt{Value: attr.GetI()}
	case onnxpb.AttributeProto_FLOAT:
		return AttrFloat{Value: attr.GetF()}
	case onnxpb.AttributeProto_STRING:
		return AttrString{Value: string(attr.GetS())}
	case onnxpb.AttributeProto_INTS:
		return AttrInts{Value: attr.GetInts()}
	case onnxpb.AttributeProto_FLOATS:
		return AttrFloats{Value: attr.GetFloats()}
	case onnxpb.AttributeProto_TENSOR:
		if tp := attr.GetT(); tp != nil {
			return AttrTensor{Value: convertTensorProto(tp)}
		}
		return AttrTensor{}
	case onnxpb.AttributeProto_GRAPH:
		if g := attr.GetG(); g != nil {
			return AttrGraph{Value: convertGraph(g)}
		}
		return AttrGraph{}
	default:
		if attr.GetI() != 0 {
			return AttrInt{Value: attr.GetI()}
		}
		if attr.GetF() != 0 {
			return AttrFloat{Value: attr.GetF()}
		}
		return AttrString{Value: string(attr.GetS())}
	}
}
