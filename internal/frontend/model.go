package frontend

// DataType mirrors ONNX TensorProto.DataType values in the frontend model.
type DataType int32

const (
	DataTypeUndefined DataType = 0
	DataTypeFloat     DataType = 1
	DataTypeUint8     DataType = 2
	DataTypeInt8      DataType = 3
	DataTypeUint16    DataType = 4
	DataTypeInt16     DataType = 5
	DataTypeInt32     DataType = 6
	DataTypeInt64     DataType = 7
	DataTypeString    DataType = 8
	DataTypeBool      DataType = 9
	DataTypeFloat16   DataType = 10
	DataTypeDouble    DataType = 11
	DataTypeUint32    DataType = 12
	DataTypeUint64    DataType = 13
)

// Shape represents tensor dimensions. Negative values indicate dynamic dims.
type Shape []int64

// TensorSpec describes graph inputs/outputs.
type TensorSpec struct {
	Name  string
	DType DataType
	Shape Shape
}

// Initializer holds decoded constant tensor metadata and payload.
type Initializer struct {
	Name       string
	DType      DataType
	Shape      Shape
	RawData    []byte
	FloatData  []float32
	Int32Data  []int32
	Int64Data  []int64
	DoubleData []float64
}

// AttrValue is the frontend attribute representation.
type AttrValue interface {
	attrValue()
}

type AttrInt struct{ Value int64 }
type AttrFloat struct{ Value float32 }
type AttrString struct{ Value string }
type AttrInts struct{ Value []int64 }
type AttrFloats struct{ Value []float32 }
type AttrTensor struct{ Value *Initializer }
type AttrGraph struct{ Value *Graph }

func (AttrInt) attrValue()    {}
func (AttrFloat) attrValue()  {}
func (AttrString) attrValue() {}
func (AttrInts) attrValue()   {}
func (AttrFloats) attrValue() {}
func (AttrTensor) attrValue() {}
func (AttrGraph) attrValue()  {}

// OpsetImport records a domain and its opset version.
type OpsetImport struct {
	Domain  string
	Version int64
}

// Node is a protobuf-free ONNX-like graph node.
type Node struct {
	OpType  string
	Name    string
	Domain  string
	Inputs  []string
	Outputs []string
	Attrs   map[string]AttrValue
}

// Graph is the frontend graph representation.
type Graph struct {
	Name         string
	Nodes        []*Node
	Inputs       []TensorSpec
	Outputs      []TensorSpec
	Initializers map[string]*Initializer
}

// Model is the frontend model representation.
type Model struct {
	Graph  *Graph
	Opsets []OpsetImport
}
