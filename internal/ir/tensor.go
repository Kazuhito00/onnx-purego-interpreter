package ir

// TensorSpec describes a named tensor's type and shape (used for graph inputs/outputs).
type TensorSpec struct {
	Name  string
	DType DataType
	Shape Shape // may contain -1 for dynamic dims
}

// Initializer holds a constant tensor embedded in the model (weights, biases, etc.).
type Initializer struct {
	Name      string
	DType     DataType
	Shape     Shape
	RawData   []byte    // raw little-endian bytes
	FloatData []float32 // populated for float tensors
	Int32Data []int32
	Int64Data []int64
	DoubleData []float64
}
