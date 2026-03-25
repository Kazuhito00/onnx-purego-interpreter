package ir

// DataType mirrors ONNX TensorProto.DataType values.
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

func (d DataType) String() string {
	switch d {
	case DataTypeFloat:
		return "float32"
	case DataTypeDouble:
		return "float64"
	case DataTypeInt32:
		return "int32"
	case DataTypeInt64:
		return "int64"
	case DataTypeUint8:
		return "uint8"
	case DataTypeInt8:
		return "int8"
	case DataTypeUint16:
		return "uint16"
	case DataTypeInt16:
		return "int16"
	case DataTypeBool:
		return "bool"
	case DataTypeUint32:
		return "uint32"
	case DataTypeUint64:
		return "uint64"
	default:
		return "undefined"
	}
}

// Shape represents tensor dimensions. A negative value indicates a dynamic/symbolic dimension.
type Shape []int64
