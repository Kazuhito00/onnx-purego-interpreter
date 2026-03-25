package tensor

// Numeric is the constraint for types that can be stored in Dense tensors.
// Dense テンソルに格納可能な型の制約。
type Numeric interface {
	~float32 | ~float64 | ~int32 | ~int64 | ~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~uint64
}

// DType identifies the element data type of a tensor.
// テンソル要素のデータ型を識別する。
type DType int32

const (
	DTypeUndefined DType = 0
	DTypeFloat32   DType = 1
	DTypeUint8     DType = 2
	DTypeInt8      DType = 3
	DTypeUint16    DType = 4
	DTypeInt16     DType = 5
	DTypeInt32     DType = 6
	DTypeInt64     DType = 7
	DTypeBool      DType = 9
	DTypeFloat64   DType = 11
	DTypeUint32    DType = 12
	DTypeUint64    DType = 13
)

func (d DType) String() string {
	switch d {
	case DTypeFloat32:
		return "float32"
	case DTypeFloat64:
		return "float64"
	case DTypeInt32:
		return "int32"
	case DTypeInt64:
		return "int64"
	case DTypeUint8:
		return "uint8"
	case DTypeInt8:
		return "int8"
	case DTypeUint16:
		return "uint16"
	case DTypeInt16:
		return "int16"
	case DTypeBool:
		return "bool"
	case DTypeUint32:
		return "uint32"
	case DTypeUint64:
		return "uint64"
	default:
		return "undefined"
	}
}

// DTypeOf returns the DType for a Go numeric type.
// Go の数値型に対応する DType を返す。
func DTypeOf[T Numeric](v T) DType {
	switch any(v).(type) {
	case float32:
		return DTypeFloat32
	case float64:
		return DTypeFloat64
	case int32:
		return DTypeInt32
	case int64:
		return DTypeInt64
	case uint8:
		return DTypeUint8
	case int8:
		return DTypeInt8
	case uint16:
		return DTypeUint16
	case int16:
		return DTypeInt16
	case uint32:
		return DTypeUint32
	case uint64:
		return DTypeUint64
	default:
		return DTypeUndefined
	}
}
