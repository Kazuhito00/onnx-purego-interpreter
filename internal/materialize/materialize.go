package materialize

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

func elementCount(shape ir.Shape) int {
	if len(shape) == 0 {
		return 1
	}
	n := 1
	for _, d := range shape {
		n *= int(d)
	}
	return n
}

// Float32 extracts float32 data from an initializer.
func Float32(init *ir.Initializer) ([]float32, error) {
	if len(init.FloatData) > 0 {
		return init.FloatData, nil
	}
	if len(init.RawData) > 0 {
		n := len(init.RawData) / 4
		data := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(init.RawData[i*4:])
			data[i] = math.Float32frombits(bits)
		}
		return data, nil
	}
	if elementCount(init.Shape) == 0 {
		return []float32{}, nil
	}
	return nil, fmt.Errorf("materialize: no float32 data in initializer %s", init.Name)
}

// Int64 extracts int64 data from an initializer.
func Int64(init *ir.Initializer) ([]int64, error) {
	if len(init.Int64Data) > 0 {
		return init.Int64Data, nil
	}
	if len(init.RawData) > 0 {
		n := len(init.RawData) / 8
		data := make([]int64, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint64(init.RawData[i*8:])
			data[i] = int64(bits)
		}
		return data, nil
	}
	if elementCount(init.Shape) == 0 {
		return []int64{}, nil
	}
	return nil, fmt.Errorf("materialize: no int64 data in initializer %s", init.Name)
}

// Float64 extracts float64 data from an initializer.
func Float64(init *ir.Initializer) ([]float64, error) {
	if len(init.DoubleData) > 0 {
		return init.DoubleData, nil
	}
	if len(init.RawData) > 0 {
		n := len(init.RawData) / 8
		data := make([]float64, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint64(init.RawData[i*8:])
			data[i] = math.Float64frombits(bits)
		}
		return data, nil
	}
	if elementCount(init.Shape) == 0 {
		return []float64{}, nil
	}
	return nil, fmt.Errorf("materialize: no float64 data in initializer %s", init.Name)
}

// Int32 extracts int32 data from an initializer.
func Int32(init *ir.Initializer) ([]int32, error) {
	if len(init.Int32Data) > 0 {
		return init.Int32Data, nil
	}
	if len(init.RawData) > 0 {
		n := len(init.RawData) / 4
		data := make([]int32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(init.RawData[i*4:])
			data[i] = int32(bits)
		}
		return data, nil
	}
	if elementCount(init.Shape) == 0 {
		return []int32{}, nil
	}
	return nil, fmt.Errorf("materialize: no int32 data in initializer %s", init.Name)
}

// Tensor converts an initializer to a runtime tensor.
func Tensor(init *ir.Initializer) (tensor.Tensor, error) {
	shape := make(tensor.Shape, len(init.Shape))
	for i, d := range init.Shape {
		shape[i] = int(d)
	}

	switch init.DType {
	case ir.DataTypeFloat:
		data, err := Float32(init)
		if err != nil {
			return nil, err
		}
		return tensor.NewDense[float32](shape, data), nil
	case ir.DataTypeDouble:
		data, err := Float64(init)
		if err != nil {
			return nil, err
		}
		return tensor.NewDense[float64](shape, data), nil
	case ir.DataTypeInt64:
		data, err := Int64(init)
		if err != nil {
			return nil, err
		}
		return tensor.NewDense[int64](shape, data), nil
	case ir.DataTypeInt32:
		data, err := Int32(init)
		if err != nil {
			return nil, err
		}
		return tensor.NewDense[int32](shape, data), nil
	case ir.DataTypeUint8:
		if len(init.RawData) > 0 {
			data := make([]uint8, len(init.RawData))
			copy(data, init.RawData)
			return tensor.NewDense[uint8](shape, data), nil
		}
		if elementCount(init.Shape) == 0 {
			return tensor.NewDense[uint8](shape, []uint8{}), nil
		}
		return nil, fmt.Errorf("materialize: no uint8 data in initializer %s", init.Name)
	case ir.DataTypeInt8:
		if len(init.RawData) > 0 {
			n := len(init.RawData)
			data := make([]int8, n)
			for i := 0; i < n; i++ {
				data[i] = int8(init.RawData[i])
			}
			return tensor.NewDense[int8](shape, data), nil
		}
		if elementCount(init.Shape) == 0 {
			return tensor.NewDense[int8](shape, []int8{}), nil
		}
		return nil, fmt.Errorf("materialize: no int8 data in initializer %s", init.Name)
	case ir.DataTypeUint16:
		if len(init.RawData) > 0 {
			n := len(init.RawData) / 2
			data := make([]uint16, n)
			for i := 0; i < n; i++ {
				data[i] = binary.LittleEndian.Uint16(init.RawData[i*2:])
			}
			return tensor.NewDense[uint16](shape, data), nil
		}
		if elementCount(init.Shape) == 0 {
			return tensor.NewDense[uint16](shape, []uint16{}), nil
		}
		return nil, fmt.Errorf("materialize: no uint16 data in initializer %s", init.Name)
	case ir.DataTypeInt16:
		if len(init.RawData) > 0 {
			n := len(init.RawData) / 2
			data := make([]int16, n)
			for i := 0; i < n; i++ {
				data[i] = int16(binary.LittleEndian.Uint16(init.RawData[i*2:]))
			}
			return tensor.NewDense[int16](shape, data), nil
		}
		if elementCount(init.Shape) == 0 {
			return tensor.NewDense[int16](shape, []int16{}), nil
		}
		return nil, fmt.Errorf("materialize: no int16 data in initializer %s", init.Name)
	case ir.DataTypeUint32:
		if len(init.RawData) > 0 {
			n := len(init.RawData) / 4
			data := make([]uint32, n)
			for i := 0; i < n; i++ {
				data[i] = binary.LittleEndian.Uint32(init.RawData[i*4:])
			}
			return tensor.NewDense[uint32](shape, data), nil
		}
		if elementCount(init.Shape) == 0 {
			return tensor.NewDense[uint32](shape, []uint32{}), nil
		}
		return nil, fmt.Errorf("materialize: no uint32 data in initializer %s", init.Name)
	case ir.DataTypeUint64:
		if len(init.RawData) > 0 {
			n := len(init.RawData) / 8
			data := make([]uint64, n)
			for i := 0; i < n; i++ {
				data[i] = binary.LittleEndian.Uint64(init.RawData[i*8:])
			}
			return tensor.NewDense[uint64](shape, data), nil
		}
		if elementCount(init.Shape) == 0 {
			return tensor.NewDense[uint64](shape, []uint64{}), nil
		}
		return nil, fmt.Errorf("materialize: no uint64 data in initializer %s", init.Name)
	case ir.DataTypeBool:
		if len(init.RawData) > 0 {
			data := make([]uint8, len(init.RawData))
			copy(data, init.RawData)
			return tensor.NewDense[uint8](shape, data), nil
		}
		if len(init.Int32Data) > 0 {
			data := make([]uint8, len(init.Int32Data))
			for i, v := range init.Int32Data {
				if v != 0 {
					data[i] = 1
				}
			}
			return tensor.NewDense[uint8](shape, data), nil
		}
		if elementCount(init.Shape) == 0 {
			return tensor.NewDense[uint8](shape, []uint8{}), nil
		}
		return nil, fmt.Errorf("materialize: no bool data in initializer %s", init.Name)
	default:
		return nil, fmt.Errorf("materialize: unsupported initializer dtype %v for %s", init.DType, init.Name)
	}
}
