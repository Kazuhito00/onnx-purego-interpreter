package reader

import (
	"fmt"
	"os"

	"google.golang.org/protobuf/proto"

	onnxpb "github.com/Kazuhito00/onnx-purego-interpreter/internal/onnxpb"
)

// DecodeBytes deserializes an ONNX model from raw protobuf bytes.
func DecodeBytes(data []byte) (*onnxpb.ModelProto, error) {
	model := &onnxpb.ModelProto{}
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("reader: failed to unmarshal ONNX model: %w", err)
	}
	return model, nil
}

// DecodeFile reads and deserializes an .onnx file.
func DecodeFile(path string) (*onnxpb.ModelProto, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reader: failed to read file %s: %w", path, err)
	}
	return DecodeBytes(data)
}
