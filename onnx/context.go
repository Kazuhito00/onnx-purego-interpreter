package onnx

import (
	"fmt"

	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// executionContext holds the tensor values during graph execution.
type executionContext struct {
	values map[string]tensor.Tensor
}

// newContext creates a new empty execution context.
func newContext() *executionContext {
	return &executionContext{values: make(map[string]tensor.Tensor)}
}

// Set stores a tensor value by name.
func (ctx *executionContext) Set(name string, t tensor.Tensor) {
	ctx.values[name] = t
}

// Get retrieves a tensor value by name. Returns nil if not found.
func (ctx *executionContext) Get(name string) tensor.Tensor {
	return ctx.values[name]
}

// MustGet retrieves a tensor value by name, returning an error if not found.
func (ctx *executionContext) MustGet(name string) (tensor.Tensor, error) {
	t, ok := ctx.values[name]
	if !ok {
		return nil, fmt.Errorf("engine: tensor %q not found in context", name)
	}
	return t, nil
}
