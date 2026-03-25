// Package tensor provides the core tensor types used throughout the ONNX runtime.
// ONNX ランタイム全体で使用されるテンソル型を提供する。
//
// The primary types are:
// 主要な型:
//   - [Tensor] — type-erased interface for all tensors / 全テンソルの型消去インターフェース
//   - [Dense] — generic row-major dense tensor / ジェネリック行優先密テンソル (Dense[float32], Dense[int64], etc.)
//   - [Shape] — dimension sizes with broadcasting support / ブロードキャスト対応の次元サイズ
package tensor

import (
	"fmt"
	"unsafe"
)

// Tensor is the type-erased interface for runtime tensors.
// ランタイムテンソルの型消去インターフェース。
type Tensor interface {
	DType() DType
	Shape() Shape
	Len() int
	Clone() Tensor
	DataPtr() unsafe.Pointer
}

// Dense is a generic dense tensor with row-major flat storage.
// 行優先フラット配列によるジェネリック密テンソル。
type Dense[T Numeric] struct {
	shape Shape
	data  []T
}

// NewDense creates a new Dense tensor with the given shape and data.
// data is NOT copied — caller must not mutate it after passing.
// 指定した shape と data で新しい Dense テンソルを作成する。
// data はコピーされない — 渡した後に変更してはならない。
func NewDense[T Numeric](shape Shape, data []T) *Dense[T] {
	expected := shape.Size()
	if len(data) != expected {
		panic(fmt.Sprintf("tensor: data length %d does not match shape %v (expected %d)", len(data), shape, expected))
	}
	return &Dense[T]{shape: shape.Clone(), data: data}
}

// NewDenseZeros creates a zero-filled Dense tensor.
// ゼロ埋めされた Dense テンソルを作成する。
func NewDenseZeros[T Numeric](shape Shape) *Dense[T] {
	return &Dense[T]{shape: shape.Clone(), data: make([]T, shape.Size())}
}

// NewDenseScalar creates a scalar (0-D) tensor.
// スカラー (0 次元) テンソルを作成する。
func NewDenseScalar[T Numeric](v T) *Dense[T] {
	return &Dense[T]{shape: Shape{}, data: []T{v}}
}

// DType returns the ONNX data type of the tensor elements.
// テンソル要素の ONNX データ型を返す。
func (d *Dense[T]) DType() DType {
	var zero T
	return DTypeOf(zero)
}

// Shape returns the dimension sizes.
// 次元サイズを返す。
func (d *Dense[T]) Shape() Shape { return d.shape }

// Len returns the total number of elements.
// 全要素数を返す。
func (d *Dense[T]) Len() int { return len(d.data) }

// Data returns the underlying flat data slice.
// 内部のフラットデータスライスを返す。
func (d *Dense[T]) Data() []T { return d.data }

// Clone returns a deep copy of the tensor.
// テンソルのディープコピーを返す。
func (d *Dense[T]) Clone() Tensor {
	newData := make([]T, len(d.data))
	copy(newData, d.data)
	return &Dense[T]{shape: d.shape.Clone(), data: newData}
}

// DataPtr returns an unsafe pointer to the first element, or nil if empty.
func (d *Dense[T]) DataPtr() unsafe.Pointer {
	if len(d.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&d.data[0])
}

// At returns the element at the given flat index.
func (d *Dense[T]) At(i int) T { return d.data[i] }

// Set sets the element at the given flat index.
func (d *Dense[T]) Set(i int, v T) { d.data[i] = v }

// Reshape returns a view with the new shape sharing the same underlying data.
// Zero-copy for performance. The caller must not mutate the returned tensor
// if the original is still in use.
// 同じデータを共有する新しい shape のビューを返す (zero-copy)。
// 元のテンソルが使用中の場合、返されたテンソルを変更してはならない。
func (d *Dense[T]) Reshape(newShape Shape) *Dense[T] {
	if newShape.Size() != d.Len() {
		panic(fmt.Sprintf("tensor: cannot reshape %v to %v", d.shape, newShape))
	}
	return &Dense[T]{shape: newShape.Clone(), data: d.data}
}

// ReshapeCopy returns a new tensor with copied data and the new shape.
// データをコピーした新しい shape のテンソルを返す。
func (d *Dense[T]) ReshapeCopy(newShape Shape) *Dense[T] {
	if newShape.Size() != d.Len() {
		panic(fmt.Sprintf("tensor: cannot reshape %v to %v", d.shape, newShape))
	}
	newData := make([]T, len(d.data))
	copy(newData, d.data)
	return &Dense[T]{shape: newShape.Clone(), data: newData}
}

// String returns a summary of the tensor.
func (d *Dense[T]) String() string {
	return fmt.Sprintf("Dense[%s](%v, len=%d)", d.DType(), d.shape, d.Len())
}
