package tensor

import "fmt"

// Shape represents the dimensions of a tensor.
// テンソルの次元を表す。
type Shape []int

// Size returns the total number of elements.
// 全要素数を返す。
func (s Shape) Size() int {
	if len(s) == 0 {
		return 1 // scalar
	}
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

// NDim returns the number of dimensions.
// 次元数を返す。
func (s Shape) NDim() int {
	return len(s)
}

// Clone returns a copy of the shape.
// shape のコピーを返す。
func (s Shape) Clone() Shape {
	c := make(Shape, len(s))
	copy(c, s)
	return c
}

// Equal checks shape equality.
// shape の等値判定を行う。
func (s Shape) Equal(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i := range s {
		if s[i] != other[i] {
			return false
		}
	}
	return true
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int(s))
}

// BroadcastShape computes the broadcast-compatible output shape for two input shapes.
// Returns error if shapes are not broadcast-compatible.
// 2 つの入力 shape からブロードキャスト互換の出力 shape を計算する。
// ブロードキャスト非互換の場合はエラーを返す。
func BroadcastShape(a, b Shape) (Shape, error) {
	na, nb := len(a), len(b)
	ndim := na
	if nb > ndim {
		ndim = nb
	}
	result := make(Shape, ndim)
	for i := 0; i < ndim; i++ {
		da := 1
		if i < na {
			da = a[na-1-i]
		}
		db := 1
		if i < nb {
			db = b[nb-1-i]
		}
		if da == db {
			result[ndim-1-i] = da
		} else if da == 1 {
			result[ndim-1-i] = db
		} else if db == 1 {
			result[ndim-1-i] = da
		} else {
			return nil, fmt.Errorf("tensor: shapes %v and %v are not broadcast-compatible", a, b)
		}
	}
	return result, nil
}

// Strides computes row-major strides for a given shape.
// 指定した shape の行優先ストライドを計算する。
func Strides(s Shape) []int {
	if len(s) == 0 {
		return nil
	}
	strides := make([]int, len(s))
	strides[len(s)-1] = 1
	for i := len(s) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * s[i+1]
	}
	return strides
}

// BroadcastIndex maps a flat output index to the corresponding flat input index,
// accounting for broadcasting (dimensions of size 1 are broadcast).
// フラット出力インデックスを対応するフラット入力インデックスにマッピングする。
// サイズ 1 の次元はブロードキャストされる。
func BroadcastIndex(flatIdx int, outShape, inShape Shape, outStrides, inStrides []int) int {
	idx := 0
	ndimOut := len(outShape)
	ndimIn := len(inShape)
	for i := 0; i < ndimOut; i++ {
		coord := (flatIdx / outStrides[i]) % outShape[i]
		inDimIdx := i - (ndimOut - ndimIn)
		if inDimIdx >= 0 && inDimIdx < ndimIn {
			if inShape[inDimIdx] != 1 {
				idx += coord * inStrides[inDimIdx]
			}
		}
	}
	return idx
}
