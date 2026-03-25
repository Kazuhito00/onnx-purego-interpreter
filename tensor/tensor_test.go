package tensor

import (
	"testing"
)

func TestDenseFloat32(t *testing.T) {
	d := NewDense[float32](Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if d.DType() != DTypeFloat32 {
		t.Errorf("expected float32, got %v", d.DType())
	}
	if !d.Shape().Equal(Shape{2, 3}) {
		t.Errorf("expected shape [2,3], got %v", d.Shape())
	}
	if d.Len() != 6 {
		t.Errorf("expected len 6, got %d", d.Len())
	}
	if d.At(0) != 1 || d.At(5) != 6 {
		t.Errorf("unexpected values")
	}
}

func TestDenseScalar(t *testing.T) {
	s := NewDenseScalar[float32](3.14)
	if s.Shape().NDim() != 0 {
		t.Errorf("scalar should have 0 dims, got %d", s.Shape().NDim())
	}
	if s.Len() != 1 {
		t.Errorf("scalar should have len 1, got %d", s.Len())
	}
	if s.At(0) != 3.14 {
		t.Errorf("unexpected value %f", s.At(0))
	}
}

func TestDenseClone(t *testing.T) {
	orig := NewDense[float32](Shape{2}, []float32{1, 2})
	clone := orig.Clone().(*Dense[float32])
	clone.Set(0, 99)
	if orig.At(0) != 1 {
		t.Errorf("clone should not affect original")
	}
}

func TestShapeSize(t *testing.T) {
	tests := []struct {
		shape Shape
		size  int
	}{
		{Shape{}, 1},
		{Shape{5}, 5},
		{Shape{2, 3}, 6},
		{Shape{2, 3, 4}, 24},
	}
	for _, tt := range tests {
		if tt.shape.Size() != tt.size {
			t.Errorf("Shape%v.Size() = %d, want %d", tt.shape, tt.shape.Size(), tt.size)
		}
	}
}

func TestBroadcastShape(t *testing.T) {
	tests := []struct {
		a, b Shape
		want Shape
		err  bool
	}{
		{Shape{3}, Shape{3}, Shape{3}, false},
		{Shape{1, 3}, Shape{3, 1}, Shape{3, 3}, false},
		{Shape{2, 1, 3}, Shape{1, 4, 3}, Shape{2, 4, 3}, false},
		{Shape{3}, Shape{2, 3}, Shape{2, 3}, false},
		{Shape{2}, Shape{3}, Shape{}, true},
	}
	for _, tt := range tests {
		got, err := BroadcastShape(tt.a, tt.b)
		if tt.err {
			if err == nil {
				t.Errorf("BroadcastShape(%v, %v) expected error", tt.a, tt.b)
			}
			continue
		}
		if err != nil {
			t.Errorf("BroadcastShape(%v, %v) unexpected error: %v", tt.a, tt.b, err)
			continue
		}
		if !got.Equal(tt.want) {
			t.Errorf("BroadcastShape(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestStrides(t *testing.T) {
	s := Strides(Shape{2, 3, 4})
	expected := []int{12, 4, 1}
	for i := range s {
		if s[i] != expected[i] {
			t.Errorf("stride[%d] = %d, want %d", i, s[i], expected[i])
		}
	}
}

func TestReshape(t *testing.T) {
	d := NewDense[float32](Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	r := d.Reshape(Shape{3, 2})
	if !r.Shape().Equal(Shape{3, 2}) {
		t.Errorf("unexpected shape %v", r.Shape())
	}
	// Reshape is a zero-copy view (shares data)
	if r.At(0) != 1 || r.At(5) != 6 {
		t.Errorf("reshape data mismatch")
	}
}

func TestReshapeCopy(t *testing.T) {
	d := NewDense[float32](Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	r := d.ReshapeCopy(Shape{3, 2})
	r.Set(0, 99)
	if d.At(0) != 1 {
		t.Errorf("ReshapeCopy should not affect original")
	}
}

func TestDenseInt64(t *testing.T) {
	d := NewDense[int64](Shape{3}, []int64{10, 20, 30})
	if d.DType() != DTypeInt64 {
		t.Errorf("expected int64, got %v", d.DType())
	}
	if d.At(1) != 20 {
		t.Errorf("expected 20, got %d", d.At(1))
	}
}
