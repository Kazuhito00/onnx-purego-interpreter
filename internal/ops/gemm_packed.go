package ops

import "github.com/Kazuhito00/onnx-purego-interpreter/tensor"

// Ensure tensor import is used
var _ tensor.Shape

// PackedB holds a pre-packed B matrix for GEMM-friendly access.
// Layout: for each (k-panel, n-panel), store kc×nr contiguous block.
type PackedB struct {
	Data []float32
	K, N int
}

// PackMatMulB packs B[K,N] into panel layout for efficient GEMM.
func PackMatMulB(B []float32, K, N int) *PackedB {
	// Compute packed size: round up to panel boundaries
	nPanels := (N + nr - 1) / nr
	kPanels := (K + kc - 1) / kc
	packedSize := nPanels * kPanels * kc * nr
	packed := make([]float32, packedSize)

	off := 0
	for j0 := 0; j0 < N; j0 += nr {
		jEnd := j0 + nr
		if jEnd > N {
			jEnd = N
		}
		jLen := jEnd - j0
		for k0 := 0; k0 < K; k0 += kc {
			kEnd := k0 + kc
			if kEnd > K {
				kEnd = K
			}
			for k := k0; k < kEnd; k++ {
				for j := 0; j < jLen; j++ {
					packed[off] = B[k*N+j0+j]
					off++
				}
				// Zero-pad to nr
				for j := jLen; j < nr; j++ {
					packed[off] = 0
					off++
				}
			}
			// Zero-pad remaining k rows in this panel
			for k := kEnd; k < k0+kc; k++ {
				for j := 0; j < nr; j++ {
					packed[off] = 0
					off++
				}
			}
		}
	}
	return &PackedB{Data: packed, K: K, N: N}
}

// gemmF32Packed computes C += A * packedB where A[M,K], C[M,N].
func gemmF32Packed(A []float32, pb *PackedB, C []float32, M int) {
	K, N := pb.K, pb.N
	packed := pb.Data
	nPanels := (N + nr - 1) / nr

	for i0 := 0; i0 < M; i0 += mc {
		iEnd := min(i0+mc, M)

		panelOff := 0
		for jp := 0; jp < nPanels; jp++ {
			j0 := jp * nr
			jEnd := min(j0+nr, N)
			jLen := jEnd - j0

			for k0 := 0; k0 < K; k0 += kc {
				kEnd := min(k0+kc, K)
				kLen := kEnd - k0

				// Process micro-tiles using packed B panel
				i := i0
				for ; i+3 < iEnd; i += 4 {
					microKernelPacked4x8(A, packed[panelOff:], C, i, j0, k0, kLen, K, N, jLen)
				}
				for ; i < iEnd; i++ {
					aRow := A[i*K+k0 : i*K+k0+kLen : i*K+k0+kLen] // BCE
					cSlice := C[i*N+j0 : i*N+j0+jLen : i*N+j0+jLen] // BCE
					bp := packed[panelOff:]
					bIdx := 0
					for k := 0; k < kLen; k++ {
						aik := aRow[k]
						if aik == 0 {
							bIdx += nr
							continue
						}
						bSlice := bp[bIdx : bIdx+jLen : bIdx+jLen] // BCE
						for j := 0; j < jLen; j++ {
							cSlice[j] += aik * bSlice[j]
						}
						bIdx += nr
					}
				}

				panelOff += kc * nr
			}
		}
	}
}

// microKernelPacked4x8 uses packed B layout for 4x8 block.
func microKernelPacked4x8(A []float32, packedB []float32, C []float32, i, j0, k0, kLen, lda, N, jLen int) {
	var c00, c01, c02, c03, c04, c05, c06, c07 float32
	var c10, c11, c12, c13, c14, c15, c16, c17 float32
	var c20, c21, c22, c23, c24, c25, c26, c27 float32
	var c30, c31, c32, c33, c34, c35, c36, c37 float32

	a0 := A[i*lda+k0 : i*lda+k0+kLen : i*lda+k0+kLen]             // BCE
	a1 := A[(i+1)*lda+k0 : (i+1)*lda+k0+kLen : (i+1)*lda+k0+kLen] // BCE
	a2 := A[(i+2)*lda+k0 : (i+2)*lda+k0+kLen : (i+2)*lda+k0+kLen] // BCE
	a3 := A[(i+3)*lda+k0 : (i+3)*lda+k0+kLen : (i+3)*lda+k0+kLen] // BCE

	bIdx := 0
	for k := 0; k < kLen; k++ {
		a0k := a0[k]; a1k := a1[k]; a2k := a2[k]; a3k := a3[k]
		bSlice := packedB[bIdx : bIdx+8 : bIdx+8] // BCE
		b0 := bSlice[0]; b1 := bSlice[1]; b2 := bSlice[2]; b3 := bSlice[3]
		b4 := bSlice[4]; b5 := bSlice[5]; b6 := bSlice[6]; b7 := bSlice[7]

		c00 += a0k*b0; c01 += a0k*b1; c02 += a0k*b2; c03 += a0k*b3
		c04 += a0k*b4; c05 += a0k*b5; c06 += a0k*b6; c07 += a0k*b7
		c10 += a1k*b0; c11 += a1k*b1; c12 += a1k*b2; c13 += a1k*b3
		c14 += a1k*b4; c15 += a1k*b5; c16 += a1k*b6; c17 += a1k*b7
		c20 += a2k*b0; c21 += a2k*b1; c22 += a2k*b2; c23 += a2k*b3
		c24 += a2k*b4; c25 += a2k*b5; c26 += a2k*b6; c27 += a2k*b7
		c30 += a3k*b0; c31 += a3k*b1; c32 += a3k*b2; c33 += a3k*b3
		c34 += a3k*b4; c35 += a3k*b5; c36 += a3k*b6; c37 += a3k*b7
		bIdx += nr
	}

	o0 := i*N + j0
	if jLen >= 8 {
		cr0 := C[o0 : o0+8 : o0+8] // BCE
		cr0[0] += c00; cr0[1] += c01; cr0[2] += c02; cr0[3] += c03
		cr0[4] += c04; cr0[5] += c05; cr0[6] += c06; cr0[7] += c07
	} else {
		cs := [8]float32{c00, c01, c02, c03, c04, c05, c06, c07}
		for j := 0; j < jLen; j++ { C[o0+j] += cs[j] }
	}
	o1 := o0 + N
	if jLen >= 8 {
		cr1 := C[o1 : o1+8 : o1+8] // BCE
		cr1[0] += c10; cr1[1] += c11; cr1[2] += c12; cr1[3] += c13
		cr1[4] += c14; cr1[5] += c15; cr1[6] += c16; cr1[7] += c17
	} else {
		cs := [8]float32{c10, c11, c12, c13, c14, c15, c16, c17}
		for j := 0; j < jLen; j++ { C[o1+j] += cs[j] }
	}
	o2 := o1 + N
	if jLen >= 8 {
		cr2 := C[o2 : o2+8 : o2+8] // BCE
		cr2[0] += c20; cr2[1] += c21; cr2[2] += c22; cr2[3] += c23
		cr2[4] += c24; cr2[5] += c25; cr2[6] += c26; cr2[7] += c27
	} else {
		cs := [8]float32{c20, c21, c22, c23, c24, c25, c26, c27}
		for j := 0; j < jLen; j++ { C[o2+j] += cs[j] }
	}
	o3 := o2 + N
	if jLen >= 8 {
		cr3 := C[o3 : o3+8 : o3+8] // BCE
		cr3[0] += c30; cr3[1] += c31; cr3[2] += c32; cr3[3] += c33
		cr3[4] += c34; cr3[5] += c35; cr3[6] += c36; cr3[7] += c37
	} else {
		cs := [8]float32{c30, c31, c32, c33, c34, c35, c36, c37}
		for j := 0; j < jLen; j++ { C[o3+j] += cs[j] }
	}
}

// depthwiseF32 implements depthwise convolution directly without im2col.
