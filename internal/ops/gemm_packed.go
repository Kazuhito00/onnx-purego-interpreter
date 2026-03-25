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
					aRow := A[i*K+k0:]
					cOff := i*N + j0
					bp := packed[panelOff:]
					for k := 0; k < kLen; k++ {
						aik := aRow[k]
						if aik == 0 {
							continue
						}
						bOff := k * nr
						for j := 0; j < jLen; j++ {
							C[cOff+j] += aik * bp[bOff+j]
						}
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

	a0 := A[i*lda+k0:]
	a1 := A[(i+1)*lda+k0:]
	a2 := A[(i+2)*lda+k0:]
	a3 := A[(i+3)*lda+k0:]

	for k := 0; k < kLen; k++ {
		bOff := k * nr
		a0k := a0[k]; a1k := a1[k]; a2k := a2[k]; a3k := a3[k]
		b0 := packedB[bOff]; b1 := packedB[bOff+1]; b2 := packedB[bOff+2]; b3 := packedB[bOff+3]
		b4 := packedB[bOff+4]; b5 := packedB[bOff+5]; b6 := packedB[bOff+6]; b7 := packedB[bOff+7]

		c00 += a0k*b0; c01 += a0k*b1; c02 += a0k*b2; c03 += a0k*b3
		c04 += a0k*b4; c05 += a0k*b5; c06 += a0k*b6; c07 += a0k*b7
		c10 += a1k*b0; c11 += a1k*b1; c12 += a1k*b2; c13 += a1k*b3
		c14 += a1k*b4; c15 += a1k*b5; c16 += a1k*b6; c17 += a1k*b7
		c20 += a2k*b0; c21 += a2k*b1; c22 += a2k*b2; c23 += a2k*b3
		c24 += a2k*b4; c25 += a2k*b5; c26 += a2k*b6; c27 += a2k*b7
		c30 += a3k*b0; c31 += a3k*b1; c32 += a3k*b2; c33 += a3k*b3
		c34 += a3k*b4; c35 += a3k*b5; c36 += a3k*b6; c37 += a3k*b7
	}

	o0 := i*N + j0
	if jLen >= 8 {
		C[o0] += c00; C[o0+1] += c01; C[o0+2] += c02; C[o0+3] += c03
		C[o0+4] += c04; C[o0+5] += c05; C[o0+6] += c06; C[o0+7] += c07
	} else {
		cs := [8]float32{c00, c01, c02, c03, c04, c05, c06, c07}
		for j := 0; j < jLen; j++ { C[o0+j] += cs[j] }
	}
	o1 := o0 + N
	if jLen >= 8 {
		C[o1] += c10; C[o1+1] += c11; C[o1+2] += c12; C[o1+3] += c13
		C[o1+4] += c14; C[o1+5] += c15; C[o1+6] += c16; C[o1+7] += c17
	} else {
		cs := [8]float32{c10, c11, c12, c13, c14, c15, c16, c17}
		for j := 0; j < jLen; j++ { C[o1+j] += cs[j] }
	}
	o2 := o1 + N
	if jLen >= 8 {
		C[o2] += c20; C[o2+1] += c21; C[o2+2] += c22; C[o2+3] += c23
		C[o2+4] += c24; C[o2+5] += c25; C[o2+6] += c26; C[o2+7] += c27
	} else {
		cs := [8]float32{c20, c21, c22, c23, c24, c25, c26, c27}
		for j := 0; j < jLen; j++ { C[o2+j] += cs[j] }
	}
	o3 := o2 + N
	if jLen >= 8 {
		C[o3] += c30; C[o3+1] += c31; C[o3+2] += c32; C[o3+3] += c33
		C[o3+4] += c34; C[o3+5] += c35; C[o3+6] += c36; C[o3+7] += c37
	} else {
		cs := [8]float32{c30, c31, c32, c33, c34, c35, c36, c37}
		for j := 0; j < jLen; j++ { C[o3+j] += cs[j] }
	}
}

// depthwiseF32 implements depthwise convolution directly without im2col.
