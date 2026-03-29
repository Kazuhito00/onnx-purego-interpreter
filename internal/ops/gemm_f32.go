package ops

// Specialized float32 GEMM kernels with microkernel and tiling.
// Pure Go, no CGo, no assembly.

const (
	mc = 128
	nc = 192
	kc = 128
	mr = 4
	nr = 8
)

func gemmF32(A, B, C []float32, M, N, K int) {
	if M*K+K*N > 32*1024 {
		gemmF32Tiled(A, B, C, M, N, K)
	} else {
		gemmF32Simple(A, B, C, M, N, K)
	}
}

func gemmF32Simple(A, B, C []float32, M, N, K int) {
	for i := 0; i < M; i++ {
		cRow := C[i*N : i*N+N : i*N+N] // BCE
		aRow := A[i*K : i*K+K : i*K+K] // BCE
		for k := 0; k < K; k++ {
			aik := aRow[k]
			if aik == 0 {
				continue
			}
			bRow := B[k*N : k*N+N : k*N+N] // BCE
			j := 0
			for ; j+3 < N; j += 4 {
				cRow[j] += aik * bRow[j]
				cRow[j+1] += aik * bRow[j+1]
				cRow[j+2] += aik * bRow[j+2]
				cRow[j+3] += aik * bRow[j+3]
			}
			for ; j < N; j++ {
				cRow[j] += aik * bRow[j]
			}
		}
	}
}

func gemmF32Tiled(A, B, C []float32, M, N, K int) {
	for j0 := 0; j0 < N; j0 += nc {
		jEnd := min(j0+nc, N)
		for k0 := 0; k0 < K; k0 += kc {
			kEnd := min(k0+kc, K)
			kLen := kEnd - k0
			for i0 := 0; i0 < M; i0 += mc {
				iE := min(i0+mc, M)
				i := i0
				for ; i+3 < iE; i += 4 {
					j := j0
					for ; j+7 < jEnd; j += 8 {
						microKernel4x8(A, B, C, i, j, k0, kLen, K, N)
					}
					for ; j+3 < jEnd; j += 4 {
						microKernel4x4(A, B, C, i, j, k0, kLen, K, N)
					}
					if j < jEnd {
						microKernelMxN(A, B, C, i, j, k0, kLen, K, N, 4, jEnd-j)
					}
				}
				rem := iE - i
				if rem >= 2 {
					j := j0
					for ; j+7 < jEnd; j += 8 {
						microKernel2x8(A, B, C, i, j, k0, kLen, K, N)
					}
					if j < jEnd {
						microKernelMxN(A, B, C, i, j, k0, kLen, K, N, 2, jEnd-j)
					}
					i += 2
					rem -= 2
				}
				if rem >= 1 {
					j := j0
					for ; j+7 < jEnd; j += 8 {
						microKernel1x8(A, B, C, i, j, k0, kLen, K, N)
					}
					if j < jEnd {
						microKernelMxN(A, B, C, i, j, k0, kLen, K, N, 1, jEnd-j)
					}
				}
			}
		}
	}
}

// microKernel4x8: 32 accumulators, k-unrolled by 4
func microKernel4x8(A, B, C []float32, i, j, k0, kLen, lda, ldb int) {
	var c00, c01, c02, c03, c04, c05, c06, c07 float32
	var c10, c11, c12, c13, c14, c15, c16, c17 float32
	var c20, c21, c22, c23, c24, c25, c26, c27 float32
	var c30, c31, c32, c33, c34, c35, c36, c37 float32

	// BCE: exact-length slices eliminate bounds checks in hot loop
	a0 := A[i*lda+k0 : i*lda+k0+kLen : i*lda+k0+kLen]
	a1 := A[(i+1)*lda+k0 : (i+1)*lda+k0+kLen : (i+1)*lda+k0+kLen]
	a2 := A[(i+2)*lda+k0 : (i+2)*lda+k0+kLen : (i+2)*lda+k0+kLen]
	a3 := A[(i+3)*lda+k0 : (i+3)*lda+k0+kLen : (i+3)*lda+k0+kLen]

	k := 0
	bIdx := k0*ldb + j // incrementing B index (avoids multiply per iteration)
	// Unroll k by 4
	for ; k+3 < kLen; k += 4 {
		a0k := a0[k]
		a1k := a1[k]
		a2k := a2[k]
		a3k := a3[k]
		bs := B[bIdx : bIdx+8 : bIdx+8] // BCE
		b0 := bs[0]
		b1 := bs[1]
		b2 := bs[2]
		b3 := bs[3]
		b4 := bs[4]
		b5 := bs[5]
		b6 := bs[6]
		b7 := bs[7]
		c00 += a0k * b0
		c01 += a0k * b1
		c02 += a0k * b2
		c03 += a0k * b3
		c04 += a0k * b4
		c05 += a0k * b5
		c06 += a0k * b6
		c07 += a0k * b7
		c10 += a1k * b0
		c11 += a1k * b1
		c12 += a1k * b2
		c13 += a1k * b3
		c14 += a1k * b4
		c15 += a1k * b5
		c16 += a1k * b6
		c17 += a1k * b7
		c20 += a2k * b0
		c21 += a2k * b1
		c22 += a2k * b2
		c23 += a2k * b3
		c24 += a2k * b4
		c25 += a2k * b5
		c26 += a2k * b6
		c27 += a2k * b7
		c30 += a3k * b0
		c31 += a3k * b1
		c32 += a3k * b2
		c33 += a3k * b3
		c34 += a3k * b4
		c35 += a3k * b5
		c36 += a3k * b6
		c37 += a3k * b7

		bIdx += ldb
		a0k = a0[k+1]
		a1k = a1[k+1]
		a2k = a2[k+1]
		a3k = a3[k+1]
		bs = B[bIdx : bIdx+8 : bIdx+8] // BCE
		b0 = bs[0]
		b1 = bs[1]
		b2 = bs[2]
		b3 = bs[3]
		b4 = bs[4]
		b5 = bs[5]
		b6 = bs[6]
		b7 = bs[7]
		c00 += a0k * b0
		c01 += a0k * b1
		c02 += a0k * b2
		c03 += a0k * b3
		c04 += a0k * b4
		c05 += a0k * b5
		c06 += a0k * b6
		c07 += a0k * b7
		c10 += a1k * b0
		c11 += a1k * b1
		c12 += a1k * b2
		c13 += a1k * b3
		c14 += a1k * b4
		c15 += a1k * b5
		c16 += a1k * b6
		c17 += a1k * b7
		c20 += a2k * b0
		c21 += a2k * b1
		c22 += a2k * b2
		c23 += a2k * b3
		c24 += a2k * b4
		c25 += a2k * b5
		c26 += a2k * b6
		c27 += a2k * b7
		c30 += a3k * b0
		c31 += a3k * b1
		c32 += a3k * b2
		c33 += a3k * b3
		c34 += a3k * b4
		c35 += a3k * b5
		c36 += a3k * b6
		c37 += a3k * b7

		bIdx += ldb
		a0k = a0[k+2]
		a1k = a1[k+2]
		a2k = a2[k+2]
		a3k = a3[k+2]
		bs = B[bIdx : bIdx+8 : bIdx+8] // BCE
		b0 = bs[0]
		b1 = bs[1]
		b2 = bs[2]
		b3 = bs[3]
		b4 = bs[4]
		b5 = bs[5]
		b6 = bs[6]
		b7 = bs[7]
		c00 += a0k * b0
		c01 += a0k * b1
		c02 += a0k * b2
		c03 += a0k * b3
		c04 += a0k * b4
		c05 += a0k * b5
		c06 += a0k * b6
		c07 += a0k * b7
		c10 += a1k * b0
		c11 += a1k * b1
		c12 += a1k * b2
		c13 += a1k * b3
		c14 += a1k * b4
		c15 += a1k * b5
		c16 += a1k * b6
		c17 += a1k * b7
		c20 += a2k * b0
		c21 += a2k * b1
		c22 += a2k * b2
		c23 += a2k * b3
		c24 += a2k * b4
		c25 += a2k * b5
		c26 += a2k * b6
		c27 += a2k * b7
		c30 += a3k * b0
		c31 += a3k * b1
		c32 += a3k * b2
		c33 += a3k * b3
		c34 += a3k * b4
		c35 += a3k * b5
		c36 += a3k * b6
		c37 += a3k * b7

		bIdx += ldb
		a0k = a0[k+3]
		a1k = a1[k+3]
		a2k = a2[k+3]
		a3k = a3[k+3]
		bs = B[bIdx : bIdx+8 : bIdx+8] // BCE
		b0 = bs[0]
		b1 = bs[1]
		b2 = bs[2]
		b3 = bs[3]
		b4 = bs[4]
		b5 = bs[5]
		b6 = bs[6]
		b7 = bs[7]
		c00 += a0k * b0
		c01 += a0k * b1
		c02 += a0k * b2
		c03 += a0k * b3
		c04 += a0k * b4
		c05 += a0k * b5
		c06 += a0k * b6
		c07 += a0k * b7
		c10 += a1k * b0
		c11 += a1k * b1
		c12 += a1k * b2
		c13 += a1k * b3
		c14 += a1k * b4
		c15 += a1k * b5
		c16 += a1k * b6
		c17 += a1k * b7
		c20 += a2k * b0
		c21 += a2k * b1
		c22 += a2k * b2
		c23 += a2k * b3
		c24 += a2k * b4
		c25 += a2k * b5
		c26 += a2k * b6
		c27 += a2k * b7
		c30 += a3k * b0
		c31 += a3k * b1
		c32 += a3k * b2
		c33 += a3k * b3
		c34 += a3k * b4
		c35 += a3k * b5
		c36 += a3k * b6
		c37 += a3k * b7
		bIdx += ldb
	}
	// k remainder
	for ; k < kLen; k++ {
		a0k := a0[k]
		a1k := a1[k]
		a2k := a2[k]
		a3k := a3[k]
		bs := B[bIdx : bIdx+8 : bIdx+8] // BCE
		b0 := bs[0]
		b1 := bs[1]
		b2 := bs[2]
		b3 := bs[3]
		b4 := bs[4]
		b5 := bs[5]
		b6 := bs[6]
		b7 := bs[7]
		c00 += a0k * b0
		c01 += a0k * b1
		c02 += a0k * b2
		c03 += a0k * b3
		c04 += a0k * b4
		c05 += a0k * b5
		c06 += a0k * b6
		c07 += a0k * b7
		c10 += a1k * b0
		c11 += a1k * b1
		c12 += a1k * b2
		c13 += a1k * b3
		c14 += a1k * b4
		c15 += a1k * b5
		c16 += a1k * b6
		c17 += a1k * b7
		c20 += a2k * b0
		c21 += a2k * b1
		c22 += a2k * b2
		c23 += a2k * b3
		c24 += a2k * b4
		c25 += a2k * b5
		c26 += a2k * b6
		c27 += a2k * b7
		c30 += a3k * b0
		c31 += a3k * b1
		c32 += a3k * b2
		c33 += a3k * b3
		c34 += a3k * b4
		c35 += a3k * b5
		c36 += a3k * b6
		c37 += a3k * b7
		bIdx += ldb
	}

	// Store results - use exact-length slices for BCE
	o0 := i*ldb + j
	cr0 := C[o0 : o0+8 : o0+8]
	cr0[0] += c00
	cr0[1] += c01
	cr0[2] += c02
	cr0[3] += c03
	cr0[4] += c04
	cr0[5] += c05
	cr0[6] += c06
	cr0[7] += c07
	o1 := o0 + ldb
	cr1 := C[o1 : o1+8 : o1+8]
	cr1[0] += c10
	cr1[1] += c11
	cr1[2] += c12
	cr1[3] += c13
	cr1[4] += c14
	cr1[5] += c15
	cr1[6] += c16
	cr1[7] += c17
	o2 := o1 + ldb
	cr2 := C[o2 : o2+8 : o2+8]
	cr2[0] += c20
	cr2[1] += c21
	cr2[2] += c22
	cr2[3] += c23
	cr2[4] += c24
	cr2[5] += c25
	cr2[6] += c26
	cr2[7] += c27
	o3 := o2 + ldb
	cr3 := C[o3 : o3+8 : o3+8]
	cr3[0] += c30
	cr3[1] += c31
	cr3[2] += c32
	cr3[3] += c33
	cr3[4] += c34
	cr3[5] += c35
	cr3[6] += c36
	cr3[7] += c37
}

// microKernel4x4: 16 accumulators for column remainder
func microKernel4x4(A, B, C []float32, i, j, k0, kLen, lda, ldb int) {
	var c00, c01, c02, c03 float32
	var c10, c11, c12, c13 float32
	var c20, c21, c22, c23 float32
	var c30, c31, c32, c33 float32
	a0 := A[i*lda+k0 : i*lda+k0+kLen : i*lda+k0+kLen]             // BCE
	a1 := A[(i+1)*lda+k0 : (i+1)*lda+k0+kLen : (i+1)*lda+k0+kLen] // BCE
	a2 := A[(i+2)*lda+k0 : (i+2)*lda+k0+kLen : (i+2)*lda+k0+kLen] // BCE
	a3 := A[(i+3)*lda+k0 : (i+3)*lda+k0+kLen : (i+3)*lda+k0+kLen] // BCE
	bIdx := k0*ldb + j
	for k := 0; k < kLen; k++ {
		a0k := a0[k]
		a1k := a1[k]
		a2k := a2[k]
		a3k := a3[k]
		bSlice := B[bIdx : bIdx+4 : bIdx+4] // BCE
		b0 := bSlice[0]
		b1 := bSlice[1]
		b2 := bSlice[2]
		b3 := bSlice[3]
		c00 += a0k * b0
		c01 += a0k * b1
		c02 += a0k * b2
		c03 += a0k * b3
		c10 += a1k * b0
		c11 += a1k * b1
		c12 += a1k * b2
		c13 += a1k * b3
		c20 += a2k * b0
		c21 += a2k * b1
		c22 += a2k * b2
		c23 += a2k * b3
		c30 += a3k * b0
		c31 += a3k * b1
		c32 += a3k * b2
		c33 += a3k * b3
		bIdx += ldb
	}
	o0 := i*ldb + j
	cr0 := C[o0 : o0+4 : o0+4] // BCE
	cr0[0] += c00
	cr0[1] += c01
	cr0[2] += c02
	cr0[3] += c03
	o1 := o0 + ldb
	cr1 := C[o1 : o1+4 : o1+4] // BCE
	cr1[0] += c10
	cr1[1] += c11
	cr1[2] += c12
	cr1[3] += c13
	o2 := o1 + ldb
	cr2 := C[o2 : o2+4 : o2+4] // BCE
	cr2[0] += c20
	cr2[1] += c21
	cr2[2] += c22
	cr2[3] += c23
	o3 := o2 + ldb
	cr3 := C[o3 : o3+4 : o3+4] // BCE
	cr3[0] += c30
	cr3[1] += c31
	cr3[2] += c32
	cr3[3] += c33
}

// microKernel2x8: 16 accumulators for row remainder
func microKernel2x8(A, B, C []float32, i, j, k0, kLen, lda, ldb int) {
	var c00, c01, c02, c03, c04, c05, c06, c07 float32
	var c10, c11, c12, c13, c14, c15, c16, c17 float32
	a0 := A[i*lda+k0 : i*lda+k0+kLen : i*lda+k0+kLen]             // BCE
	a1 := A[(i+1)*lda+k0 : (i+1)*lda+k0+kLen : (i+1)*lda+k0+kLen] // BCE
	bIdx := k0*ldb + j
	for k := 0; k < kLen; k++ {
		a0k := a0[k]
		a1k := a1[k]
		bSlice := B[bIdx : bIdx+8 : bIdx+8] // BCE
		b0 := bSlice[0]
		b1 := bSlice[1]
		b2 := bSlice[2]
		b3 := bSlice[3]
		b4 := bSlice[4]
		b5 := bSlice[5]
		b6 := bSlice[6]
		b7 := bSlice[7]
		c00 += a0k * b0
		c01 += a0k * b1
		c02 += a0k * b2
		c03 += a0k * b3
		c04 += a0k * b4
		c05 += a0k * b5
		c06 += a0k * b6
		c07 += a0k * b7
		c10 += a1k * b0
		c11 += a1k * b1
		c12 += a1k * b2
		c13 += a1k * b3
		c14 += a1k * b4
		c15 += a1k * b5
		c16 += a1k * b6
		c17 += a1k * b7
		bIdx += ldb
	}
	o0 := i*ldb + j
	cr0 := C[o0 : o0+8 : o0+8] // BCE
	cr0[0] += c00
	cr0[1] += c01
	cr0[2] += c02
	cr0[3] += c03
	cr0[4] += c04
	cr0[5] += c05
	cr0[6] += c06
	cr0[7] += c07
	o1 := o0 + ldb
	cr1 := C[o1 : o1+8 : o1+8] // BCE
	cr1[0] += c10
	cr1[1] += c11
	cr1[2] += c12
	cr1[3] += c13
	cr1[4] += c14
	cr1[5] += c15
	cr1[6] += c16
	cr1[7] += c17
}

// microKernel1x8: 8 accumulators for single row remainder
func microKernel1x8(A, B, C []float32, i, j, k0, kLen, lda, ldb int) {
	var c0, c1, c2, c3, c4, c5, c6, c7 float32
	aRow := A[i*lda+k0 : i*lda+k0+kLen : i*lda+k0+kLen] // BCE
	bIdx := k0*ldb + j
	for k := 0; k < kLen; k++ {
		aik := aRow[k]
		bSlice := B[bIdx : bIdx+8 : bIdx+8] // BCE
		c0 += aik * bSlice[0]
		c1 += aik * bSlice[1]
		c2 += aik * bSlice[2]
		c3 += aik * bSlice[3]
		c4 += aik * bSlice[4]
		c5 += aik * bSlice[5]
		c6 += aik * bSlice[6]
		c7 += aik * bSlice[7]
		bIdx += ldb
	}
	o := i*ldb + j
	cr := C[o : o+8 : o+8] // BCE
	cr[0] += c0
	cr[1] += c1
	cr[2] += c2
	cr[3] += c3
	cr[4] += c4
	cr[5] += c5
	cr[6] += c6
	cr[7] += c7
}

// microKernelMxN: generic small M x N remainder kernel
func microKernelMxN(A, B, C []float32, i, j, k0, kLen, lda, ldb, mR, nR int) {
	for ii := 0; ii < mR; ii++ {
		aRow := A[(i+ii)*lda+k0 : (i+ii)*lda+k0+kLen : (i+ii)*lda+k0+kLen] // BCE
		cSlice := C[(i+ii)*ldb+j : (i+ii)*ldb+j+nR : (i+ii)*ldb+j+nR]     // BCE
		bIdx := k0*ldb + j
		for k := 0; k < kLen; k++ {
			aik := aRow[k]
			if aik == 0 {
				bIdx += ldb
				continue
			}
			bSlice := B[bIdx : bIdx+nR : bIdx+nR] // BCE
			for jj := 0; jj < nR; jj++ {
				cSlice[jj] += aik * bSlice[jj]
			}
			bIdx += ldb
		}
	}
}

// gemmF32WithBias computes C = A*B + bias (fused).
func gemmF32WithBias(A, B, C []float32, M, N, K int, bias []float32) {
	gemmF32(A, B, C, M, N, K)
	for i := 0; i < M; i++ {
		bv := bias[i]
		cRow := C[i*N : i*N+N : i*N+N] // BCE
		for j := 0; j < N; j++ {
			cRow[j] += bv
		}
	}
}
