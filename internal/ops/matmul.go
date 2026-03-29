package ops

import (
	"fmt"
	"sync"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// matMul2D performs 2D matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// Tiled ikj loop for L1/L2 cache locality (pure Go).
func matMul2D[T tensor.Numeric](a, b *tensor.Dense[T], M, K, N int) *tensor.Dense[T] {
	data := make([]T, M*N)
	ad, bd := a.Data(), b.Data()
	tiledGemm(ad, bd, data, M, N, K)
	return tensor.NewDense[T](tensor.Shape{M, N}, data)
}

func gemmF32Attention(A, B, C []float32, M, N, K int) {
	// Tiny-M/N attention projections are numerically sensitive; the reference
	// loop is slower but avoids extra error from the tiled microkernel.
	if M <= 4 || N <= 4 {
		gemmF32PreciseSmall(A, B, C, M, N, K)
		return
	}
	gemmF32(A, B, C, M, N, K)
}

func gemmF32PreciseSmall(A, B, C []float32, M, N, K int) {
	for i := 0; i < M; i++ {
		cRow := C[i*N : i*N+N : i*N+N] // BCE
		aRow := A[i*K : i*K+K : i*K+K] // BCE
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < K; k++ {
				sum += float64(aRow[k]) * float64(B[k*N+j])
			}
			cRow[j] += float32(sum)
		}
	}
}

// tiledGemm computes C += A*B. Tiled for large, simple ikj for small.
func tiledGemm[T tensor.Numeric](A, B, C []T, M, N, K int) {
	if M*K+K*N > 32*1024 {
		const tM = 32
		const tN = 128
		const tK = 64
		for i0 := 0; i0 < M; i0 += tM {
			iEnd := min(i0+tM, M)
			for k0 := 0; k0 < K; k0 += tK {
				kEnd := min(k0+tK, K)
				for j0 := 0; j0 < N; j0 += tN {
					jEnd := min(j0+tN, N)
					jLen := jEnd - j0
					for i := i0; i < iEnd; i++ {
						cRow := C[i*N+j0 : i*N+jEnd : i*N+jEnd] // BCE
						aBase := i * K
						for k := k0; k < kEnd; k++ {
							aik := A[aBase+k]
							if aik == 0 {
								continue
							}
							bRow := B[k*N+j0 : k*N+jEnd : k*N+jEnd] // BCE
							for j := 0; j < jLen; j++ {
								cRow[j] += aik * bRow[j]
							}
						}
					}
				}
			}
		}
	} else {
		for i := 0; i < M; i++ {
			cRow := C[i*N : i*N+N : i*N+N] // BCE
			aBase := i * K
			for k := 0; k < K; k++ {
				aik := A[aBase+k]
				if aik == 0 {
					continue
				}
				bRow := B[k*N : k*N+N : k*N+N] // BCE
				for j := 0; j < N; j++ {
					cRow[j] += aik * bRow[j]
				}
			}
		}
	}
}

var activeMatMulConfig *KernelConfig

func makeMatMul(kc *KernelConfig) OpFunc {
	return func(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
		activeMatMulConfig = kc
		return opMatMulWithConfig(node, inputs, kc)
	}
}

func opMatMul(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return opMatMulWithConfig(node, inputs, nil)
}

func opMatMulWithConfig(node *ir.Node, inputs []tensor.Tensor, kc *KernelConfig) ([]tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	useTiled := kc == nil || kc.UseTiledGEMM

	// Fast path: packed weight for 2D MatMul (only with tiled GEMM)
	if useTiled {
	if pb, ok := b.(*tensor.PackedF32); ok {
		if at, ok := a.(*tensor.Dense[float32]); ok && at.Shape().NDim() == 2 {
			M := at.Shape()[0]
			N := pb.N
			outData := make([]float32, M*N)
			gemmF32Packed(at.Data(), &PackedB{Data: pb.Packed, K: pb.K, N: pb.N}, outData, M)
			return []tensor.Tensor{tensor.NewDense[float32](tensor.Shape{M, N}, outData)}, nil
		}
		// N-D with packed: fall back to original data
		if at, ok := a.(*tensor.Dense[float32]); ok {
			bt := tensor.NewDense[float32](pb.Shape(), pb.Original)
			out, err := doMatMul(at, bt)
			if err != nil {
				return nil, fmt.Errorf("MatMul: %w", err)
			}
			return []tensor.Tensor{out}, nil
		}
	}
	} // end useTiled packed path

	switch at := a.(type) {
	case *tensor.Dense[float32]:
		bt := b.(*tensor.Dense[float32])
		out, err := doMatMul(at, bt)
		if err != nil {
			return nil, fmt.Errorf("MatMul: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		bt := b.(*tensor.Dense[float64])
		out, err := doMatMul(at, bt)
		if err != nil {
			return nil, fmt.Errorf("MatMul: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[int32]:
		bt := b.(*tensor.Dense[int32])
		out, err := doMatMul(at, bt)
		if err != nil {
			return nil, fmt.Errorf("MatMul: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[int64]:
		bt := b.(*tensor.Dense[int64])
		out, err := doMatMul(at, bt)
		if err != nil {
			return nil, fmt.Errorf("MatMul: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("MatMul: unsupported type %T", a)
	}
}

// doMatMul handles 2D and batched N-D matmul.
func doMatMul[T tensor.Numeric](a, b *tensor.Dense[T]) (*tensor.Dense[T], error) {
	as, bs := a.Shape(), b.Shape()
	aNdim, bNdim := as.NDim(), bs.NDim()

	// 1D x 1D → scalar (dot product)
	if aNdim == 1 && bNdim == 1 {
		if as[0] != bs[0] {
			return nil, fmt.Errorf("shapes %v and %v incompatible for dot product", as, bs)
		}
		var sum T
		ad, bd := a.Data(), b.Data()
		for i := 0; i < as[0]; i++ {
			sum += ad[i] * bd[i]
		}
		return tensor.NewDenseScalar(sum), nil
	}

	// 2D x 2D
	if aNdim == 2 && bNdim == 2 {
		M, K1 := as[0], as[1]
		K2, N := bs[0], bs[1]
		if K1 != K2 {
			return nil, fmt.Errorf("shapes %v and %v incompatible: K=%d vs %d", as, bs, K1, K2)
		}
		return matMul2D(a, b, M, K1, N), nil
	}

	// 2D x 1D → [M]
	if aNdim == 2 && bNdim == 1 {
		M, K := as[0], as[1]
		if K != bs[0] {
			return nil, fmt.Errorf("shapes %v and %v incompatible", as, bs)
		}
		data := make([]T, M)
		ad, bd := a.Data(), b.Data()
		for i := 0; i < M; i++ {
			aRow := ad[i*K : i*K+K : i*K+K] // BCE
			var sum T
			for k := 0; k < K; k++ {
				sum += aRow[k] * bd[k]
			}
			data[i] = sum
		}
		return tensor.NewDense[T](tensor.Shape{M}, data), nil
	}

	// 1D x 2D → [N]
	if aNdim == 1 && bNdim == 2 {
		K := as[0]
		K2, N := bs[0], bs[1]
		if K != K2 {
			return nil, fmt.Errorf("shapes %v and %v incompatible", as, bs)
		}
		data := make([]T, N)
		ad, bd := a.Data(), b.Data()
		for j := 0; j < N; j++ {
			var sum T
			for k := 0; k < K; k++ {
				sum += ad[k] * bd[k*N+j]
			}
			data[j] = sum
		}
		return tensor.NewDense[T](tensor.Shape{N}, data), nil
	}

	// N-D batched matmul: broadcast batch dims, matmul last two dims
	// Treat as batch of 2D matmuls
	aMatShape := tensor.Shape{as[aNdim-2], as[aNdim-1]}
	bMatShape := tensor.Shape{bs[bNdim-2], bs[bNdim-1]}

	M, K := aMatShape[0], aMatShape[1]
	K2, N := bMatShape[0], bMatShape[1]
	if K != K2 {
		return nil, fmt.Errorf("shapes %v and %v incompatible: K=%d vs %d", as, bs, K, K2)
	}

	// Broadcast batch dimensions
	aBatch := as[:aNdim-2]
	bBatch := bs[:bNdim-2]
	outBatch, err := tensor.BroadcastShape(aBatch, bBatch)
	if err != nil {
		return nil, fmt.Errorf("batch shapes %v and %v not broadcast-compatible: %w", aBatch, bBatch, err)
	}

	batchSize := outBatch.Size()
	outShape := make(tensor.Shape, len(outBatch)+2)
	copy(outShape, outBatch)
	outShape[len(outBatch)] = M
	outShape[len(outBatch)+1] = N

	matSize := M * K
	bMatSize := K * N
	outMatSize := M * N

	outData := make([]T, batchSize*outMatSize)
	ad, bd := a.Data(), b.Data()

	aBatchStrides := tensor.Strides(aBatch)
	bBatchStrides := tensor.Strides(bBatch)
	outBatchStrides := tensor.Strides(outBatch)

	// Float32 fast path with parallelization
	if af, ok := any(ad).([]float32); ok {
		bf := any(bd).([]float32)
		of := any(outData).([]float32)

		// Pre-compute batch offsets to avoid BroadcastIndex in hot loop
		type batchOff struct{ aOff, bOff, oOff int }
		offsets := make([]batchOff, batchSize)
		for batch := 0; batch < batchSize; batch++ {
			aIdx := tensor.BroadcastIndex(batch, outBatch, aBatch, outBatchStrides, aBatchStrides)
			bIdx := tensor.BroadcastIndex(batch, outBatch, bBatch, outBatchStrides, bBatchStrides)
			offsets[batch] = batchOff{aIdx * matSize, bIdx * bMatSize, batch * outMatSize}
		}

		// Parallelize across batches for large workloads
		workPerBatch := M * N * K
		if batchSize >= 4 && workPerBatch >= 1024 {
			nWorkers := activeMatMulConfig.Workers()
			if nWorkers > batchSize {
				nWorkers = batchSize
			}
			var wg sync.WaitGroup
			chunkSize := (batchSize + nWorkers - 1) / nWorkers
			for w := 0; w < nWorkers; w++ {
				start := w * chunkSize
				end := start + chunkSize
				if end > batchSize {
					end = batchSize
				}
				if start >= end {
					break
				}
				wg.Add(1)
				go func(start, end int) {
					defer wg.Done()
					for batch := start; batch < end; batch++ {
						o := offsets[batch]
						gemmF32Attention(af[o.aOff:o.aOff+matSize], bf[o.bOff:o.bOff+bMatSize], of[o.oOff:o.oOff+outMatSize], M, N, K)
					}
				}(start, end)
			}
			wg.Wait()
		} else {
			for batch := 0; batch < batchSize; batch++ {
				o := offsets[batch]
				gemmF32Attention(af[o.aOff:o.aOff+matSize], bf[o.bOff:o.bOff+bMatSize], of[o.oOff:o.oOff+outMatSize], M, N, K)
			}
		}
	} else {
		// Generic path
		for batch := 0; batch < batchSize; batch++ {
			aIdx := tensor.BroadcastIndex(batch, outBatch, aBatch, outBatchStrides, aBatchStrides)
			bIdx := tensor.BroadcastIndex(batch, outBatch, bBatch, outBatchStrides, bBatchStrides)
			aOff := aIdx * matSize
			bOff := bIdx * bMatSize
			oOff := batch * outMatSize
			for i := 0; i < M; i++ {
				cRow := outData[oOff+i*N : oOff+i*N+N : oOff+i*N+N] // BCE
				for k := 0; k < K; k++ {
					aik := ad[aOff+i*K+k]
					if aik == 0 {
						continue
					}
					bRow := bd[bOff+k*N : bOff+k*N+N : bOff+k*N+N] // BCE
					for j := 0; j < N; j++ {
						cRow[j] += aik * bRow[j]
					}
				}
			}
		}
	}

	return tensor.NewDense[T](outShape, outData), nil
}

func opGemm(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Gemm: Y = alpha * A' * B' + beta * C
	// where A' = transA ? A^T : A, B' = transB ? B^T : B
	alpha := node.GetAttrFloat("alpha", 1.0)
	beta := node.GetAttrFloat("beta", 1.0)
	transA := node.GetAttrInt("transA", 0) != 0
	transB := node.GetAttrInt("transB", 0) != 0

	switch at := inputs[0].(type) {
	case *tensor.Dense[float32]:
		bt := inputs[1].(*tensor.Dense[float32])
		var ct *tensor.Dense[float32]
		if len(inputs) > 2 && inputs[2] != nil {
			ct = inputs[2].(*tensor.Dense[float32])
		}
		out, err := doGemm(at, bt, ct, float32(alpha), float32(beta), transA, transB)
		if err != nil {
			return nil, fmt.Errorf("Gemm: %w", err)
		}
		return []tensor.Tensor{out}, nil
	case *tensor.Dense[float64]:
		bt := inputs[1].(*tensor.Dense[float64])
		var ct *tensor.Dense[float64]
		if len(inputs) > 2 && inputs[2] != nil {
			ct = inputs[2].(*tensor.Dense[float64])
		}
		out, err := doGemm(at, bt, ct, float64(alpha), float64(beta), transA, transB)
		if err != nil {
			return nil, fmt.Errorf("Gemm: %w", err)
		}
		return []tensor.Tensor{out}, nil
	default:
		return nil, fmt.Errorf("Gemm: unsupported type %T", inputs[0])
	}
}

func doGemm[T tensor.Numeric](a, b, c *tensor.Dense[T], alpha, beta T, transA, transB bool) (*tensor.Dense[T], error) {
	as, bs := a.Shape(), b.Shape()
	if as.NDim() != 2 || bs.NDim() != 2 {
		return nil, fmt.Errorf("Gemm requires 2D inputs, got %v and %v", as, bs)
	}

	M, KA := as[0], as[1]
	if transA {
		M, KA = KA, M
	}
	KB, N := bs[0], bs[1]
	if transB {
		KB, N = N, KB
	}
	if KA != KB {
		return nil, fmt.Errorf("Gemm K mismatch: %d vs %d", KA, KB)
	}
	K := KA

	data := make([]T, M*N)
	ad, bd := a.Data(), b.Data()
	aStride := as[1]
	bStride := bs[1]

	// Optimized GEMM with trans variants using ikj order
	switch {
	case !transA && !transB:
		for i := 0; i < M; i++ {
			cRow := data[i*N : i*N+N : i*N+N] // BCE
			for k := 0; k < K; k++ {
				aik := alpha * ad[i*aStride+k]
				if aik == 0 {
					continue
				}
				bRow := bd[k*bStride : k*bStride+N : k*bStride+N] // BCE
				for j := 0; j < N; j++ {
					cRow[j] += aik * bRow[j]
				}
			}
		}
	case !transA && transB:
		for i := 0; i < M; i++ {
			cRow := data[i*N : i*N+N : i*N+N]             // BCE
			aRow := ad[i*aStride : i*aStride+K : i*aStride+K] // BCE
			for j := 0; j < N; j++ {
				bRow := bd[j*bStride : j*bStride+K : j*bStride+K] // BCE
				var sum T
				for k := 0; k < K; k++ {
					sum += aRow[k] * bRow[k]
				}
				cRow[j] = alpha * sum
			}
		}
	case transA && !transB:
		for k := 0; k < K; k++ {
			for i := 0; i < M; i++ {
				aik := alpha * ad[k*aStride+i]
				if aik == 0 {
					continue
				}
				cRow := data[i*N : i*N+N : i*N+N]             // BCE
				bRow := bd[k*bStride : k*bStride+N : k*bStride+N] // BCE
				for j := 0; j < N; j++ {
					cRow[j] += aik * bRow[j]
				}
			}
		}
	case transA && transB:
		for i := 0; i < M; i++ {
			cRow := data[i*N : i*N+N : i*N+N] // BCE
			for j := 0; j < N; j++ {
				var sum T
				for k := 0; k < K; k++ {
					sum += ad[k*aStride+i] * bd[j*bStride+k]
				}
				cRow[j] = alpha * sum
			}
		}
	}

	// Add bias C if present
	if c != nil {
		cs := c.Shape()
		cd := c.Data()
		if cs.NDim() == 1 && cs[0] == N {
			// C is [N], broadcast to [M, N]
			for i := 0; i < M; i++ {
				dRow := data[i*N : i*N+N : i*N+N] // BCE
				for j := 0; j < N; j++ {
					dRow[j] += beta * cd[j]
				}
			}
		} else if cs.NDim() == 2 && cs[0] == M && cs[1] == N {
			// C is [M, N]
			for i := 0; i < M*N; i++ {
				data[i] += beta * cd[i]
			}
		} else if cs.NDim() == 1 && cs[0] == 1 {
			// C is scalar
			for i := range data {
				data[i] += beta * cd[0]
			}
		} else if cs.Size() == 1 {
			for i := range data {
				data[i] += beta * cd[0]
			}
		} else {
			return nil, fmt.Errorf("Gemm: unsupported C shape %v for output [%d,%d]", cs, M, N)
		}
	}

	return tensor.NewDense[T](tensor.Shape{M, N}, data), nil
}
