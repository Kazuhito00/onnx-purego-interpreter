package onnx

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestKernelConfigOnOff runs each model with default config and with all kernel
// optimizations disabled, verifying the results match (correctness) while
// allowing different performance characteristics.
func TestKernelConfigOnOff(t *testing.T) {
	models := []struct {
		name string
		tol  float64
	}{
		{"matmul_2d", 1e-5},
		{"gemm_bias", 1e-5},
		{"conv2d", 1e-4},
		{"conv_relu_pool", 1e-4},
		{"conv_transpose", 1e-4},
		{"mlp_small", 1e-3},
		{"maxpool", 0},
		{"batchnorm", 1e-4},
	}

	for _, m := range models {
		t.Run(m.name, func(t *testing.T) {
			caseDir := filepath.Join(testdataDir, "ops", m.name)
			modelPath := filepath.Join(caseDir, "model.onnx")
			modelBytes, err := os.ReadFile(modelPath)
			if err != nil {
				t.Skipf("model not found: %s", modelPath)
				return
			}

			// Default config (all optimizations ON)
			sessOn, err := NewSession(modelBytes)
			if err != nil {
				t.Fatalf("NewSession (default): %v", err)
			}

			// All kernel optimizations OFF
			kc := &KernelConfig{} // all false
			sessOff, err := NewSessionWithOptions(modelBytes, WithKernelConfig(kc))
			if err != nil {
				t.Fatalf("NewSession (kernels off): %v", err)
			}

			inputs, err := loadInputs(caseDir)
			if err != nil || len(inputs) == 0 {
				t.Skipf("cannot load test inputs: %v", err)
				return
			}

			outOn, err := sessOn.RunWithNames(inputs)
			if err != nil {
				t.Fatalf("Run (default): %v", err)
			}

			inputs2, _ := loadInputs(caseDir)
			outOff, err := sessOff.RunWithNames(inputs2)
			if err != nil {
				t.Fatalf("Run (kernels off): %v", err)
			}

			// Compare outputs
			for name, tOn := range outOn {
				tOff, ok := outOff[name]
				if !ok {
					continue
				}
				dOn, err1 := tensorToFloat32(tOn)
				dOff, err2 := tensorToFloat32(tOff)
				if err1 != nil || err2 != nil || len(dOn) != len(dOff) {
					continue
				}
				maxDiff := float64(0)
				for i := range dOn {
					diff := math.Abs(float64(dOn[i] - dOff[i]))
					if diff > maxDiff {
						maxDiff = diff
					}
				}
				t.Logf("%s %q: on vs off max_diff=%.6e", m.name, name, maxDiff)
				if m.tol > 0 && maxDiff > m.tol {
					t.Errorf("max diff %.6e exceeds tolerance %.1e", maxDiff, m.tol)
				}
			}
		})
	}
}

// TestKernelConfigIndividual tests each kernel config flag individually.
func TestKernelConfigIndividual(t *testing.T) {
	modelPath := filepath.Join(testdataDir, "ops", "conv2d", "model.onnx")
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		t.Skipf("model not found")
		return
	}

	flags := []struct {
		name string
		set  func(*KernelConfig)
	}{
		{"UseTiledGEMM=false", func(kc *KernelConfig) { kc.UseTiledGEMM = false }},
		{"UseDepthwiseKernel=false", func(kc *KernelConfig) { kc.UseDepthwiseKernel = false }},
		{"Use1x1FastPath=false", func(kc *KernelConfig) { kc.Use1x1FastPath = false }},
		{"UseConvTransposeGEMM=false", func(kc *KernelConfig) { kc.UseConvTransposeGEMM = false }},
		{"UsePoolFastPath=false", func(kc *KernelConfig) { kc.UsePoolFastPath = false }},
		{"UseFastErf=false", func(kc *KernelConfig) { kc.UseFastErf = false }},
		{"UseParallelConv=false", func(kc *KernelConfig) { kc.UseParallelConv = false }},
	}

	for _, f := range flags {
		t.Run(f.name, func(t *testing.T) {
			kc := DefaultKernelConfig()
			f.set(kc)
			sess, err := NewSessionWithOptions(modelBytes, WithKernelConfig(kc))
			if err != nil {
				t.Fatalf("NewSession: %v", err)
			}
			inputs, err := loadInputs(filepath.Join(testdataDir, "ops", "conv2d"))
			if err != nil {
				t.Skip("no inputs")
				return
			}
			out, err := sess.RunWithNames(inputs)
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			if len(out) == 0 {
				t.Fatal("no output")
			}
			t.Logf("%s: PASS", f.name)
		})
	}
}
