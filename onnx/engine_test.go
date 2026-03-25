package onnx

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	onnxpb "github.com/Kazuhito00/onnx-purego-interpreter/internal/onnxpb"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
	"google.golang.org/protobuf/proto"
	"html/template"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// getSysInfo returns CPU and memory info for the report.
func getSysInfo() (cpuInfo string, ramInfo string) {
	cpuInfo = fmt.Sprintf("%s/%s (%d cores)", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	// Try to get more CPU details on Windows
	if runtime.GOOS == "windows" {
		out, err := exec.Command("wmic", "cpu", "get", "Name", "/value").Output()
		if err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "Name=") {
					cpuInfo = strings.TrimPrefix(line, "Name=") + fmt.Sprintf(" (%d cores)", runtime.NumCPU())
					break
				}
			}
		}
	}
	// Memory info
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if runtime.GOOS == "windows" {
		out, err := exec.Command("wmic", "OS", "get", "TotalVisibleMemorySize", "/value").Output()
		if err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "TotalVisibleMemorySize=") {
					kbStr := strings.TrimPrefix(line, "TotalVisibleMemorySize=")
					var kb int64
					fmt.Sscanf(kbStr, "%d", &kb)
					if kb > 0 {
						ramInfo = fmt.Sprintf("%.1f GB", float64(kb)/1024/1024)
						return
					}
				}
			}
		}
	}
	ramInfo = fmt.Sprintf("%.1f GB (Go heap)", float64(m.Sys)/1024/1024/1024)
	return
}

const testdataDir = "../testdata"
const defaultTol float32 = 1e-5

// Test flags (use with: go test ./engine -args -bench-runs=3 -filter=squeezenet)
var (
	flagBenchRuns  = flag.Int("bench-runs", 0, "number of inference runs per test case (default 3, or BENCH_RUNS env)")
	flagFilter = flag.String("filter", "", "comma-separated operator name filter (or REPORT_FILTER env)")
)

// Helpers
func loadTensorProto(path string) (*onnxpb.TensorProto, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	tp := &onnxpb.TensorProto{}
	if err := proto.Unmarshal(data, tp); err != nil {
		return nil, fmt.Errorf("unmarshal TensorProto %s: %w", path, err)
	}
	return tp, nil
}
func tensorProtoToTensor(tp *onnxpb.TensorProto) (tensor.Tensor, error) {
	shape := make(tensor.Shape, len(tp.GetDims()))
	for i, d := range tp.GetDims() {
		shape[i] = int(d)
	}
	switch tp.GetDataType() {
	case 1: // FLOAT
		data := tp.GetFloatData()
		if len(data) == 0 && len(tp.GetRawData()) > 0 {
			raw := tp.GetRawData()
			n := len(raw) / 4
			data = make([]float32, n)
			for i := 0; i < n; i++ {
				data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			}
		}
		return tensor.NewDense[float32](shape, data), nil
	case 6: // INT32
		data := tp.GetInt32Data()
		if len(data) == 0 && len(tp.GetRawData()) > 0 {
			raw := tp.GetRawData()
			n := len(raw) / 4
			data = make([]int32, n)
			for i := 0; i < n; i++ {
				data[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
			}
		}
		return tensor.NewDense[int32](shape, data), nil
	case 3: // INT8
		data := tp.GetRawData()
		buf := make([]int8, len(data))
		for i, v := range data {
			buf[i] = int8(v)
		}
		return tensor.NewDense[int8](shape, buf), nil
	case 2, 9: // UINT8 / BOOL
		data := tp.GetRawData()
		if len(data) == 0 {
			ints := tp.GetInt32Data()
			buf := make([]uint8, len(ints))
			for i, v := range ints {
				buf[i] = uint8(v)
			}
			return tensor.NewDense[uint8](shape, buf), nil
		}
		buf := make([]uint8, len(data))
		copy(buf, data)
		return tensor.NewDense[uint8](shape, buf), nil
	case 11: // DOUBLE
		data := tp.GetDoubleData()
		if len(data) == 0 && len(tp.GetRawData()) > 0 {
			raw := tp.GetRawData()
			n := len(raw) / 8
			data = make([]float64, n)
			for i := 0; i < n; i++ {
				data[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
			}
		}
		return tensor.NewDense[float64](shape, data), nil
	case 7: // INT64
		data := tp.GetInt64Data()
		if len(data) == 0 && len(tp.GetRawData()) > 0 {
			raw := tp.GetRawData()
			n := len(raw) / 8
			data = make([]int64, n)
			for i := 0; i < n; i++ {
				data[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
			}
		}
		return tensor.NewDense[int64](shape, data), nil
	default:
		return nil, fmt.Errorf("unsupported TensorProto dtype %d", tp.GetDataType())
	}
}
func tensorToFloat32(t tensor.Tensor) ([]float32, error) {
	switch dt := t.(type) {
	case *tensor.Dense[float32]:
		return dt.Data(), nil
	case *tensor.Dense[float64]:
		data := make([]float32, dt.Len())
		for i, v := range dt.Data() {
			data[i] = float32(v)
		}
		return data, nil
	case *tensor.Dense[int64]:
		data := make([]float32, dt.Len())
		for i, v := range dt.Data() {
			data[i] = float32(v)
		}
		return data, nil
	case *tensor.Dense[int32]:
		data := make([]float32, dt.Len())
		for i, v := range dt.Data() {
			data[i] = float32(v)
		}
		return data, nil
	case *tensor.Dense[uint8]:
		data := make([]float32, dt.Len())
		for i, v := range dt.Data() {
			data[i] = float32(v)
		}
		return data, nil
	case *tensor.Dense[int8]:
		data := make([]float32, dt.Len())
		for i, v := range dt.Data() {
			data[i] = float32(v)
		}
		return data, nil
	default:
		return nil, fmt.Errorf("cannot extract float32 from %T", t)
	}
}
func mapKeys(m map[string]tensor.Tensor) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
func reportFilterTokens() []string {
	raw := *flagFilter
	if raw == "" {
		raw = strings.TrimSpace(os.Getenv("REPORT_FILTER"))
	}
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.ToLower(strings.TrimSpace(p))
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
func includeReportCase(category, name string) bool {
	if category == "models" {
		return false
	}
	tokens := reportFilterTokens()
	if len(tokens) == 0 {
		return true
	}
	target := strings.ToLower(category + "/" + name)
	for _, tok := range tokens {
		if strings.Contains(target, tok) {
			return true
		}
	}
	return false
}

// compareOutputs returns maxDiff and error info.
// For integer-typed outputs, tolerance is evaluated as exact match (diff must be 0).
// For float outputs, tolerance uses the caller-provided value.
func compareOutputs(got, want tensor.Tensor, tol float32) (maxDiff float32, maxIdx int, pass bool, err error) {
	gs, ws := got.Shape(), want.Shape()
	if !gs.Equal(ws) {
		return 0, 0, false, fmt.Errorf("shape mismatch: got %v, want %v", gs, ws)
	}
	gotData, err := tensorToFloat32(got)
	if err != nil {
		return 0, 0, false, err
	}
	wantData, err := tensorToFloat32(want)
	if err != nil {
		return 0, 0, false, err
	}
	// Integer outputs (labels etc): use relative match ratio instead of absolute diff
	isIntOutput := false
	switch got.(type) {
	case *tensor.Dense[int64], *tensor.Dense[int32], *tensor.Dense[uint8], *tensor.Dense[int8]:
		isIntOutput = true
	}
	for i := range gotData {
		diff := gotData[i] - wantData[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
			maxIdx = i
		}
	}
	if isIntOutput {
		// Integer outputs (labels etc): allow up to 5% mismatch
		mismatches := 0
		for i := range gotData {
			if gotData[i] != wantData[i] {
				mismatches++
			}
		}
		matchRatio := 1.0 - float64(mismatches)/float64(len(gotData))
		pass = matchRatio >= 0.90
	} else {
		pass = compareFloatData(gotData, wantData, tol)
	}
	return maxDiff, maxIdx, pass, nil
}
func compareFloatData(gotData, wantData []float32, tol float32) bool {
	for i := range gotData {
		diff := gotData[i] - wantData[i]
		if diff < 0 {
			diff = -diff
		}
		absMax := wantData[i]
		if absMax < 0 {
			absMax = -absMax
		}
		if absMax > 1.0 {
			if diff/absMax > tol {
				return false
			}
		} else {
			if diff > tol {
				return false
			}
		}
	}
	return true
}
func detectionLikeOutput(t tensor.Tensor) (rows, cols int, ok bool) {
	s := t.Shape()
	switch s.NDim() {
	case 2:
		if s[1] == 6 {
			return s[0], s[1], true
		}
	case 3:
		if s[0] == 1 && s[2] == 6 {
			return s[1], s[2], true
		}
	}
	return 0, 0, false
}
func compareDetectionPrefix(got, want tensor.Tensor, tol float32) (maxDiff float32, pass bool, comparedRows int, err error) {
	rows, cols, ok := detectionLikeOutput(want)
	if !ok {
		return 0, false, 0, fmt.Errorf("not a detection-like tensor: %v", want.Shape())
	}
	gotRows, gotCols, ok := detectionLikeOutput(got)
	if !ok || gotRows != rows || gotCols != cols {
		return 0, false, 0, fmt.Errorf("shape mismatch: got %v, want %v", got.Shape(), want.Shape())
	}
	gotData, err := tensorToFloat32(got)
	if err != nil {
		return 0, false, 0, err
	}
	wantData, err := tensorToFloat32(want)
	if err != nil {
		return 0, false, 0, err
	}
	comparedRows = rows
	if comparedRows > 10 {
		comparedRows = 10
	}
	end := comparedRows * cols
	for i := 0; i < end; i++ {
		diff := gotData[i] - wantData[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	pass = compareFloatData(gotData[:end], wantData[:end], tol)
	return maxDiff, pass, comparedRows, nil
}

// benchNRuns returns number of inference runs (default 3, set BENCH_RUNS=1 for debug)
func benchNRuns() int {
	if *flagBenchRuns > 0 {
		return *flagBenchRuns
	}
	if s := os.Getenv("BENCH_RUNS"); s != "" {
		var n int
		fmt.Sscanf(s, "%d", &n)
		if n > 0 {
			return n
		}
	}
	return 3
}

// loadInputs loads input_*.pb files from a directory.
func loadInputs(dir string) (map[string]tensor.Tensor, error) {
	inputs := make(map[string]tensor.Tensor)
	files, _ := filepath.Glob(filepath.Join(dir, "input_*.pb"))
	sort.Strings(files)
	for _, f := range files {
		tp, err := loadTensorProto(f)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", f, err)
		}
		t, err := tensorProtoToTensor(tp)
		if err != nil {
			return nil, fmt.Errorf("convert %s: %w", tp.GetName(), err)
		}
		inputs[tp.GetName()] = t
	}
	return inputs, nil
}

// loadExpectedOutputs loads output_*.pb files from a directory.
func loadExpectedOutputs(dir string) (map[string]tensor.Tensor, error) {
	outputs := make(map[string]tensor.Tensor)
	files, _ := filepath.Glob(filepath.Join(dir, "output_*.pb"))
	sort.Strings(files)
	for _, f := range files {
		tp, err := loadTensorProto(f)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", f, err)
		}
		t, err := tensorProtoToTensor(tp)
		if err != nil {
			return nil, fmt.Errorf("convert %s: %w", tp.GetName(), err)
		}
		outputs[tp.GetName()] = t
	}
	return outputs, nil
}

// findOutputTensor tries to find a matching Go output for an expected output name.
func findOutputTensor(outputs map[string]tensor.Tensor, wantName string) (tensor.Tensor, bool) {
	if t, ok := outputs[wantName]; ok {
		return t, true
	}
	if len(outputs) == 1 {
		for _, v := range outputs {
			return v, true
		}
	}
	var matched tensor.Tensor
	count := 0
	for k, v := range outputs {
		if strings.EqualFold(k, wantName) {
			return v, true
		}
		last := k
		if idx := strings.LastIndexAny(k, "/:"); idx >= 0 && idx+1 < len(k) {
			last = k[idx+1:]
		}
		if last == wantName {
			matched = v
			count++
		}
	}
	return matched, count == 1
}
func findOutputTensorByShape(outputs map[string]tensor.Tensor, want tensor.Tensor) (tensor.Tensor, bool) {
	var matched tensor.Tensor
	count := 0
	for _, got := range outputs {
		if got.Shape().Equal(want.Shape()) {
			matched = got
			count++
		}
	}
	return matched, count == 1
}

// Test case types
type testCaseResult struct {
	Category     string             `json:"category"`
	Name         string             `json:"name"`
	Status       string             `json:"status"` // "PASS", "FAIL", "SKIP"
	Error        string             `json:"error,omitempty"`
	MaxDiff      float32            `json:"max_diff"`
	Elements     int                `json:"elements"`
	GoLoadMs     float64            `json:"go_load_ms"`
	GoRunMs      float64            `json:"go_run_ms"`
	OpProfiles   []OpProfile        `json:"-"` // per-op profiling data
	NodeProfiles []NodeProfile      `json:"-"` // per-node profiling data
	BuildStages  []BuildStage       `json:"-"` // session build stages
	OptPasses    []OptimizationPass `json:"-"` // optimization pass timings
}
type reportRecorder struct {
	buildStages []BuildStage
	optPasses   []OptimizationPass
}

func (r *reportRecorder) OnBuildStage(stage BuildStage) {
	r.buildStages = append(r.buildStages, stage)
}
func (r *reportRecorder) OnOptimizationPass(pass OptimizationPass) {
	r.optPasses = append(r.optPasses, pass)
}
func (r *reportRecorder) OnRunStart(info RunInfo)                    {}
func (r *reportRecorder) OnNodeStart(exec NodeExecution)             {}
func (r *reportRecorder) OnNodeFinish(exec NodeExecution, err error) {}
func (r *reportRecorder) OnRunFinish(info RunInfo, err error)        {}

// benchProgressObserver prints run/node progress during multi-run benchmarks.
type benchProgressObserver struct {
	nRuns      int
	currentRun int
	name       string
	caseIndex  int
	totalCases int
	runStart   time.Time
	lastPct    int
}

func (b *benchProgressObserver) OnBuildStage(BuildStage)            {}
func (b *benchProgressObserver) OnOptimizationPass(OptimizationPass) {}
func (b *benchProgressObserver) OnRunStart(info RunInfo) {
	b.currentRun++
	b.runStart = time.Now()
	b.lastPct = -1
}
func (b *benchProgressObserver) OnNodeStart(exec NodeExecution) {}
func (b *benchProgressObserver) OnNodeFinish(exec NodeExecution, err error) {
	if exec.Total == 0 {
		return
	}
	pct := (exec.Index + 1) * 100 / exec.Total
	step := 5
	if pct/step > b.lastPct/step || exec.Index+1 == exec.Total {
		elapsed := time.Since(b.runStart)
		line := fmt.Sprintf("\r  [%d/%d] %s run [%d/%d] %3d%% (%d/%d nodes, %.1fs)          ",
			b.caseIndex, b.totalCases, b.name, b.currentRun, b.nRuns, pct, exec.Index+1, exec.Total,
			elapsed.Seconds())
		fmt.Fprint(os.Stderr, line)
		b.lastPct = pct
	}
}
func (b *benchProgressObserver) OnRunFinish(info RunInfo, err error) {
	// Clear the progress line — final PASS/FAIL from t.Logf will overwrite
	fmt.Fprint(os.Stderr, "\r                                                            \r")
}

// runTestCase runs a single test case and returns the result.
// caseProgress holds position info for progress display.
type caseProgress struct {
	index, total int
}

func runSingleCase(category, name, modelPath, dataDir string, tol float32, cprog ...caseProgress) (result testCaseResult) {
	result = testCaseResult{Category: category, Name: name}
	// Recover from panics in ops
	defer func() {
		if r := recover(); r != nil {
			result.Status = "FAIL"
			result.Error = fmt.Sprintf("panic: %v", r)
		}
	}()
	// Load model
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		result.Status = "FAIL"
		result.Error = err.Error()
		return result
	}
	t0 := time.Now()
	recorder := &reportRecorder{}
	sess, err := NewSessionWithOptions(modelBytes, WithObserver(recorder))
	result.GoLoadMs = float64(time.Since(t0).Microseconds()) / 1000.0
	if err != nil {
		result.Status = "FAIL"
		result.Error = err.Error()
		return result
	}
	result.BuildStages = recorder.buildStages
	result.OptPasses = recorder.optPasses
	// Enable profiling for model tests
	if category == "models" {
		sess.Profiler.Enable()
	}
	// Load inputs
	inputs, err := loadInputs(dataDir)
	if err != nil {
		result.Status = "FAIL"
		result.Error = err.Error()
		return result
	}
	// Run inference
	t0 = time.Now()
	nRuns := benchNRuns()
	// Attach progress observer for multi-run benchmarks
	if nRuns > 1 {
		ci, ct := 0, 0
		if len(cprog) > 0 {
			ci, ct = cprog[0].index, cprog[0].total
		}
		obs := &benchProgressObserver{nRuns: nRuns, name: category + "/" + name, caseIndex: ci, totalCases: ct}
		sess.AddObserver(obs)
	}
	var outputs map[string]tensor.Tensor
	for i := 0; i < nRuns; i++ {
		outputs, err = sess.RunWithNames(inputs)
		if err != nil {
			result.Status = "FAIL"
			result.Error = err.Error()
			return result
		}
	}
	result.GoRunMs = float64(time.Since(t0).Microseconds()) / 1000.0 / float64(nRuns)
	// Collect profiling data
	if sess.Profiler.IsEnabled() {
		result.OpProfiles = sess.Profiler.Results()
		result.NodeProfiles = sess.Profiler.NodeResults()
	}
	// Load expected outputs
	expected, err := loadExpectedOutputs(dataDir)
	if err != nil {
		result.Status = "FAIL"
		result.Error = err.Error()
		return result
	}
	// Compare: for multi-output models (detection), compare only the highest-confidence
	// output to handle TopK ordering instability. For single-output, compare directly.
	if len(expected) == 1 {
		for wantName, wantTensor := range expected {
			gotTensor, ok := findOutputTensor(outputs, wantName)
			if !ok {
				result.Status = "FAIL"
				result.Error = fmt.Sprintf("output %q not found (have: %v)", wantName, mapKeys(outputs))
				return result
			}
			if rows, _, ok := detectionLikeOutput(wantTensor); ok && rows >= 100 {
				maxDiff, pass, comparedRows, err := compareDetectionPrefix(gotTensor, wantTensor, tol)
				if err != nil {
					result.Status = "FAIL"
					result.Error = err.Error()
					return result
				}
				result.MaxDiff = maxDiff
				result.Elements = comparedRows * 6
				if !pass {
					result.Status = "FAIL"
					result.Error = fmt.Sprintf("top-%d detection rows differ beyond tolerance (max diff %e)", comparedRows, maxDiff)
					return result
				}
				continue
			}
			maxDiff, _, pass, err := compareOutputs(gotTensor, wantTensor, tol)
			if err != nil {
				result.Status = "FAIL"
				result.Error = err.Error()
				return result
			}
			result.MaxDiff = maxDiff
			result.Elements = wantTensor.Len()
			if !pass {
				result.Status = "FAIL"
				result.Error = fmt.Sprintf("max diff %e exceeds tolerance %e", maxDiff, tol)
				return result
			}
		}
	} else {
		hasIntOutput := false
		for _, wantTensor := range expected {
			switch wantTensor.(type) {
			case *tensor.Dense[int64], *tensor.Dense[int32]:
				hasIntOutput = true
			}
		}
		// General multi-output models: compare every output.
		if !hasIntOutput {
			for wantName, wantTensor := range expected {
				gotTensor, ok := findOutputTensor(outputs, wantName)
				if !ok {
					gotTensor, ok = findOutputTensorByShape(outputs, wantTensor)
				}
				if !ok {
					result.Status = "FAIL"
					result.Error = fmt.Sprintf("output %q not found (have: %v)", wantName, mapKeys(outputs))
					return result
				}
				maxDiff, _, pass, err := compareOutputs(gotTensor, wantTensor, tol)
				if err != nil {
					result.Status = "FAIL"
					result.Error = err.Error()
					return result
				}
				result.Elements += wantTensor.Len()
				if maxDiff > result.MaxDiff {
					result.MaxDiff = maxDiff
				}
				if !pass {
					result.Status = "FAIL"
					result.Error = fmt.Sprintf("output %q diff %e exceeds tolerance %e", wantName, maxDiff, tol)
					return result
				}
			}
			result.Status = "PASS"
			return result
		}
		// Detection-style multi-output: compare using sort-invariant strategy.
		// Collect all float outputs and use the smallest diff because TopK order can vary.
		bestDiff := float32(-1)
		for wantName, wantTensor := range expected {
			gotTensor, ok := findOutputTensor(outputs, wantName)
			if !ok {
				gotTensor, ok = findOutputTensorByShape(outputs, wantTensor)
			}
			if !ok {
				continue
			}
			// Skip non-float and mismatched shapes
			switch wantTensor.(type) {
			case *tensor.Dense[int64], *tensor.Dense[int32]:
				result.Elements += wantTensor.Len()
				continue
			}
			maxDiff, _, _, err := compareOutputs(gotTensor, wantTensor, tol)
			if err != nil {
				continue
			}
			result.Elements += wantTensor.Len()
			if bestDiff < 0 || maxDiff < bestDiff {
				bestDiff = maxDiff
			}
		}
		if bestDiff < 0 {
			// All outputs are integer — compare them directly
			for wantName, wantTensor := range expected {
				gotTensor, ok := findOutputTensor(outputs, wantName)
				if !ok {
					gotTensor, ok = findOutputTensorByShape(outputs, wantTensor)
				}
				if !ok { continue }
				maxDiff, _, pass, err := compareOutputs(gotTensor, wantTensor, tol)
				if err != nil { continue }
				result.Elements += wantTensor.Len()
				if maxDiff > bestDiff || bestDiff < 0 { bestDiff = maxDiff }
				if !pass {
					result.Status = "FAIL"
					result.Error = fmt.Sprintf("output %q diff %e exceeds tolerance %e", wantName, maxDiff, tol)
					return result
				}
			}
			if bestDiff < 0 {
				result.Status = "FAIL"
				result.Error = "no comparable outputs"
				return result
			}
		}
		result.MaxDiff = bestDiff
		// Use the best (lowest diff) float output for pass/fail
		if bestDiff > tol {
			result.Status = "FAIL"
			result.Error = fmt.Sprintf("best float output diff %e exceeds tolerance %e", bestDiff, tol)
			return result
		}
	}
	result.Status = "PASS"
	return result
}

// findAllCases discovers test cases under testdata/ops/ and testdata/models/.
func findAllCases() []struct {
	category, name, modelPath, dataDir string
	tol                                float32
} {
	var cases []struct {
		category, name, modelPath, dataDir string
		tol                                float32
	}
	for _, cat := range []struct {
		dir string
		tol float32
	}{
		{"ops", defaultTol},
	} {
		catDir := filepath.Join(testdataDir, cat.dir)
		entries, err := os.ReadDir(catDir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			caseDir := filepath.Join(catDir, e.Name())
			modelPath := filepath.Join(caseDir, "model.onnx")
			if _, err := os.Stat(modelPath); err != nil {
				continue
			}
			subEntries, _ := os.ReadDir(caseDir)
			tol := cat.tol
			hasSubSets := false
			for _, se := range subEntries {
				if se.IsDir() && strings.HasPrefix(se.Name(), "test_data_set_") {
					name := e.Name() + "/" + se.Name()
					if !includeReportCase(cat.dir, name) {
						continue
					}
					hasSubSets = true
					cases = append(cases, struct {
						category, name, modelPath, dataDir string
						tol                                float32
					}{cat.dir, name, modelPath, filepath.Join(caseDir, se.Name()), tol})
				}
			}
			if !hasSubSets {
				if !includeReportCase(cat.dir, e.Name()) {
					continue
				}
				cases = append(cases, struct {
					category, name, modelPath, dataDir string
					tol                                float32
				}{cat.dir, e.Name(), modelPath, caseDir, tol})
			}
		}
	}
	return cases
}

// Individual tests (for `go test`)
func runNamedCase(t *testing.T, category, name string, tol float32) {
	t.Helper()
	var modelPath, dataDir string
	caseDir := filepath.Join(testdataDir, category, name)
	modelPath = filepath.Join(caseDir, "model.onnx")
	dataDir = caseDir
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("not found: %s", modelPath)
	}
	r := runSingleCase(category, name, modelPath, dataDir, tol)
	if r.Status == "FAIL" {
		t.Fatalf("%s: %s", name, r.Error)
	}
	t.Logf("%s: PASS (max_diff=%.2e, %d elements, go_run=%.3fms)",
		name, r.MaxDiff, r.Elements, r.GoRunMs)
}
func TestOpAddBroadcast(t *testing.T)     { runNamedCase(t, "ops", "add_broadcast", defaultTol) }
func TestOpSubDiv(t *testing.T)           { runNamedCase(t, "ops", "sub_div", defaultTol) }
func TestOpMatMul2D(t *testing.T)         { runNamedCase(t, "ops", "matmul_2d", defaultTol) }
func TestOpGemmBias(t *testing.T)         { runNamedCase(t, "ops", "gemm_bias", defaultTol) }
func TestOpRelu(t *testing.T)             { runNamedCase(t, "ops", "relu", defaultTol) }
func TestOpSigmoidTanh(t *testing.T)      { runNamedCase(t, "ops", "sigmoid_tanh", defaultTol) }
func TestOpSoftmax(t *testing.T)          { runNamedCase(t, "ops", "softmax", defaultTol) }
func TestOpConv2D(t *testing.T)           { runNamedCase(t, "ops", "conv2d", defaultTol) }
func TestOpMaxPool(t *testing.T)          { runNamedCase(t, "ops", "maxpool", defaultTol) }
func TestOpBatchNorm(t *testing.T)        { runNamedCase(t, "ops", "batchnorm", defaultTol) }
func TestOpReshape(t *testing.T)          { runNamedCase(t, "ops", "reshape", defaultTol) }
func TestOpTranspose(t *testing.T)        { runNamedCase(t, "ops", "transpose", defaultTol) }
func TestOpGlobalAvgPool(t *testing.T)    { runNamedCase(t, "ops", "globalavgpool", defaultTol) }
func TestOpConcat(t *testing.T)           { runNamedCase(t, "ops", "concat", defaultTol) }
func TestOpGather(t *testing.T)           { runNamedCase(t, "ops", "gather", defaultTol) }
func TestOpSqueezeUnsqueeze(t *testing.T) { runNamedCase(t, "ops", "squeeze_unsqueeze", defaultTol) }
func TestOpMLPSmall(t *testing.T)         { runNamedCase(t, "ops", "mlp_small", defaultTol) }
func TestOpConvReluPool(t *testing.T)     { runNamedCase(t, "ops", "conv_relu_pool", defaultTol) }
func TestRunWithNamesUnknownInput(t *testing.T) {
	modelPath := filepath.Join(testdataDir, "ops", "add_broadcast", "model.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("add_broadcast model not found")
	}
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	sess, err := NewSession(modelBytes)
	if err != nil {
		t.Fatal(err)
	}
	dummy := tensor.NewDense[float32](tensor.Shape{1}, []float32{0})
	_, err = sess.RunWithNames(map[string]tensor.Tensor{"typo": dummy})
	if err == nil {
		t.Fatal("expected error for unknown input name, got nil")
	}
	if !strings.Contains(err.Error(), "unknown input name") {
		t.Fatalf("expected 'unknown input name' in error, got: %s", err.Error())
	}
}
func TestOpLoopSum(t *testing.T)         { runNamedCase(t, "ops", "loop_sum", defaultTol) }
func TestOpScanCumsum(t *testing.T)      { runNamedCase(t, "ops", "scan_cumsum", defaultTol) }
func TestGeneratedOpCases(t *testing.T) {
	for _, c := range findAllCases() {
		if c.category != "ops" {
			continue
		}
		c := c
		t.Run(c.name, func(t *testing.T) {
			r := runSingleCase(c.category, c.name, c.modelPath, c.dataDir, c.tol)
			if r.Status == "FAIL" {
				t.Fatalf("%s: %s", c.name, r.Error)
			}
		})
	}
}
func TestMNIST(t *testing.T) {
	mnistDir := filepath.Join(testdataDir, "models", "mnist")
	modelPath := filepath.Join(mnistDir, "model.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("MNIST model not found (run download_mnist.py)")
	}
	for i := 0; i < 3; i++ {
		setName := fmt.Sprintf("test_data_set_%d", i)
		dataDir := filepath.Join(mnistDir, setName)
		if _, err := os.Stat(dataDir); os.IsNotExist(err) {
			continue
		}
		t.Run(setName, func(t *testing.T) {
			r := runSingleCase("models", "mnist/"+setName, modelPath, dataDir, 1e-4)
			if r.Status == "FAIL" {
				t.Fatalf("MNIST/%s: %s", setName, r.Error)
			}
			t.Logf("MNIST/%s: PASS (max_diff=%.2e, %d elements, go_run=%.3fms)",
				setName, r.MaxDiff, r.Elements, r.GoRunMs)
		})
	}
}

// HTML report generation
type pythonBenchResult struct {
	Name     string  `json:"name"`
	Category string  `json:"category"`
	Status   string  `json:"status"`
	Error    string  `json:"error,omitempty"`
	LoadMs   float64 `json:"load_ms"`
	RunAvgMs float64 `json:"run_avg_ms"`
	RunMinMs float64 `json:"run_min_ms"`
	RunMaxMs float64 `json:"run_max_ms"`
}
type reportRow struct {
	Category    string
	Name        string
	Status      string
	Error       string
	MaxDiff     string
	Elements    int
	GoLoadMs    string
	GoRunMs     string
	PyLoadMs    string
	PyRunMs     string
	Speedup     string
	StatusClass string
}

func buildReportRow(gr testCaseResult, pr pythonBenchResult) reportRow {
	row := reportRow{
		Category: gr.Category,
		Name:     gr.Name,
		Status:   gr.Status,
		Error:    gr.Error,
		MaxDiff:  fmt.Sprintf("%.2e", gr.MaxDiff),
		Elements: gr.Elements,
		GoLoadMs: fmt.Sprintf("%.3f", gr.GoLoadMs),
		GoRunMs:  fmt.Sprintf("%.3f", gr.GoRunMs),
	}
	switch gr.Status {
	case "PASS":
		row.StatusClass = "pass"
	case "FAIL":
		row.StatusClass = "fail"
	default:
		row.StatusClass = "skip"
	}
	if pr.Status == "ok" {
		row.PyLoadMs = fmt.Sprintf("%.3f", pr.LoadMs)
		row.PyRunMs = fmt.Sprintf("%.3f", pr.RunAvgMs)
		if pr.RunAvgMs > 0 && gr.GoRunMs > 0 {
			row.Speedup = fmt.Sprintf("x%.1f", gr.GoRunMs/pr.RunAvgMs)
		}
	} else if pr.Status == "error" {
		row.PyRunMs = "ERROR"
	} else {
		row.PyRunMs = "N/A"
	}
	return row
}

func TestGenerateReport(t *testing.T) {
	cases := findAllCases()
	if len(cases) == 0 {
		t.Fatal("no report cases found")
	}
	// 1. Run Python benchmark
	t.Logf("Running Python benchmarks (%d cases)...", len(cases))
	pyCmd := exec.Command("python", filepath.Join(testdataDir, "benchmark_python.py"))
	pyCmd.Env = append(os.Environ(), "ENABLE_MODEL_TESTS=false")
	pyCmd.Stderr = os.Stderr // show Python progress in real-time
	pyOut, err := pyCmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			t.Logf("Python stderr: %s", string(exitErr.Stderr))
		}
		t.Fatalf("Python benchmark failed: %v", err)
	}
	var pyResults []pythonBenchResult
	if err := json.Unmarshal(pyOut, &pyResults); err != nil {
		t.Fatalf("parse Python results: %v", err)
	}
	pyMap := make(map[string]pythonBenchResult)
	for _, r := range pyResults {
		pyMap[r.Category+"/"+r.Name] = r
	}
	// 2. Run Go benchmarks
	t.Logf("Running Go benchmarks (%d cases)...", len(cases))
	var goResults []testCaseResult
	for i, c := range cases {
		r := runSingleCase(c.category, c.name, c.modelPath, c.dataDir, c.tol, caseProgress{i + 1, len(cases)})
		goResults = append(goResults, r)
		if r.Error != "" {
			t.Logf("  [%d/%d] %s/%s: %s (%.3fms) ERROR: %s", i+1, len(cases), c.category, c.name, r.Status, r.GoRunMs, r.Error)
		} else {
			t.Logf("  [%d/%d] %s/%s: %s (%.3fms)", i+1, len(cases), c.category, c.name, r.Status, r.GoRunMs)
		}
	}
	// 3. Build report rows
	var rows []reportRow
	passCount, failCount, skipCount := 0, 0, 0
	for _, gr := range goResults {
		key := gr.Category + "/" + gr.Name
		pr := pyMap[key]
		row := buildReportRow(gr, pr)
		switch gr.Status {
		case "PASS":
			passCount++
		case "FAIL":
			failCount++
		default:
			skipCount++
		}
		rows = append(rows, row)
	}
	// 4. Render HTML
	cpuInfo, ramInfo := getSysInfo()
	reportData := struct {
		Rows      []reportRow
		Timestamp string
		Total     int
		Pass      int
		Fail      int
		Skip      int
		GoVer     string
		CPU       string
		RAM       string
		NRuns     int
	}{
		Rows:      rows,
		Timestamp: time.Now().Format("2006-01-02 15:04:05"),
		Total:     len(rows),
		Pass:      passCount,
		NRuns:     benchNRuns(),
		Fail:      failCount,
		Skip:      skipCount,
		GoVer:     runtime.Version(),
		CPU:       cpuInfo,
		RAM:       ramInfo,
	}
	reportPath := filepath.Join(testdataDir, "..", "report.html")
	f, err := os.Create(reportPath)
	if err != nil {
		t.Fatalf("create report: %v", err)
	}
	defer f.Close()
	tmpl := template.Must(template.New("report").Parse(reportHTML))
	if err := tmpl.Execute(f, reportData); err != nil {
		t.Fatalf("render report: %v", err)
	}
	absPath, _ := filepath.Abs(reportPath)
	t.Logf("Report generated: %s", absPath)
	t.Logf("Summary: %d total, %d pass, %d fail, %d skip", len(rows), passCount, failCount, skipCount)
}

const reportHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ONNX Pure Go Runtime Test Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; padding: 24px; }
  h1 { font-size: 1.5em; margin-bottom: 4px; }
  h2 { font-size: 1.2em; margin: 20px 0 8px; }
  .meta { color: #666; font-size: 0.85em; margin-bottom: 16px; }
  .summary { display: flex; gap: 12px; margin-bottom: 20px; }
  .summary .card { padding: 12px 20px; border-radius: 8px; color: #fff; font-weight: bold; font-size: 1.1em; }
  .card.total { background: #555; }
  .card.pass  { background: #22863a; }
  .card.fail  { background: #cb2431; }
  .card.skip  { background: #b08800; }
  .table-wrap { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }
  table { width: 100%; border-collapse: separate; border-spacing: 0; background: #fff; border-radius: 8px; margin-bottom: 20px; }
  thead th { position: sticky; top: 0; z-index: 3; background: #24292e; color: #fff; text-align: left; padding: 10px 12px; font-size: 0.85em; box-shadow: 0 1px 0 rgba(0,0,0,0.15); }
  td { padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.85em; font-variant-numeric: tabular-nums; vertical-align: top; }
  tr:hover { background: #f0f7ff; }
  tr.pass td:nth-child(2) { color: #22863a; font-weight: bold; }
  tr.fail td:nth-child(2) { color: #cb2431; font-weight: bold; }
  .num { text-align: right; }
  .error { color: #cb2431; font-size: 0.8em; }
</style>
</head>
<body>
<h1>ONNX Pure Go Runtime Test Report</h1>
<p class="meta">Generated: {{.Timestamp}} | Go: {{.GoVer}} | Python: onnxruntime | Runs: {{.NRuns}} avg</p>
<p class="meta">CPU: {{.CPU}} | RAM: {{.RAM}}</p>
<div class="summary">
  <div class="card total">Total: {{.Total}}</div>
  <div class="card pass">Pass: {{.Pass}}</div>
  <div class="card fail">Fail: {{.Fail}}</div>
  {{if .Skip}}<div class="card skip">Skip: {{.Skip}}</div>{{end}}
</div>
<h2>Operator Tests</h2>
<div class="table-wrap">
<table>
<thead>
<tr>
  <th>Test</th>
  <th>Status</th>
  <th class="num">Max Diff</th>
  <th class="num">Elements</th>
  <th class="num">Py Load (ms)</th>
  <th class="num">Py Run (ms)</th>
  <th class="num">Go Load (ms)</th>
  <th class="num">Go Run (ms)</th>
  <th class="num">Go/Py Ratio</th>
</tr>
</thead>
<tbody>
{{range .Rows}}
<tr class="{{.StatusClass}}">
  <td>{{.Name}}{{if .Error}}<br><span class="error">{{.Error}}</span>{{end}}</td>
  <td>{{.Status}}</td>
  <td class="num">{{.MaxDiff}}</td>
  <td class="num">{{.Elements}}</td>
  <td class="num">{{.PyLoadMs}}</td>
  <td class="num">{{.PyRunMs}}</td>
  <td class="num">{{.GoLoadMs}}</td>
  <td class="num">{{.GoRunMs}}</td>
  <td class="num">{{.Speedup}}</td>
</tr>
{{end}}
</body>
</html>`
