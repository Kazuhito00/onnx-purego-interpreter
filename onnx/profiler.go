package onnx

import (
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

// OpProfile holds per-op profiling statistics.
// op 種別ごとのプロファイリング統計を保持する。
type OpProfile struct {
	OpType     string
	Calls      int
	TotalNs    int64
	AllocBytes int64
}

// NodeProfile holds per-node profiling statistics.
// ノード単位のプロファイリング統計を保持する。
type NodeProfile struct {
	NodeName   string
	OpType     string
	Calls      int
	TotalNs    int64
	AllocBytes int64
}

// Profiler collects per-op execution statistics during Session.Run.
// Session.Run 中の op 単位の実行統計を収集する。
type Profiler struct {
	mu      sync.Mutex
	ops     map[string]*OpProfile
	nodes   map[string]*NodeProfile
	enabled bool
}

// NewProfiler creates a new disabled profiler.
// 無効状態の新しい Profiler を作成する。
func NewProfiler() *Profiler {
	return &Profiler{ops: make(map[string]*OpProfile), nodes: make(map[string]*NodeProfile)}
}

// Enable turns profiling on.
// プロファイリングを有効化する。
func (p *Profiler) Enable() { p.enabled = true }

// Disable turns profiling off.
// プロファイリングを無効化する。
func (p *Profiler) Disable() { p.enabled = false }

// IsEnabled returns whether profiling is active.
// プロファイリングが有効かどうかを返す。
func (p *Profiler) IsEnabled() bool { return p.enabled }

// Reset clears all collected data.
// 収集済みの全データをクリアする。
func (p *Profiler) Reset() {
	p.mu.Lock()
	p.ops = make(map[string]*OpProfile)
	p.nodes = make(map[string]*NodeProfile)
	p.mu.Unlock()
}

// Record records a single op execution.
// 1 回の op 実行を記録する。
func (p *Profiler) Record(opType string, elapsed time.Duration, allocBytes int64) {
	if !p.enabled {
		return
	}
	p.mu.Lock()
	prof, ok := p.ops[opType]
	if !ok {
		prof = &OpProfile{OpType: opType}
		p.ops[opType] = prof
	}
	prof.Calls++
	prof.TotalNs += elapsed.Nanoseconds()
	prof.AllocBytes += allocBytes
	p.mu.Unlock()
}

func (p *Profiler) OnRunStart(info RunInfo) {}

func (p *Profiler) OnBuildStage(stage BuildStage) {}

func (p *Profiler) OnOptimizationPass(pass OptimizationPass) {}

func (p *Profiler) OnNodeStart(exec NodeExecution) {}

func (p *Profiler) OnNodeFinish(exec NodeExecution, err error) {
	if err != nil {
		return
	}
	p.Record(exec.Node.OpType, exec.Elapsed, exec.AllocBytes)
	if !p.enabled {
		return
	}
	p.mu.Lock()
	key := exec.Node.OpType + "::" + exec.Node.Name
	prof, ok := p.nodes[key]
	if !ok {
		prof = &NodeProfile{NodeName: exec.Node.Name, OpType: exec.Node.OpType}
		p.nodes[key] = prof
	}
	prof.Calls++
	prof.TotalNs += exec.Elapsed.Nanoseconds()
	prof.AllocBytes += exec.AllocBytes
	p.mu.Unlock()
}

func (p *Profiler) OnRunFinish(info RunInfo, err error) {}

// Results returns sorted profiling results (by total time, descending).
func (p *Profiler) Results() []OpProfile {
	p.mu.Lock()
	defer p.mu.Unlock()
	results := make([]OpProfile, 0, len(p.ops))
	for _, v := range p.ops {
		results = append(results, *v)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].TotalNs > results[j].TotalNs
	})
	return results
}

// NodeResults returns sorted per-node profiling results.
func (p *Profiler) NodeResults() []NodeProfile {
	p.mu.Lock()
	defer p.mu.Unlock()
	results := make([]NodeProfile, 0, len(p.nodes))
	for _, v := range p.nodes {
		results = append(results, *v)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].TotalNs > results[j].TotalNs
	})
	return results
}

// TotalNs returns total profiled time.
func (p *Profiler) TotalNs() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	var total int64
	for _, v := range p.ops {
		total += v.TotalNs
	}
	return total
}

// Summary returns a human-readable summary string.
func (p *Profiler) Summary() string {
	results := p.Results()
	total := p.TotalNs()
	if total == 0 {
		return "No profiling data"
	}
	s := fmt.Sprintf("Op Profiling Summary (total: %.1fms)\n", float64(total)/1e6)
	s += fmt.Sprintf("%-25s %8s %10s %10s %10s\n", "Op", "Calls", "Total(ms)", "Avg(ms)", "Alloc(KB)")
	s += "─────────────────────────────────────────────────────────────────────────\n"
	for _, r := range results {
		pct := float64(r.TotalNs) / float64(total) * 100
		s += fmt.Sprintf("%-25s %8d %9.1f %9.3f %9.1f  (%.1f%%)\n",
			r.OpType, r.Calls,
			float64(r.TotalNs)/1e6,
			float64(r.TotalNs)/float64(r.Calls)/1e6,
			float64(r.AllocBytes)/1024,
			pct)
	}
	return s
}

// getAllocBytes returns bytes allocated by the current goroutine since start.
// Uses runtime.ReadMemStats for approximate measurement.
func getAllocBytes() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.TotalAlloc)
}
