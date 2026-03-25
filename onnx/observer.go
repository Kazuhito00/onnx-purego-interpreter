package onnx

import (
	"fmt"
	"io"
	"time"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
)

// RunInfo describes a single inference run.
// 1 回の推論実行を記述する。
type RunInfo struct {
	TotalNodes int
	Compiled   bool
	StartedAt  time.Time
}

// BuildStage describes a session construction phase.
// セッション構築の各フェーズを記述する。
type BuildStage struct {
	Name      string
	StartedAt time.Time
	Elapsed   time.Duration
	Detail    string
}

// OptimizationPass describes a single optimization pass execution.
// 1 つの最適化パスの実行結果を記述する。
type OptimizationPass struct {
	Name        string
	BeforeNodes int
	AfterNodes  int
	Elapsed     time.Duration
}

// NodeExecution describes one node execution within a run.
// 推論実行中の 1 ノードの実行情報を記述する。
type NodeExecution struct {
	Node       *ir.Node
	Index      int
	Total      int
	Elapsed    time.Duration
	AllocBytes int64
}

// Observer receives inference lifecycle events.
// 推論ライフサイクルイベントを受信するインターフェース。
type Observer interface {
	OnBuildStage(stage BuildStage)
	OnOptimizationPass(pass OptimizationPass)
	OnRunStart(info RunInfo)
	OnNodeStart(exec NodeExecution)
	OnNodeFinish(exec NodeExecution, err error)
	OnRunFinish(info RunInfo, err error)
}

type observerList []Observer

func (l observerList) onRunStart(info RunInfo) {
	for _, o := range l {
		o.OnRunStart(info)
	}
}

func (l observerList) onBuildStage(stage BuildStage) {
	for _, o := range l {
		o.OnBuildStage(stage)
	}
}

func (l observerList) onOptimizationPass(pass OptimizationPass) {
	for _, o := range l {
		o.OnOptimizationPass(pass)
	}
}

func (l observerList) onNodeStart(exec NodeExecution) {
	for _, o := range l {
		o.OnNodeStart(exec)
	}
}

func (l observerList) onNodeFinish(exec NodeExecution, err error) {
	for _, o := range l {
		o.OnNodeFinish(exec, err)
	}
}

func (l observerList) onRunFinish(info RunInfo, err error) {
	for _, o := range l {
		o.OnRunFinish(info, err)
	}
}

// ProgressLogger emits simple per-node progress logs.
// ノード単位の簡易プログレスログを出力する Observer 実装。
type ProgressLogger struct {
	Writer io.Writer
}

func (p *ProgressLogger) OnBuildStage(stage BuildStage) {
	if p == nil || p.Writer == nil {
		return
	}
	if stage.Elapsed > 0 {
		fmt.Fprintf(p.Writer, "build stage=%s elapsed=%s detail=%s\n", stage.Name, stage.Elapsed, stage.Detail)
		return
	}
	fmt.Fprintf(p.Writer, "build stage=%s detail=%s\n", stage.Name, stage.Detail)
}

func (p *ProgressLogger) OnOptimizationPass(pass OptimizationPass) {
	if p == nil || p.Writer == nil {
		return
	}
	fmt.Fprintf(
		p.Writer,
		"opt pass=%s before=%d after=%d elapsed=%s\n",
		pass.Name, pass.BeforeNodes, pass.AfterNodes, pass.Elapsed,
	)
}

func (p *ProgressLogger) OnRunStart(info RunInfo) {
	if p == nil || p.Writer == nil {
		return
	}
	fmt.Fprintf(p.Writer, "run start: nodes=%d compiled=%t\n", info.TotalNodes, info.Compiled)
}

func (p *ProgressLogger) OnNodeStart(exec NodeExecution) {}

func (p *ProgressLogger) OnNodeFinish(exec NodeExecution, err error) {
	if p == nil || p.Writer == nil {
		return
	}
	status := "ok"
	if err != nil {
		status = "error"
	}
	fmt.Fprintf(
		p.Writer,
		"progress %d/%d op=%s name=%s elapsed=%s alloc=%dB status=%s\n",
		exec.Index+1, exec.Total, exec.Node.OpType, exec.Node.Name, exec.Elapsed, exec.AllocBytes, status,
	)
}

func (p *ProgressLogger) OnRunFinish(info RunInfo, err error) {
	if p == nil || p.Writer == nil {
		return
	}
	status := "ok"
	if err != nil {
		status = err.Error()
	}
	fmt.Fprintf(p.Writer, "run finish: status=%s\n", status)
}
