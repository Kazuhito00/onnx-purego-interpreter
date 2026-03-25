// Package onnx provides the public API for loading and running ONNX models.
// ONNX モデルの読み込みと推論実行のための公開 API を提供する。
//
// A typical workflow is:
// 基本的な使い方:
//
//	sess, err := onnx.NewSession(modelBytes)
//	outputs, err := sess.Run(inputTensor)
//
// The package handles the full pipeline: protobuf decoding, IR construction,
// graph optimization, lowering to a compiled execution plan, and runtime
// inference using pure Go kernels.
// protobuf デコードから IR 構築、グラフ最適化、コンパイル済み実行プランへの lowering、
// Pure Go カーネルによる推論実行までの全パイプラインを処理する。
package onnx

import (
	"fmt"
	"io"
	"os"
	"time"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/analysis"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/frontend"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/lowering"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ops"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/optimize"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/reader"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// Session holds a compiled ONNX model ready for execution.
// A Session is NOT safe for concurrent use by multiple goroutines.
// The internal arena is reused across runs — create separate sessions for parallel inference.
// コンパイル済み ONNX モデルを保持し、推論実行に使用する。
// Session は複数 goroutine からの並列使用に対して安全ではない。
// 内部 arena は実行間で使い回される — 並列推論には別セッションを作成すること。
type Session struct {
	frontendModel *frontend.Model
	graph         *ir.Graph
	order         []*ir.Node
	registry      *ops.Registry
	initializers  map[string]tensor.Tensor // pre-materialized
	plan          *lowering.Plan           // pre-compiled execution plan
	arena         *lowering.Arena          // pre-allocated execution arena
	hasSubgraphs  bool                     // true if model contains If/Loop nodes
	Profiler      *Profiler
	observers     observerList
	warnings      []string // opset compatibility warnings
}

// Warnings returns opset compatibility warnings detected during session creation.
// セッション作成時に検出された opset 互換性の警告を返す。
func (s *Session) Warnings() []string { return s.warnings }

// NewSession creates a session from raw .onnx protobuf bytes.
// 生の .onnx protobuf バイト列からセッションを作成する。
func NewSession(onnxBytes []byte) (*Session, error) {
	return NewSessionWithOptions(onnxBytes)
}

// NewSessionWithOptions creates a session with options and enables build/run observers from the start.
// オプション付きでセッションを作成し、ビルド/推論の Observer を最初から有効化する。
func NewSessionWithOptions(onnxBytes []byte, opts ...SessionOption) (*Session, error) {
	cfg := sessionOptions{}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	profiler := NewProfiler()
	observers := observerList{profiler}
	observers = append(observers, cfg.observers...)

	emitStage := func(name string, started time.Time, detail string) {
		observers.onBuildStage(BuildStage{
			Name:      name,
			StartedAt: started,
			Elapsed:   time.Since(started),
			Detail:    detail,
		})
	}

	t0 := time.Now()
	m, err := reader.DecodeBytes(onnxBytes)
	if err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}
	emitStage("reader.decode", t0, "")

	t0 = time.Now()
	fm, err := frontend.Build(m)
	if err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}
	emitStage("frontend.build", t0, fmt.Sprintf("nodes=%d", len(fm.Graph.Nodes)))

	t0 = time.Now()
	g, err := ir.FromFrontend(fm)
	if err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}
	emitStage("ir.canonicalize", t0, fmt.Sprintf("nodes=%d", len(g.Nodes)))

	t0 = time.Now()
	if err := analysis.ValidateGraph(g); err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}
	opsetWarnings := analysis.ValidateOpsetCompatibility(g)
	emitStage("analysis.validate.pre", t0, "")

	t0 = time.Now()
	disabled := buildDisabledPasses(cfg)
	for _, pass := range optimize.Apply(g, disabled) {
		name := pass.Name
		if pass.Disabled {
			name += " (disabled)"
		}
		observers.onOptimizationPass(OptimizationPass{
			Name:        name,
			BeforeNodes: pass.BeforeNodes,
			AfterNodes:  pass.AfterNodes,
			Elapsed:     pass.Elapsed,
		})
	}
	emitStage("optimize.apply", t0, fmt.Sprintf("nodes=%d", len(g.Nodes)))

	t0 = time.Now()
	if err := analysis.ValidateGraph(g); err != nil {
		return nil, fmt.Errorf("engine: post-optimize validation failed: %w", err)
	}
	emitStage("analysis.validate.post", t0, "")

	reg := ops.NewRegistry()
	ops.RegisterAll(reg, cfg.kernelConfig)

	t0 = time.Now()
	build, err := lowering.Prepare(g, reg)
	if err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}
	emitStage("lowering.prepare", t0, fmt.Sprintf("order=%d instructions=%d slots=%d", len(build.Order), len(build.Plan.Instructions), build.Plan.SlotCount))

	return &Session{
		frontendModel: fm,
		graph:         g,
		order:         build.Order,
		registry:      reg,
		initializers:  build.Initializers,
		plan:          build.Plan,
		arena:         build.Arena,
		hasSubgraphs:  build.HasSubgraphs,
		Profiler:      profiler,
		observers:     observers,
		warnings:      opsetWarnings,
	}, nil
}

// buildDisabledPasses computes the set of disabled passes from session options.
func buildDisabledPasses(cfg sessionOptions) map[string]bool {
	if cfg.noOptimize {
		// Disable all passes
		all := make(map[string]bool)
		for _, name := range optimize.PassNames() {
			all[name] = true
		}
		return all
	}
	if cfg.onlyPasses != nil {
		// Disable everything except the specified passes
		disabled := make(map[string]bool)
		for _, name := range optimize.PassNames() {
			if !cfg.onlyPasses[name] {
				disabled[name] = true
			}
		}
		return disabled
	}
	return cfg.disabledPasses
}

// Graph returns the IR graph (for inspection/testing).
// IR グラフを返します（検査・テスト用）。
func (s *Session) Graph() *ir.Graph {
	return s.graph
}

// AddObserver registers an execution observer.
// 実行 Observer を登録する。
func (s *Session) AddObserver(observer Observer) {
	if observer == nil {
		return
	}
	s.observers = append(s.observers, observer)
}

// SetProgressLogger attaches a simple progress logger to the session.
// セッションに簡易プログレスロガーを接続する。
func (s *Session) SetProgressLogger(w io.Writer) {
	if w == nil {
		return
	}
	s.AddObserver(&ProgressLogger{Writer: w})
}

// WriteFrontendDOT writes the frontend ONNX-like graph as Graphviz DOT.
// フロントエンド ONNX グラフを Graphviz DOT 形式でファイルに書き出す。
func (s *Session) WriteFrontendDOT(path string) error {
	return os.WriteFile(path, []byte(s.FrontendDOT()), 0o644)
}

// WriteCanonicalDOT writes the canonical IR graph as Graphviz DOT.
// 正規化済み IR グラフを Graphviz DOT 形式でファイルに書き出す。
func (s *Session) WriteCanonicalDOT(path string) error {
	return os.WriteFile(path, []byte(s.CanonicalDOT()), 0o644)
}

// WritePlanDOT writes the lowered execution plan as Graphviz DOT.
// lowering 済み実行プランを Graphviz DOT 形式でファイルに書き出す。
func (s *Session) WritePlanDOT(path string) error {
	return os.WriteFile(path, []byte(s.PlanDOT()), 0o644)
}

// WriteFrontendJSON writes the frontend ONNX-like graph as JSON.
// フロントエンド ONNX グラフを JSON 形式でファイルに書き出す。
func (s *Session) WriteFrontendJSON(path string) error {
	data, err := s.FrontendJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// WriteCanonicalJSON writes the canonical IR graph as JSON.
// 正規化済み IR グラフを JSON 形式でファイルに書き出す。
func (s *Session) WriteCanonicalJSON(path string) error {
	data, err := s.CanonicalJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// WritePlanJSON writes the lowered execution plan as JSON.
// lowering 済み実行プランを JSON 形式でファイルに書き出す。
func (s *Session) WritePlanJSON(path string) error {
	data, err := s.PlanJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// RunDebug executes the model and returns all intermediate tensor values for debugging.
// モデルを実行し、デバッグ用に全中間テンソル値を返す。
func (s *Session) RunDebug(inputs map[string]tensor.Tensor) (map[string]tensor.Tensor, error) {
	if s.plan != nil {
		return s.runCompiledDebug(inputs)
	}

	ctx := newContext()
	for name, t := range s.initializers {
		ctx.Set(name, t)
	}
	for name, t := range inputs {
		ctx.Set(name, t)
	}

	for _, node := range s.order {
		nodeInputs := make([]tensor.Tensor, len(node.Inputs))
		for i, name := range node.Inputs {
			if name == "" {
				continue
			}
			t := ctx.Get(name)
			if t == nil {
				return nil, fmt.Errorf("engine: input %q for node %s (%s) not found", name, node.Name, node.OpType)
			}
			nodeInputs[i] = t
		}

		outputs, err := s.executeNode(ctx, node, nodeInputs)
		if err != nil {
			return nil, fmt.Errorf("engine: executing %s (%s): %w", node.Name, node.OpType, err)
		}
		for i, name := range node.Outputs {
			if name == "" || i >= len(outputs) {
				continue
			}
			ctx.Set(name, outputs[i])
		}
	}

	return ctx.values, nil
}

// Input creates a single-entry input map. Convenience for RunWithNames.
// 単一入力の map を作成する。RunWithNames のヘルパー。
//
//	outputs, err := sess.RunWithNames(onnx.Input("input", myTensor))
func Input(name string, t tensor.Tensor) map[string]tensor.Tensor {
	return map[string]tensor.Tensor{name: t}
}

// Inputs creates an input map from name/tensor pairs. Convenience for RunWithNames.
// Panics if the number of arguments is odd, a name is not a string, or a value is not a tensor.Tensor.
// 名前/テンソルのペアから入力 map を作成する。RunWithNames のヘルパー。
// 引数が奇数、名前が string でない、値が tensor.Tensor でない場合は panic する。
//
//	outputs, err := sess.RunWithNames(onnx.Inputs("images", images, "sizes", sizes))
func Inputs(pairs ...interface{}) map[string]tensor.Tensor {
	if len(pairs)%2 != 0 {
		panic(fmt.Sprintf("onnx.Inputs: odd number of arguments (%d), expected name/tensor pairs", len(pairs)))
	}
	m := make(map[string]tensor.Tensor, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		name, ok := pairs[i].(string)
		if !ok {
			panic(fmt.Sprintf("onnx.Inputs: argument %d must be a string, got %T", i, pairs[i]))
		}
		t, ok := pairs[i+1].(tensor.Tensor)
		if !ok {
			panic(fmt.Sprintf("onnx.Inputs: argument %d must be a tensor.Tensor, got %T", i+1, pairs[i+1]))
		}
		m[name] = t
	}
	return m
}

// InputNames returns the names of the model's graph inputs (excluding initializers).
// モデルのグラフ入力名を返す（初期化子を除く）。
func (s *Session) InputNames() []string {
	var names []string
	for _, inp := range s.graph.Inputs {
		if _, isInit := s.graph.Initializers[inp.Name]; !isInit {
			names = append(names, inp.Name)
		}
	}
	return names
}

// Run executes the model by automatically mapping positional tensors to graph input names.
// Input names are resolved in graph definition order (excluding initializers).
// 位置指定のテンソルをグラフ入力名に自動マッピングして推論を実行する。
// 入力名はグラフ定義順（初期化子を除く）で解決される。
//
//	outputs, err := sess.Run(inputTensor)
//	outputs, err := sess.Run(images, sizes)
func (s *Session) Run(inputs ...tensor.Tensor) (map[string]tensor.Tensor, error) {
	names := s.InputNames()
	if len(inputs) > len(names) {
		return nil, fmt.Errorf("engine: Run got %d inputs but model has %d graph inputs", len(inputs), len(names))
	}
	m := make(map[string]tensor.Tensor, len(inputs))
	for i, t := range inputs {
		m[names[i]] = t
	}
	return s.RunWithNames(m)
}

// RunWithNames executes the model with explicitly named inputs.
// 名前指定の入力でモデルを実行する。
//
//	outputs, err := sess.RunWithNames(map[string]tensor.Tensor{"input": t})
//	outputs, err := sess.RunWithNames(onnx.Input("input", t))
func (s *Session) RunWithNames(inputs map[string]tensor.Tensor) (map[string]tensor.Tensor, error) {
	if s.plan != nil {
		return s.runCompiled(inputs)
	}
	return s.run(inputs)
}

func (s *Session) run(inputs map[string]tensor.Tensor) (map[string]tensor.Tensor, error) {
	runInfo := RunInfo{
		TotalNodes: len(s.order),
		Compiled:   false,
		StartedAt:  time.Now(),
	}
	s.observers.onRunStart(runInfo)

	ctx := newContext()

	// Load pre-materialized initializers into context
	for name, t := range s.initializers {
		ctx.Set(name, t)
	}

	// Load user inputs (may override initializers, which is valid per ONNX spec)
	for name, t := range inputs {
		ctx.Set(name, t)
	}

	// Execute nodes in topological order
	for idx, node := range s.order {
		// Gather inputs; empty string means optional/absent 驕ｶ鄙ｫ繝ｻnil
		nodeInputs := make([]tensor.Tensor, len(node.Inputs))
		for i, name := range node.Inputs {
			if name == "" {
				continue
			}
			t := ctx.Get(name)
			if t == nil {
				err := fmt.Errorf("engine: input %q for node %s (%s) not found",
					name, node.Name, node.OpType)
				s.observers.onRunFinish(runInfo, err)
				return nil, err
			}
			nodeInputs[i] = t
		}

		var outputs []tensor.Tensor
		var err error
		startExec := NodeExecution{Node: node, Index: idx, Total: len(s.order)}
		s.observers.onNodeStart(startExec)
		if len(s.observers) > 0 {
			allocBefore := getAllocBytes()
			t0 := time.Now()
			outputs, err = s.executeNode(ctx, node, nodeInputs)
			elapsed := time.Since(t0)
			allocAfter := getAllocBytes()
			finishExec := startExec
			finishExec.Elapsed = elapsed
			finishExec.AllocBytes = allocAfter - allocBefore
			s.observers.onNodeFinish(finishExec, err)
		} else {
			outputs, err = s.executeNode(ctx, node, nodeInputs)
		}
		if err != nil {
			runErr := fmt.Errorf("engine: executing %s (%s): %w", node.Name, node.OpType, err)
			s.observers.onRunFinish(runInfo, runErr)
			return nil, runErr
		}

		// Store outputs
		for i, name := range node.Outputs {
			if name == "" || i >= len(outputs) {
				continue
			}
			ctx.Set(name, outputs[i])
		}
	}

	// Collect graph outputs
	result := make(map[string]tensor.Tensor)
	for _, out := range s.graph.Outputs {
		t := ctx.Get(out.Name)
		if t == nil {
			err := fmt.Errorf("engine: output %q not found after execution", out.Name)
			s.observers.onRunFinish(runInfo, err)
			return nil, err
		}
		result[out.Name] = t
	}

	s.observers.onRunFinish(runInfo, nil)
	return result, nil
}

func (s *Session) executeNode(ctx *executionContext, node *ir.Node, nodeInputs []tensor.Tensor) ([]tensor.Tensor, error) {
	_ = ctx
	opKey := node.OpType
	if node.Domain != "" {
		opKey = node.Domain + "::" + node.OpType
	}
	opFunc := s.registry.Lookup(opKey)
	if opFunc == nil {
		return nil, fmt.Errorf("engine: unsupported op %q (node %s)", opKey, node.Name)
	}
	return opFunc(node, nodeInputs)
}
