package optimize

import (
	"time"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
)

type passDef struct {
	name string
	run  func(*ir.Graph)
}

// PassResult describes one optimization pass execution.
type PassResult struct {
	Name        string
	BeforeNodes int
	AfterNodes  int
	Elapsed     time.Duration
	Disabled    bool
}

var pipeline = []passDef{
	{name: "materialize_constants", run: materializeConstants},
	{name: "fold_constants", run: foldConstants},
	{name: "fuse_pad_conv", run: fusePadConv},
	{name: "eliminate_dropout", run: eliminateDropout},
	{name: "eliminate_identity", run: eliminateIdentity},
	// fuse_mul_add_affine より前に実行しないと LN の Mul+Add が先に奪われる
	{name: "fuse_layer_norm", run: fuseLayerNormalization},
	{name: "fuse_rms_norm", run: fuseRMSNormalization},
	{name: "fuse_conv_batchnorm", run: fuseConvBatchNorm},
	{name: "fuse_conv_add_bias", run: fuseConvAddBias},
	{name: "fuse_conv_activation", run: fuseConvActivation},
	{name: "fuse_matmul_add_bias", run: fuseMatMulAddBias},
	{name: "fuse_mul_add_affine", run: fuseMulAddAffine},
	{name: "fuse_gelu", run: fuseGELU},
	{name: "fuse_conv_silu", run: fuseConvSiLU},
	{name: "fuse_conv_affine", run: fuseConvAffine},
	{name: "simplify_transposes", run: simplifyTransposes},
	{name: "eliminate_dead_nodes", run: eliminateDeadNodes},
}

// PassNames returns the names of all registered optimization passes in order.
func PassNames() []string {
	names := make([]string, len(pipeline))
	for i, p := range pipeline {
		names[i] = p.name
	}
	return names
}

// Apply runs the canonical optimization pipeline and reports pass-level results.
func Apply(g *ir.Graph, disabled map[string]bool) []PassResult {
	results := make([]PassResult, 0, len(pipeline))
	for _, pass := range pipeline {
		before := len(g.Nodes)
		if disabled[pass.name] {
			results = append(results, PassResult{
				Name:        pass.name,
				BeforeNodes: before,
				AfterNodes:  before,
				Disabled:    true,
			})
			continue
		}
		started := time.Now()
		pass.run(g)
		results = append(results, PassResult{
			Name:        pass.name,
			BeforeNodes: before,
			AfterNodes:  len(g.Nodes),
			Elapsed:     time.Since(started),
		})
	}
	return results
}
