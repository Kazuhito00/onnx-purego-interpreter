package optimize

import (
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/internal/materialize"
)

// fuseRMSNormalization は分解された RMSNorm パターンを 1 ノードに融合する。
//
//	変種1: Mul(Div(x, Sqrt(Add(ReduceMean(Pow(x,2)), eps))), scale)
//	変種2: Mul(Mul(x, Div(1, Sqrt(Add(ReduceMean(Pow(x,2)), eps)))), scale)  — PyTorch の逆数形
//
// RMSNormalization カーネルは [axis, rank) を正規化するが、分解パターンは
// 単一軸のみの縮約。rank 不明でも等価性を保証できる最終軸 (-1) に限定する。
func fuseRMSNormalization(g *ir.Graph) {
	producer, uses := producerMap(g), valueUseCounts(g)
	var dead []*ir.Node
	for _, final := range g.Nodes {
		if final.OpType != "Mul" || len(final.Inputs) != 2 {
			continue
		}
		for swap := 0; swap < 2; swap++ {
			scale := final.Inputs[1-swap]
			normed := final.Inputs[swap]
			if g.Initializers[scale] == nil {
				continue
			}
			n := producer[normed]
			if n == nil || uses[normed] != 1 || len(n.Inputs) != 2 {
				continue
			}
			matched := false
			switch n.OpType {
			case "Div": // 変種1: Div(x, sqrt連鎖)
				x, eps, axis, chain, ok := matchRMSChain(g, producer, uses, n.Inputs[1])
				if ok && axis == -1 && n.Inputs[0] == x {
					setRMSNorm(final, x, scale, eps)
					dead = append(dead, n)
					dead = append(dead, chain...)
					matched = true
				}
			case "Mul": // 変種2: Mul(x, Div(1, sqrt連鎖)) — 引数は順不同
				for s2 := 0; s2 < 2; s2++ {
					recipName, xCand := n.Inputs[s2], n.Inputs[1-s2]
					recip := producer[recipName]
					if recip == nil || recip.OpType != "Div" || uses[recipName] != 1 || len(recip.Inputs) != 2 {
						continue
					}
					if v, ok := scalarFloat(g.Initializers[recip.Inputs[0]]); !ok || v != 1 {
						continue
					}
					x, eps, axis, chain, ok := matchRMSChain(g, producer, uses, recip.Inputs[1])
					if !ok || axis != -1 || xCand != x {
						continue
					}
					setRMSNorm(final, x, scale, eps)
					dead = append(dead, n, recip)
					dead = append(dead, chain...)
					matched = true
					break
				}
			}
			if matched {
				break
			}
		}
	}
	removeNodes(g, dead)
}

func setRMSNorm(n *ir.Node, x, scale string, eps float32) {
	n.OpType = "RMSNormalization"
	n.Inputs = []string{x, scale}
	n.Attrs = map[string]ir.AttrValue{"axis": ir.AttrInt{Value: -1}, "epsilon": ir.AttrFloat{Value: eps}}
}

// matchRMSChain は Sqrt(Add(ReduceMean(Pow(x,2)), eps)) の連鎖を照合する。
// sqrtOut は Sqrt の出力値名。成功時は x・eps・縮約軸・連鎖ノード群を返す。
func matchRMSChain(g *ir.Graph, producer map[string]*ir.Node, uses map[string]int, sqrtOut string) (x string, eps float32, axis int64, chain []*ir.Node, ok bool) {
	sqrt := producer[sqrtOut]
	if sqrt == nil || sqrt.OpType != "Sqrt" || uses[sqrtOut] != 1 {
		return
	}
	add := producer[sqrt.Inputs[0]]
	if add == nil || add.OpType != "Add" || uses[add.Outputs[0]] != 1 {
		return
	}
	mean, epsName := producerAndConst(g, producer, add.Inputs[0], add.Inputs[1], "ReduceMean")
	if mean == nil {
		mean, epsName = producerAndConst(g, producer, add.Inputs[1], add.Inputs[0], "ReduceMean")
	}
	if mean == nil || uses[mean.Outputs[0]] != 1 {
		return
	}
	pow := producer[mean.Inputs[0]]
	if pow == nil || pow.OpType != "Pow" || len(pow.Inputs) != 2 || uses[pow.Outputs[0]] != 1 {
		return
	}
	if v, o := scalarFloat(g.Initializers[pow.Inputs[1]]); !o || v != 2 {
		return
	}
	eps, epsOK := scalarFloat(g.Initializers[epsName])
	if !epsOK {
		return
	}
	axis, axisOK := reductionAxis(g, mean)
	if !axisOK {
		return
	}
	return pow.Inputs[0], eps, axis, []*ir.Node{sqrt, add, mean, pow}, true
}

func fuseLayerNormalization(g *ir.Graph) {
	producer, uses := producerMap(g), valueUseCounts(g)
	var dead []*ir.Node
	for _, final := range g.Nodes {
		if final.OpType != "Add" || len(final.Inputs) != 2 {
			continue
		}
		mul, bias := producerAndConst(g, producer, final.Inputs[0], final.Inputs[1], "Mul")
		if mul == nil {
			mul, bias = producerAndConst(g, producer, final.Inputs[1], final.Inputs[0], "Mul")
		}
		if mul == nil || uses[mul.Outputs[0]] != 1 {
			continue
		}
		div, scale := producerAndConst(g, producer, mul.Inputs[0], mul.Inputs[1], "Div")
		if div == nil {
			div, scale = producerAndConst(g, producer, mul.Inputs[1], mul.Inputs[0], "Div")
		}
		if div == nil || uses[div.Outputs[0]] != 1 || len(div.Inputs) != 2 {
			continue
		}
		sub, sqrt := producer[div.Inputs[0]], producer[div.Inputs[1]]
		if sub == nil || sub.OpType != "Sub" || sqrt == nil || sqrt.OpType != "Sqrt" || uses[sub.Outputs[0]] != 2 || uses[sqrt.Outputs[0]] != 1 {
			continue
		}
		if len(sub.Inputs) != 2 {
			continue
		}
		x := sub.Inputs[0]
		mean := producer[sub.Inputs[1]]
		if mean == nil || mean.OpType != "ReduceMean" || mean.Inputs[0] != x || uses[mean.Outputs[0]] != 1 {
			continue
		}
		add := producer[sqrt.Inputs[0]]
		if add == nil || add.OpType != "Add" || uses[add.Outputs[0]] != 1 {
			continue
		}
		variance, epsName := producerAndConst(g, producer, add.Inputs[0], add.Inputs[1], "ReduceMean")
		if variance == nil {
			variance, epsName = producerAndConst(g, producer, add.Inputs[1], add.Inputs[0], "ReduceMean")
		}
		if variance == nil || uses[variance.Outputs[0]] != 1 {
			continue
		}
		pow := producer[variance.Inputs[0]]
		if pow == nil || pow.OpType != "Pow" || len(pow.Inputs) != 2 || pow.Inputs[0] != sub.Outputs[0] || uses[pow.Outputs[0]] != 1 {
			continue
		}
		if v, ok := scalarFloat(g.Initializers[pow.Inputs[1]]); !ok || v != 2 {
			continue
		}
		eps, ok := scalarFloat(g.Initializers[epsName])
		if !ok {
			continue
		}
		// LayerNormalization カーネルは [axis, rank) を正規化するが、分解パターンは
		// 単一軸のみの縮約。rank 不明でも等価性を保証できる最終軸 (-1) に限定する
		axis1, ok1 := reductionAxis(g, mean)
		axis2, ok2 := reductionAxis(g, variance)
		if !ok1 || !ok2 || axis1 != axis2 || axis1 != -1 {
			continue
		}
		final.OpType = "LayerNormalization"
		final.Inputs = []string{x, scale, bias}
		final.Attrs = map[string]ir.AttrValue{"axis": ir.AttrInt{Value: axis1}, "epsilon": ir.AttrFloat{Value: eps}}
		dead = append(dead, mul, div, sub, sqrt, add, variance, pow, mean)
	}
	removeNodes(g, dead)
}

func producerAndConst(g *ir.Graph, producers map[string]*ir.Node, nodeValue, constValue, op string) (*ir.Node, string) {
	n := producers[nodeValue]
	if n == nil || n.OpType != op || g.Initializers[constValue] == nil {
		return nil, ""
	}
	return n, constValue
}

func scalarFloat(init *ir.Initializer) (float32, bool) {
	if init == nil || init.DType != ir.DataTypeFloat {
		return 0, false
	}
	v, err := materialize.Float32(init)
	return func() (float32, bool) {
		if err != nil || len(v) != 1 {
			return 0, false
		}
		return v[0], true
	}()
}

func reductionAxis(g *ir.Graph, n *ir.Node) (int64, bool) {
	if axes := n.GetAttrInts("axes", nil); len(axes) == 1 {
		return axes[0], true
	}
	if len(n.Inputs) > 1 {
		init := g.Initializers[n.Inputs[1]]
		if init == nil {
			return 0, false
		}
		v, err := materialize.Int64(init)
		if err == nil && len(v) == 1 {
			return v[0], true
		}
	}
	return 0, false
}
