package ir

import "fmt"

// AttrValue is the interface for ONNX node attribute values.
type AttrValue interface {
	attrValue() // marker
}

type AttrInt struct{ Value int64 }
type AttrFloat struct{ Value float32 }
type AttrString struct{ Value string }
type AttrInts struct{ Value []int64 }
type AttrFloats struct{ Value []float32 }
type AttrTensor struct{ Value *Initializer }

// AttrGraph holds a subgraph (used by If, Loop, Scan nodes).
type AttrGraph struct {
	Value *Graph
}

func (AttrInt) attrValue()    {}
func (AttrFloat) attrValue()  {}
func (AttrString) attrValue() {}
func (AttrInts) attrValue()   {}
func (AttrFloats) attrValue() {}
func (AttrTensor) attrValue() {}
func (*AttrGraph) attrValue() {}

// Node represents a single operation in the graph.
type Node struct {
	OpType       string
	Name         string
	Domain       string
	OpsetVersion int64    // opset version for this node's domain
	Inputs       []string // input tensor names; empty string = optional/absent
	Outputs      []string
	Attrs        map[string]AttrValue
}

// GetAttrInt returns an integer attribute or a default value.
func (n *Node) GetAttrInt(name string, def int64) int64 {
	if v, ok := n.Attrs[name]; ok {
		if a, ok := v.(AttrInt); ok {
			return a.Value
		}
	}
	return def
}

// GetAttrFloat returns a float attribute or a default value.
func (n *Node) GetAttrFloat(name string, def float32) float32 {
	if v, ok := n.Attrs[name]; ok {
		if a, ok := v.(AttrFloat); ok {
			return a.Value
		}
	}
	return def
}

// GetAttrString returns a string attribute or a default value.
func (n *Node) GetAttrString(name string, def string) string {
	if v, ok := n.Attrs[name]; ok {
		if a, ok := v.(AttrString); ok {
			return a.Value
		}
	}
	return def
}

// GetAttrInts returns an int-slice attribute or a default value.
func (n *Node) GetAttrInts(name string, def []int64) []int64 {
	if v, ok := n.Attrs[name]; ok {
		if a, ok := v.(AttrInts); ok {
			return a.Value
		}
	}
	return def
}

// GetAttrFloats returns a float-slice attribute or a default value.
func (n *Node) GetAttrFloats(name string, def []float32) []float32 {
	if v, ok := n.Attrs[name]; ok {
		if a, ok := v.(AttrFloats); ok {
			return a.Value
		}
	}
	return def
}

// GetAttrTensor returns a tensor attribute or nil.
func (n *Node) GetAttrTensor(name string) *Initializer {
	if v, ok := n.Attrs[name]; ok {
		if a, ok := v.(AttrTensor); ok {
			return a.Value
		}
	}
	return nil
}

// OpsetImport records a domain and its opset version.
type OpsetImport struct {
	Domain  string
	Version int64
}

// Graph is the intermediate representation of an ONNX computation graph.
type Graph struct {
	Name         string
	Nodes        []*Node
	Inputs       []TensorSpec
	Outputs      []TensorSpec
	Initializers map[string]*Initializer
	Opsets       []OpsetImport // opset imports from the model
}

// OpsetVersion returns the opset version for the given domain (empty string = default ONNX domain).
func (g *Graph) OpsetVersion(domain string) int64 {
	for _, o := range g.Opsets {
		if o.Domain == domain {
			return o.Version
		}
	}
	return 0
}

func (g *Graph) String() string {
	return fmt.Sprintf("Graph(%s, %d nodes, %d inputs, %d outputs)",
		g.Name, len(g.Nodes), len(g.Inputs), len(g.Outputs))
}
