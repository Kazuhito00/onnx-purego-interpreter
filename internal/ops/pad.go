package ops

import "github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"

// computePads resolves padding from explicit pads or auto_pad attribute.
// Returns padTop, padLeft, padBottom, padRight.
func computePads(node *ir.Node, H, W, effKH, effKW, strideH, strideW int) (int, int, int, int) {
	autoPad := node.GetAttrString("auto_pad", "NOTSET")

	switch autoPad {
	case "SAME_UPPER":
		return samePads(H, W, effKH, effKW, strideH, strideW, false)
	case "SAME_LOWER":
		return samePads(H, W, effKH, effKW, strideH, strideW, true)
	case "VALID":
		return 0, 0, 0, 0
	default: // "NOTSET" or empty — use explicit pads
		pads := node.GetAttrInts("pads", []int64{0, 0, 0, 0})
		return int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
	}
}

// samePads computes SAME padding so that output_size = ceil(input_size / stride).
// If lower is true, extra padding goes to top/left (SAME_LOWER).
func samePads(H, W, effKH, effKW, strideH, strideW int, lower bool) (int, int, int, int) {
	oh := (H + strideH - 1) / strideH
	ow := (W + strideW - 1) / strideW
	padH := max(0, (oh-1)*strideH+effKH-H)
	padW := max(0, (ow-1)*strideW+effKW-W)

	var padTop, padBottom, padLeft, padRight int
	if lower {
		padBottom = padH / 2
		padTop = padH - padBottom
		padRight = padW / 2
		padLeft = padW - padRight
	} else {
		padTop = padH / 2
		padBottom = padH - padTop
		padLeft = padW / 2
		padRight = padW - padLeft
	}
	return padTop, padLeft, padBottom, padRight
}
