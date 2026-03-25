package ops

import (
	"math"
	"sort"

	"github.com/Kazuhito00/onnx-purego-interpreter/internal/ir"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// NonMaxSuppression — selects boxes that do not overlap too much.
// Inputs:
//
//	boxes [num_batches, spatial_dim, 4]     float32
//	scores [num_batches, num_classes, spatial_dim] float32
//	max_output_boxes_per_class  int64 scalar (optional)
//	iou_threshold               float32 scalar (optional)
//	score_threshold             float32 scalar (optional)
//
// Output: selected_indices [num_selected, 3] int64  (batch, class, box_index)
func opNonMaxSuppression(node *ir.Node, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	boxes := inputs[0].(*tensor.Dense[float32])
	scores := inputs[1].(*tensor.Dense[float32])

	maxBoxes := int64(0) // 0 means select all
	if len(inputs) > 2 && inputs[2] != nil {
		if t, ok := inputs[2].(*tensor.Dense[int64]); ok && t.Len() > 0 {
			maxBoxes = t.Data()[0]
		}
	}
	iouThresh := float32(0)
	if len(inputs) > 3 && inputs[3] != nil {
		if t, ok := inputs[3].(*tensor.Dense[float32]); ok && t.Len() > 0 {
			iouThresh = t.Data()[0]
		}
	}
	scoreThresh := float32(-math.MaxFloat32)
	if len(inputs) > 4 && inputs[4] != nil {
		if t, ok := inputs[4].(*tensor.Dense[float32]); ok && t.Len() > 0 {
			scoreThresh = t.Data()[0]
		}
	}

	centerPointBox := int(node.GetAttrInt("center_point_box", 0))

	bs := boxes.Shape()  // [B, N, 4]
	ss := scores.Shape() // [B, C, N]
	numBatches := bs[0]
	numBoxes := bs[1]
	numClasses := ss[1]

	boxData := boxes.Data()
	scoreData := scores.Data()

	var result []int64

	for b := 0; b < numBatches; b++ {
		for c := 0; c < numClasses; c++ {
			// Collect (score, boxIdx) pairs above threshold
			type scoredBox struct {
				score float32
				idx   int
			}
			var candidates []scoredBox
			for i := 0; i < numBoxes; i++ {
				s := scoreData[b*numClasses*numBoxes+c*numBoxes+i]
				if s > scoreThresh {
					candidates = append(candidates, scoredBox{s, i})
				}
			}
			// Sort descending by score
			sort.Slice(candidates, func(i, j int) bool {
				return candidates[i].score > candidates[j].score
			})

			// Greedy NMS
			selected := make([]int, 0)
			suppressed := make([]bool, len(candidates))
			for i, cand := range candidates {
				if suppressed[i] {
					continue
				}
				if maxBoxes > 0 && int64(len(selected)) >= maxBoxes {
					break
				}
				selected = append(selected, cand.idx)
				// Suppress overlapping boxes
				for j := i + 1; j < len(candidates); j++ {
					if suppressed[j] {
						continue
					}
					iou := computeIoU(boxData, b, cand.idx, candidates[j].idx, numBoxes, centerPointBox)
					if iou > iouThresh {
						suppressed[j] = true
					}
				}
			}

			for _, boxIdx := range selected {
				result = append(result, int64(b), int64(c), int64(boxIdx))
			}
		}
	}

	numSelected := len(result) / 3
	if numSelected == 0 {
		return []tensor.Tensor{tensor.NewDense[int64](tensor.Shape{0, 3}, nil)}, nil
	}
	return []tensor.Tensor{tensor.NewDense[int64](tensor.Shape{numSelected, 3}, result)}, nil
}

func computeIoU(boxes []float32, batch, i, j, numBoxes, centerPointBox int) float32 {
	base := batch * numBoxes * 4
	var y1i, x1i, y2i, x2i, y1j, x1j, y2j, x2j float32

	if centerPointBox == 0 {
		// [y1, x1, y2, x2]
		y1i = boxes[base+i*4+0]
		x1i = boxes[base+i*4+1]
		y2i = boxes[base+i*4+2]
		x2i = boxes[base+i*4+3]
		y1j = boxes[base+j*4+0]
		x1j = boxes[base+j*4+1]
		y2j = boxes[base+j*4+2]
		x2j = boxes[base+j*4+3]
	} else {
		// [cx, cy, w, h] → convert to [y1, x1, y2, x2]
		cx, cy := boxes[base+i*4+0], boxes[base+i*4+1]
		wi, hi := boxes[base+i*4+2], boxes[base+i*4+3]
		x1i, y1i = cx-wi/2, cy-hi/2
		x2i, y2i = cx+wi/2, cy+hi/2

		cx, cy = boxes[base+j*4+0], boxes[base+j*4+1]
		wj, hj := boxes[base+j*4+2], boxes[base+j*4+3]
		x1j, y1j = cx-wj/2, cy-hj/2
		x2j, y2j = cx+wj/2, cy+hj/2
	}

	// Ensure min < max
	if y1i > y2i {
		y1i, y2i = y2i, y1i
	}
	if x1i > x2i {
		x1i, x2i = x2i, x1i
	}
	if y1j > y2j {
		y1j, y2j = y2j, y1j
	}
	if x1j > x2j {
		x1j, x2j = x2j, x1j
	}

	interY1 := max32(y1i, y1j)
	interX1 := max32(x1i, x1j)
	interY2 := min32(y2i, y2j)
	interX2 := min32(x2i, x2j)

	interArea := max32(0, interY2-interY1) * max32(0, interX2-interX1)
	areaI := (y2i - y1i) * (x2i - x1i)
	areaJ := (y2j - y1j) * (x2j - x1j)
	union := areaI + areaJ - interArea

	if union <= 0 {
		return 0
	}
	return interArea / union
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
