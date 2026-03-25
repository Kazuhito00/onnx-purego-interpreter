// DEIMv2 COCO 推論サンプル
//
// sample.jpg を DEIMv2 (COCO 80クラス) で推論し、検出結果を result.png に保存する。
// Python版 sample_onnx.py と同等の処理を Pure Go で実行。
//
// 使い方:
//
//	go run .
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"time"

	"github.com/Kazuhito00/onnx-purego-interpreter/onnx"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── モデル設定 ──

const (
	modelPath  = "deimv2_hgnetv2_atto_coco.onnx"
	imagePath  = "sample.jpg"
	outputPath = "result.png"

	inputH  = 320
	inputW  = 320
	scoreTh = 0.6
)

// ── COCO 80 クラス名 ──

var cocoLabels = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
	"mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// ── クラス色 (classID mod 10 でカラフルに) ──

var palette = []color.RGBA{
	{230, 25, 75, 255}, {60, 180, 75, 255}, {255, 225, 25, 255}, {0, 130, 200, 255},
	{245, 130, 48, 255}, {145, 30, 180, 255}, {70, 240, 240, 255}, {240, 50, 230, 255},
	{210, 245, 60, 255}, {250, 190, 212, 255},
}

// ── 検出結果 ──

type Box struct {
	ClassID    int
	Score      float32
	X1, Y1     int
	X2, Y2     int
}

func main() {
	// ── 画像読み込み ──
	imgFile, err := os.Open(imagePath)
	if err != nil {
		panic(err)
	}
	defer imgFile.Close()
	srcImg, err := jpeg.Decode(imgFile)
	if err != nil {
		panic(err)
	}
	bounds := srcImg.Bounds()
	imgW, imgH := bounds.Dx(), bounds.Dy()
	fmt.Printf("Image: %dx%d\n", imgW, imgH)

	// ── 前処理: リサイズ → RGB → /255.0 正規化 → CHW ──
	resized := resizeBilinear(srcImg, inputW, inputH)
	inputData := make([]float32, 3*inputH*inputW)
	for y := 0; y < inputH; y++ {
		for x := 0; x < inputW; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			// RGB 順、0-1 正規化
			inputData[0*inputH*inputW+y*inputW+x] = float32(r>>8) / 255.0
			inputData[1*inputH*inputW+y*inputW+x] = float32(g>>8) / 255.0
			inputData[2*inputH*inputW+y*inputW+x] = float32(b>>8) / 255.0
		}
	}
	imagesTensor := tensor.NewDense[float32](tensor.Shape{1, 3, inputH, inputW}, inputData)
	sizesTensor := tensor.NewDense[int64](tensor.Shape{1, 2}, []int64{int64(imgW), int64(imgH)})

	// ── モデル読み込み ──
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		panic(err)
	}
	sess, err := onnx.NewSessionWithOptions(modelBytes, onnx.WithProgressLogger(os.Stdout))
	if err != nil {
		panic(err)
	}
	fmt.Printf("Model loaded: %s\n", modelPath)
	fmt.Printf("Input names: %v\n", sess.InputNames())

	// ── 推論 ──
	start := time.Now()
	outputs, err := sess.Run(imagesTensor, sizesTensor)
	if err != nil {
		panic(err)
	}
	elapsed := time.Since(start)
	fmt.Printf("Inference: %.1fms\n", float64(elapsed.Microseconds())/1000)

	// ── 後処理 ──
	// 出力: labels[1,N] int64, boxes[1,N,4] float32 (ピクセル座標), scores[1,N] float32
	var labels *tensor.Dense[int64]
	var bboxes *tensor.Dense[float32]
	var scores *tensor.Dense[float32]
	for name, t := range outputs {
		switch name {
		case "labels":
			labels = t.(*tensor.Dense[int64])
		case "boxes":
			bboxes = t.(*tensor.Dense[float32])
		case "scores":
			scores = t.(*tensor.Dense[float32])
		}
	}
	if labels == nil || bboxes == nil || scores == nil {
		panic("missing output tensors")
	}

	numDet := scores.Shape()[1]
	labelsData := labels.Data()
	bboxesData := bboxes.Data()
	scoresData := scores.Data()

	var boxes []Box
	for i := 0; i < numDet; i++ {
		score := scoresData[i]
		if score < scoreTh {
			continue
		}
		classID := int(labelsData[i])
		x1 := int(bboxesData[i*4+0])
		y1 := int(bboxesData[i*4+1])
		x2 := int(bboxesData[i*4+2])
		y2 := int(bboxesData[i*4+3])
		boxes = append(boxes, Box{ClassID: classID, Score: score, X1: x1, Y1: y1, X2: x2, Y2: y2})
	}
	fmt.Printf("Detections: %d (score > %.1f)\n", len(boxes), scoreTh)

	// ── 描画 ──
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, srcImg, image.Point{}, draw.Src)

	for _, b := range boxes {
		c := palette[b.ClassID%len(palette)]
		label := fmt.Sprintf("%d", b.ClassID)
		if b.ClassID < len(cocoLabels) {
			label = cocoLabels[b.ClassID]
		}
		label = fmt.Sprintf("%s:%.0f%%", label, b.Score*100)

		// バウンディングボックス
		drawRectThick(canvas, b.X1, b.Y1, b.X2, b.Y2, c, 2)
		// ラベル背景 + テキスト
		tw := len(label)*6 + 4
		fillRect(canvas, b.X1, b.Y1-12, b.X1+tw, b.Y1, c)
		drawText(canvas, b.X1+2, b.Y1-10, label, color.RGBA{0, 0, 0, 255})
	}

	// 推論時間を左上に表示
	timeText := fmt.Sprintf("%.1f ms", float64(elapsed.Microseconds())/1000)
	tw := len(timeText)*6 + 6
	fillRect(canvas, 0, 0, tw, 14, color.RGBA{0, 0, 0, 200})
	drawText(canvas, 3, 4, timeText, color.RGBA{255, 255, 255, 255})

	// ── 保存 ──
	outFile, err := os.Create(outputPath)
	if err != nil {
		panic(err)
	}
	png.Encode(outFile, canvas)
	outFile.Close()
	fmt.Printf("Result saved: %s\n", outputPath)
}
