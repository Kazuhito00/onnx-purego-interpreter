// MobileNetV2 ImageNet 画像分類サンプル
//
// sample.jpg を MobileNetV2 (ImageNet 1000クラス) で推論し、Top-5 結果を表示する。
//
// 使い方:
//
//	go run .
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"sort"
	"time"

	"github.com/Kazuhito00/onnx-purego-interpreter/onnx"
	"github.com/Kazuhito00/onnx-purego-interpreter/tensor"
)

// ── モデル設定 ──

const (
	modelPath = "mobilenetv2-7.onnx"
	imagePath = "sample.jpg"

	inputH = 224
	inputW = 224
	topK   = 5
)

// ImageNet normalization parameters
var (
	mean = [3]float32{0.485, 0.456, 0.406}
	std  = [3]float32{0.229, 0.224, 0.225}
)

type prediction struct {
	ClassID int
	Score   float64
	Label   string
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
	fmt.Printf("Image: %dx%d\n", bounds.Dx(), bounds.Dy())

	// ── 前処理: リサイズ → RGB → /255.0 → ImageNet正規化 → CHW ──
	resized := resizeBilinear(srcImg, inputW, inputH)
	inputData := make([]float32, 3*inputH*inputW)
	for y := 0; y < inputH; y++ {
		for x := 0; x < inputW; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			inputData[0*inputH*inputW+y*inputW+x] = (float32(r>>8)/255.0 - mean[0]) / std[0]
			inputData[1*inputH*inputW+y*inputW+x] = (float32(g>>8)/255.0 - mean[1]) / std[1]
			inputData[2*inputH*inputW+y*inputW+x] = (float32(b>>8)/255.0 - mean[2]) / std[2]
		}
	}
	inputTensor := tensor.NewDense[float32](tensor.Shape{1, 3, inputH, inputW}, inputData)

	// ── モデル読み込み ──
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		panic(err)
	}
	sess, err := onnx.NewSession(modelBytes)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Model loaded: %s\n", modelPath)

	// ── 推論 ──
	start := time.Now()
	outputs, err := sess.Run(inputTensor)
	if err != nil {
		panic(err)
	}
	elapsed := time.Since(start)
	fmt.Printf("Inference: %.1fms\n", float64(elapsed.Microseconds())/1000)

	// ── 後処理: softmax → Top-5 ──
	var logits []float32
	for _, t := range outputs {
		logits = t.(*tensor.Dense[float32]).Data()
		break
	}

	probs := softmax(logits)

	preds := make([]prediction, len(probs))
	for i, p := range probs {
		label := fmt.Sprintf("class_%d", i)
		if i < len(imagenetLabels) {
			label = imagenetLabels[i]
		}
		preds[i] = prediction{ClassID: i, Score: p, Label: label}
	}
	sort.Slice(preds, func(i, j int) bool { return preds[i].Score > preds[j].Score })

	fmt.Printf("\nTop-%d:\n", topK)
	for i := 0; i < topK && i < len(preds); i++ {
		p := preds[i]
		fmt.Printf("  %d. [%4d] %-30s %.2f%%\n", i+1, p.ClassID, p.Label, p.Score*100)
	}
}

func softmax(logits []float32) []float64 {
	maxVal := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxVal {
			maxVal = float64(v)
		}
	}
	exps := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		exps[i] = math.Exp(float64(v) - maxVal)
		sum += exps[i]
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

// resizeBilinear はバイリニア補間でリサイズ
func resizeBilinear(src image.Image, w, h int) image.Image {
	bounds := src.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sx := (float64(x)+0.5)*float64(srcW)/float64(w) - 0.5
			sy := (float64(y)+0.5)*float64(srcH)/float64(h) - 0.5
			x0 := int(sx)
			y0 := int(sy)
			if x0 < 0 {
				x0 = 0
			}
			if y0 < 0 {
				y0 = 0
			}
			x1 := x0 + 1
			y1 := y0 + 1
			if x1 >= srcW {
				x1 = srcW - 1
			}
			if y1 >= srcH {
				y1 = srcH - 1
			}
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			if fx < 0 {
				fx = 0
			}
			if fy < 0 {
				fy = 0
			}

			r00, g00, b00, a00 := src.At(bounds.Min.X+x0, bounds.Min.Y+y0).RGBA()
			r10, g10, b10, a10 := src.At(bounds.Min.X+x1, bounds.Min.Y+y0).RGBA()
			r01, g01, b01, a01 := src.At(bounds.Min.X+x0, bounds.Min.Y+y1).RGBA()
			r11, g11, b11, a11 := src.At(bounds.Min.X+x1, bounds.Min.Y+y1).RGBA()

			lerp := func(v00, v10, v01, v11 uint32) uint8 {
				top := float64(v00)*(1-fx) + float64(v10)*fx
				bot := float64(v01)*(1-fx) + float64(v11)*fx
				return uint8((top*(1-fy) + bot*fy) / 256)
			}
			dst.SetRGBA(x, y, color.RGBA{
				R: lerp(r00, r10, r01, r11),
				G: lerp(g00, g10, g01, g11),
				B: lerp(b00, b10, b01, b11),
				A: lerp(a00, a10, a01, a11),
			})
		}
	}
	return dst
}
