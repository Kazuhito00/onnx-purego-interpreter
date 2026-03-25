# DEIMv2 Object Detection Example

Object detection sample using DEIMv2 (HGNetV2-Atto, COCO 80 classes).<br>
DEIMv2 (HGNetV2-Atto, COCO 80クラス) を使用した物体検出サンプル。

## Included Files / 同梱ファイル

| File / ファイル | Description / 説明 |
|---|---|
| `main.go` | Inference, post-processing, and result saving / 推論・後処理・結果保存 |
| `draw.go` | Bounding box and text drawing utilities / バウンディングボックス・テキスト描画 |
| `deimv2_hgnetv2_atto_coco.onnx` | DEIMv2 HGNetV2-Atto COCO model (input: 320×320) / DEIMv2 モデル (入力: 320×320) |
| `sample.jpg` | Sample input image / サンプル入力画像 |

## Run / 実行

```bash
cd examples/DEIMv2
go run .
```

Detection results are saved to `result.png`.<br>
検出結果は `result.png` に保存。

## Configuration / 設定

Edit constants in `main.go` to change input size or score threshold.<br>
`main.go` の定数で入力サイズやスコア閾値を変更可能。

```go
const (
    modelPath  = "deimv2_hgnetv2_atto_coco.onnx"
    imagePath  = "sample.jpg"
    outputPath = "result.png"
    inputH     = 320
    inputW     = 320
    scoreTh    = 0.6
)
```

## Model Source / モデル取得元

The ONNX model is from [DEIMv2-ONNX-Sample](https://github.com/Kazuhito00/DEIMv2-ONNX-Sample).<br>
ONNX モデルは [DEIMv2-ONNX-Sample](https://github.com/Kazuhito00/DEIMv2-ONNX-Sample) から取得。

## Output / 出力
<img width="800" height="533" alt="result" src="https://github.com/user-attachments/assets/57426a50-5b29-43d4-8479-ad029347c074" />

- Bounding boxes with class name and score drawn on detected objects / 検出オブジェクトにクラス名・スコア付きバウンディングボックスを描画
- Inference time displayed in the top-left corner / 左上に推論時間を表示

## Sample Image / サンプル画像

Sample image from [pakutaso](https://www.pakutaso.com/): "[トライアスロン競技で自転車走行する選手の力強い走り](https://www.pakutaso.com/20260155013post-56295.html)".<br>
サンプル画像は[ぱくたそ](https://www.pakutaso.com/)様の「[トライアスロン競技で自転車走行する選手の力強い走り](https://www.pakutaso.com/20260155013post-56295.html)」を使用。
