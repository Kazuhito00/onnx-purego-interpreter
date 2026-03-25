# MobileNetV2 Image Classification Example

Image classification sample using MobileNetV2 (ImageNet 1000 classes).<br>
MobileNetV2 (ImageNet 1000クラス) を使用した画像分類サンプル。

## Included Files / 同梱ファイル

| File / ファイル | Description / 説明 |
|---|---|
| `main.go` | Inference, pre/post-processing / 推論・前後処理 |
| `labels.go` | ImageNet 1000 class labels / ImageNet 1000クラスラベル |
| `mobilenetv2-7.onnx` | MobileNetV2 ONNX model (input: 224×224) / MobileNetV2 モデル (入力: 224×224) |
| `sample.jpg` | Sample input image / サンプル入力画像 |

## Run / 実行

```bash
cd examples/MobileNetV2
go run .
```

Top-5 classification results are printed to the console.<br>
Top-5 分類結果をコンソールに表示。

## Output Example / 出力例
![sample](https://github.com/user-attachments/assets/14b64923-ff86-48a5-a100-97b75a5738a3)
```
Image: 800x533
Model loaded: mobilenetv2-7.onnx
Inference: 334.5ms

Top-5:
  1. [ 681] notebook                       25.08%
  2. [ 549] envelope                       19.88%
  3. [ 583] guillotine                     14.33%
  4. [ 620] laptop                         12.14%
  5. [ 921] book jacket                    4.93%
```

## Configuration / 設定

Edit constants in `main.go` to change input size or top-K count.<br>
`main.go` の定数で入力サイズや Top-K 数を変更可能。

```go
const (
    modelPath = "mobilenetv2-7.onnx"
    imagePath = "sample.jpg"
    inputH    = 224
    inputW    = 224
    topK      = 5
)
```

## Model Source / モデル取得元

[ONNX Model Zoo - MobileNetV2](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)

## Sample Image / サンプル画像

Sample image from [pakutaso](https://www.pakutaso.com/): "[暗い床に置かれたノートパソコンの液晶から伸びる影](https://www.pakutaso.com/20230515130post-46767.html)".<br>
サンプル画像は[ぱくたそ](https://www.pakutaso.com/)様の「[暗い床に置かれたノートパソコンの液晶から伸びる影](https://www.pakutaso.com/20230515130post-46767.html)」を使用。
