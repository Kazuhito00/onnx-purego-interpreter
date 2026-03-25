# DEIMv2 Object Detection Example

DEIMv2 (HGNetV2-Atto, COCO 80クラス) を使用した物体検出サンプル。

## 同梱ファイル

| ファイル | 説明 |
|---|---|
| `main.go` | 推論・後処理・結果保存のメインコード |
| `draw.go` | バウンディングボックス・テキスト描画ユーティリティ |
| `deimv2_hgnetv2_atto_coco.onnx` | DEIMv2 HGNetV2-Atto COCO モデル (入力: 320×320) |
| `sample.jpg` | サンプル入力画像 |

## 実行

```bash
cd examples/DEIMv2
go run .
```

`result.png` に検出結果が保存されます。

## 設定

`main.go` の定数で入力サイズやスコア閾値を変更できます:

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

## 出力例

- 検出されたオブジェクトにバウンディングボックスとクラス名・スコアを描画
- 左上に推論時間を表示
