# Operator Coverage

ONNX 標準オペレーター 198 個中 **158 個** を実装済み (80%)。
内部融合オペレーター 4 個 (FastGELU, FusedConv, FusedMatMul, FusedAffine)。
Python ORT 比較テスト **122 ケース** 全 PASS (opset 18)。

## 凡例

| 記号 | 意味 |
|---|---|
| ✅ | 対応済み |
| ❌ | 未対応 |
| **Go** | onnx-purego-interpreter の実装状況 |
| **テスト** | Python ORT 比較テスト (`generate_test_models.py`) の有無 |
| **型** | Go 実装が対応するデータ型 |

**型の略記:** f32=float32, f64=float64, i32=int32, i64=int64, u8=uint8, i8=int8, bool=uint8 boolean

---

## Arithmetic

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Abs | ✅ | ✅ | f32 f64 i32 i64 | |
| Add | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting 対応 |
| Ceil | ✅ | ✅ | f32 | |
| Div | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting 対応 |
| Einsum | ✅ | ✅ | f32 | 1 入力 (転置) / 2 入力 (縮約) |
| Floor | ✅ | ✅ | f32 f64 i32 i64 | |
| Max | ✅ | ✅ | f32 | |
| Mean | ✅ | ✅ | f32 | |
| Min | ✅ | ✅ | f32 | |
| Mod | ✅ | ✅ | f32 f64 i32 i64 | fmod 属性対応 |
| Mul | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting 対応 |
| Neg | ✅ | ✅ | f32 f64 i32 i64 | |
| Pow | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting 対応 |
| Reciprocal | ✅ | ✅ | f32 f64 i32 i64 | |
| Round | ✅ | ✅ | f32 | Banker's rounding |
| Sign | ✅ | ✅ | f32 | |
| Sqrt | ✅ | ✅ | f32 f64 i32 i64 | |
| Sub | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting 対応 |
| Sum | ✅ | ✅ | f32 | |
| BitShift | ✅ | | u8 | LEFT / RIGHT 対応 |
| BitwiseAnd | ✅ | | u8 i32 | |
| BitwiseNot | ✅ | | u8 i32 | |
| BitwiseOr | ✅ | | u8 i32 | |
| BitwiseXor | ✅ | | u8 i32 | |
| Det | ✅ | | f32 | LU 分解、バッチ対応 |
| EyeLike | ✅ | | f32 | k (対角オフセット) 対応 |

## Linear Algebra

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Gemm | ✅ | ✅ | f32 f64 i32 i64 | transA / transB / alpha / beta 対応 |
| MatMul | ✅ | ✅ | f32 f64 i32 i64 | N-D batched、float32 特化 GEMM |
| Trilu | ✅ | ✅ | f32 f64 i32 i64 | upper / lower 対応 |

## Activation

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Clip | ✅ | ✅ | f32 | min / max 入力対応 (ReLU6 等) |
| Elu | ✅ | ✅ | f32 f64 | |
| Gelu | ✅ | ✅ | f32 | opset 20 |
| HardSigmoid | ✅ | ✅ | f32 f64 | |
| HardSwish | ✅ | ✅ | f32 f64 | |
| LeakyRelu | ✅ | ✅ | f32 f64 | alpha 属性対応 |
| LogSoftmax | ✅ | ✅ | f32 | |
| Mish | ✅ | ✅ | f32 | opset 18 |
| PRelu | ✅ | ✅ | f32 | per-channel slope 対応 |
| Relu | ✅ | ✅ | f32 f64 | |
| Selu | ✅ | ✅ | f32 | |
| Sigmoid | ✅ | ✅ | f32 f64 | |
| Softmax | ✅ | ✅ | f32 f64 | opset < 13 / >= 13 対応 |
| Softplus | ✅ | ✅ | f32 | |
| Softsign | ✅ | ✅ | f32 | |
| Tanh | ✅ | ✅ | f32 f64 | |
| Celu | ✅ | | f32 | alpha 対応 |
| Hardmax | ✅ | | f32 | axis 対応 |
| Shrink | ✅ | | f32 | bias / lambd 対応 |
| Swish | ✅ | | f32 f64 | SiLU (x·sigmoid(x)) |
| ThresholdedRelu | ✅ | | f32 | alpha 閾値対応 |

## Trigonometric / Transcendental

> **Note:** Cos, Sin, Exp, Log, Erf, Sqrt 等の i32/i64 対応は ONNX 仕様外の Go 実装独自の受理です。内部で float64 に変換して計算し、元の整数型に丸めて返します。

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Acos | ✅ | ✅ | f32 | |
| Asin | ✅ | ✅ | f32 | |
| Atan | ✅ | ✅ | f32 | |
| Cos | ✅ | ✅ | f32 f64 i32 i64 | |
| Cosh | ✅ | ✅ | f32 | |
| Erf | ✅ | ✅ | f32 f64 i32 i64 | GELU 計算に使用 |
| Exp | ✅ | ✅ | f32 f64 i32 i64 | |
| Log | ✅ | ✅ | f32 f64 i32 i64 | |
| Sin | ✅ | ✅ | f32 f64 i32 i64 | |
| Sinh | ✅ | ✅ | f32 | |
| Tan | ✅ | ✅ | f32 | |
| Acosh | ✅ | | f32 | |
| Asinh | ✅ | | f32 | |
| Atanh | ✅ | | f32 | |

## Convolution / Pooling

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| AveragePool | ✅ | ✅ | f32 f64 | count_include_pad 対応 |
| Conv | ✅ | ✅ | f32 f64 | im2col+GEMM、depthwise 特化、auto_pad |
| ConvTranspose | ✅ | ✅ | f32 f64 | |
| GlobalAveragePool | ✅ | ✅ | f32 f64 | |
| GlobalMaxPool | ✅ | ✅ | f32 | |
| MaxPool | ✅ | ✅ | f32 f64 | 2×2s2 / 3×3s2 特化パス |
| Upsample | ✅ | ✅ | f32 | opset ≤ 9、nearest mode |
| GlobalLpPool | ❌ | | | |
| LpPool | ❌ | | | |
| MaxUnpool | ❌ | | | |

## Normalization

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| BatchNormalization | ✅ | ✅ | f32 f64 | 推論モード、Conv+BN 融合対応 |
| Dropout | ✅ | ✅ | f32 f64 i32 i64 u8 | 推論時パススルー |
| GroupNormalization | ✅ | ✅ | f32 | num_groups / num_channels scale |
| InstanceNormalization | ✅ | ✅ | f32 f64 | |
| LayerNormalization | ✅ | ✅ | f32 | opset 17 対応 |
| LRN | ✅ | ✅ | f32 | |
| RMSNormalization | ✅ | ✅ | f32 | SimplifiedLayerNormalization エイリアス |
| LpNormalization | ✅ | | f32 | L1 / L2 対応 |
| MeanVarianceNormalization | ✅ | | f32 | axes 指定対応 |

## Shape Operations

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Concat | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | 任意軸対応 |
| DepthToSpace | ✅ | ✅ | f32 | DCR / CRD モード対応 |
| Expand | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | |
| Flatten | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | axis 属性対応 |
| Reshape | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | zero-copy view、0 / -1 対応 |
| SpaceToDepth | ✅ | ✅ | f32 | |
| Split | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | 属性 / 入力の両方対応 |
| Squeeze | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | opset < 13 / ≥ 13 対応 |
| Tile | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | |
| Transpose | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | |
| Unsqueeze | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | opset < 13 / ≥ 13 対応 |
| Compress | ✅ | | f32 | axis / flatten 対応 |
| CenterCropPad | ❌ | | | |
| Col2Im | ❌ | | | |

## Slicing / Indexing

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Gather | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 i32、任意軸 |
| GatherElements | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| GatherND | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| NonZero | ✅ | ✅ | f32 i64 u8 | 出力: i64 |
| Pad | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | constant / reflect / edge |
| ScatterElements | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| ScatterND | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| Slice | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | 負インデックス対応 |
| TopK | ✅ | ✅ | f32 | largest / smallest 対応 |
| Unique | ✅ | | f32 | 4 出力 (values, indices, inverse, counts) |
| Scatter | ❌ | | | 旧 API (ScatterND / ScatterElements で代替) |
| TensorScatter | ❌ | | | |

## Comparison / Logic

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| And | ✅ | ✅ | bool(u8) | |
| Equal | ✅ | ✅ | f32 i64 i32 u8 i8 | 出力: bool(u8) |
| Greater | ✅ | ✅ | f32 f64 i32 i64 u8 | 出力: bool(u8) |
| GreaterOrEqual | ✅ | ✅ | f32 f64 i32 i64 u8 | 出力: bool(u8) |
| Less | ✅ | ✅ | f32 f64 i32 i64 u8 | 出力: bool(u8) |
| LessOrEqual | ✅ | ✅ | f32 f64 i32 i64 u8 | 出力: bool(u8) |
| Not | ✅ | ✅ | bool(u8) | |
| Or | ✅ | ✅ | bool(u8) | |
| Where | ✅ | ✅ | cond: u8 / data: f32 f64 i32 i64 u8 i8 | |
| Xor | ✅ | | bool(u8) | |

## Reduction

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| ArgMax | ✅ | ✅ | f32 f64 | 出力: i64 |
| ArgMin | ✅ | ✅ | f32 f64 | 出力: i64 |
| CumSum | ✅ | ✅ | f32 f64 i32 i64 | |
| ReduceL1 | ✅ | ✅ | f32 | |
| ReduceL2 | ✅ | ✅ | f32 | |
| ReduceMax | ✅ | ✅ | f32 f64 i32 i64 | keepdims 対応 |
| ReduceMean | ✅ | ✅ | f32 f64 | keepdims 対応 |
| ReduceMin | ✅ | ✅ | f32 f64 i32 i64 | keepdims 対応 |
| ReduceProd | ✅ | ✅ | f32 f64 i32 i64 | |
| ReduceSum | ✅ | ✅ | f32 f64 i32 i64 | keepdims / noop 対応 |
| ReduceSumSquare | ✅ | ✅ | f32 f64 | |
| ReduceLogSum | ✅ | | f32 | |
| ReduceLogSumExp | ✅ | | f32 | |

## Data Generation / Type Conversion

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Cast | ✅ | ✅ | f32↔f64↔i32↔i64↔u8↔i8 | 6 型相互変換 |
| CastLike | ✅ | ✅ | (Cast と同等) | target tensor の dtype |
| Constant | ✅ | ✅ | f32 f64 i32 i64 | tensor / float / int 属性 |
| ConstantOfShape | ✅ | ✅ | f32 i32 i64 u8 bool | |
| Identity | ✅ | ✅ | 全型 | グラフ最適化で除去 |
| OneHot | ✅ | ✅ | f32 i32 i64 | idx: i64 i32 f32 |
| Range | ✅ | ✅ | f32 i32 i64 | |
| Shape | ✅ | ✅ | 全型 | 出力: i64 |
| Size | ✅ | ✅ | 全型 | 出力: i64 |

## Resize / Sampling

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| GridSample | ✅ | ✅ | f32 | bilinear / nearest、align_corners |
| Resize | ✅ | ✅ | f32 | nearest / linear、coord_transform 対応 |
| AffineGrid | ❌ | | | |
| ImageDecoder | ❌ | | | |

## RNN

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| GRU | ✅ | ✅ | f32 | forward / bidirectional 対応 |
| LSTM | ✅ | ✅ | f32 | forward 対応 |
| RNN | ✅ | ✅ | f32 | Tanh 活性化 |

## Control Flow

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| If | ✅ | | — | サブグラフ実行対応 |
| Loop | ✅ | | — | エンジンランタイムで処理 |
| Scan | ✅ | | — | エンジンランタイムで処理 |

## Object Detection

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| NonMaxSuppression | ✅ | ✅ | f32 | center_point_box 対応 |
| DeformConv | ✅ | | f32 | バイリニア補間、mask 対応 |
| MaxRoiPool | ✅ | | f32 | pooled_shape / spatial_scale |
| RoiAlign | ✅ | | f32 | avg / max モード、bilinear |

## Transformer

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| Attention | ✅ | | f32 | com.microsoft、multi-head |
| RotaryEmbedding | ✅ | | f32 | BxSxNxH、cos / sin キャッシュ |

## Quantization

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| DequantizeLinear | ✅ | ✅ | 入力: u8 i8 f32 → 出力: f32 | per-axis 対応 |
| QuantizeLinear | ✅ | ✅ | 入力: f32 → 出力: u8 | |
| QLinearConv | ✅ | | u8→f32→u8 | dequant → conv → quant |
| QLinearMatMul | ✅ | | u8→f32→u8 | dequant → matmul → quant |
| DynamicQuantizeLinear | ✅ | | 入力: f32 → u8, f32, u8 | min / max 自動計算 |
| ConvInteger | ❌ | | | |
| MatMulInteger | ❌ | | | |

## Inspection

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| IsInf | ✅ | ✅ | f32 f64 | 出力: bool(u8) |
| IsNaN | ✅ | ✅ | f32 f64 | 出力: bool(u8) |

## Signal Processing

| Op | Go | テスト | 型 | 備考 |
|---|---|---|---|---|
| DFT | ✅ | | f32 | onesided / inverse 対応 |
| MelWeightMatrix | ✅ | | f32 | Hz-Mel フィルタバンク生成 |
| STFT | ✅ | | f32 | onesided / window 対応 |
| BlackmanWindow | ❌ | | | |
| HammingWindow | ❌ | | | |
| HannWindow | ❌ | | | |

## Not Planned

推論ランタイムとして優先度が低いオペレーター。

| Op | Go | 理由 |
|---|---|---|
| Bernoulli | ❌ | 乱数生成 |
| Multinomial | ❌ | 乱数生成 |
| RandomNormal | ❌ | 乱数生成 |
| RandomNormalLike | ❌ | 乱数生成 |
| RandomUniform | ❌ | 乱数生成 |
| RandomUniformLike | ❌ | 乱数生成 |
| ConcatFromSequence | ❌ | シーケンス型 |
| Optional | ❌ | Optional 型 |
| OptionalGetElement | ❌ | Optional 型 |
| OptionalHasElement | ❌ | Optional 型 |
| ReverseSequence | ❌ | シーケンス型 |
| SequenceAt | ❌ | シーケンス型 |
| SequenceConstruct | ❌ | シーケンス型 |
| SequenceEmpty | ❌ | シーケンス型 |
| SequenceErase | ❌ | シーケンス型 |
| SequenceInsert | ❌ | シーケンス型 |
| SequenceLength | ❌ | シーケンス型 |
| SequenceMap | ❌ | シーケンス型 |
| SplitToSequence | ❌ | シーケンス型 |
| NegativeLogLikelihoodLoss | ❌ | 損失関数 |
| SoftmaxCrossEntropyLoss | ❌ | 損失関数 |
| RegexFullMatch | ❌ | 文字列 / NLP |
| StringConcat | ❌ | 文字列 / NLP |
| StringNormalizer | ❌ | 文字列 / NLP |
| StringSplit | ❌ | 文字列 / NLP |
| TfIdfVectorizer | ❌ | 文字列 / NLP |

