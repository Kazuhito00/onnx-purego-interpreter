# Operator Coverage

Approximately 80% of ~200 ONNX standard operators implemented.<br>
ONNX 標準オペレーター約 200 個中、約 80% を実装。

## Legend / 凡例

| Symbol / 記号 | Meaning / 意味 |
|---|---|
| ✅ | Supported / 対応済み |
| ❌ | Not supported / 未対応 |
| **Go** | Implementation status / 実装状況 |
| **Test** | Python ORT comparison test (`generate_test_models.py`) / Python ORT 比較テストの有無 |
| **Types** | Supported data types / 対応データ型 |

**Type abbreviations / 型の略記:** f32=float32, f64=float64, i32=int32, i64=int64, u8=uint8, i8=int8, bool=uint8 boolean

---

## Arithmetic

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Abs | ✅ | ✅ | f32 f64 i32 i64 | |
| Add | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting |
| Ceil | ✅ | ✅ | f32 | |
| Div | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting |
| Einsum | ✅ | ✅ | f32 | 1-input (transpose) / 2-input (contraction) |
| Floor | ✅ | ✅ | f32 f64 i32 i64 | |
| Max | ✅ | ✅ | f32 | |
| Mean | ✅ | ✅ | f32 | |
| Min | ✅ | ✅ | f32 | |
| Mod | ✅ | ✅ | f32 f64 i32 i64 | fmod attr |
| Mul | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting |
| Neg | ✅ | ✅ | f32 f64 i32 i64 | |
| Pow | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting |
| Reciprocal | ✅ | ✅ | f32 f64 i32 i64 | |
| Round | ✅ | ✅ | f32 | Banker's rounding |
| Sign | ✅ | ✅ | f32 | |
| Sqrt | ✅ | ✅ | f32 f64 i32 i64 | |
| Sub | ✅ | ✅ | f32 f64 i32 i64 | Broadcasting |
| Sum | ✅ | ✅ | f32 | |
| BitShift | ✅ | | u8 | LEFT / RIGHT |
| BitwiseAnd | ✅ | | u8 i32 | |
| BitwiseNot | ✅ | | u8 i32 | |
| BitwiseOr | ✅ | | u8 i32 | |
| BitwiseXor | ✅ | | u8 i32 | |
| Det | ✅ | | f32 | LU decomposition, batched / LU 分解、バッチ対応 |
| EyeLike | ✅ | | f32 | k (diagonal offset / 対角オフセット) |

## Linear Algebra

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Gemm | ✅ | ✅ | f32 f64 i32 i64 | transA / transB / alpha / beta |
| MatMul | ✅ | ✅ | f32 f64 i32 i64 | N-D batched, float32 specialized GEMM / float32 特化 GEMM |
| Trilu | ✅ | ✅ | f32 f64 i32 i64 | upper / lower |

## Activation

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Clip | ✅ | ✅ | f32 | min / max input (ReLU6 etc.) |
| Elu | ✅ | ✅ | f32 f64 | |
| Gelu | ✅ | ✅ | f32 | opset 20 |
| HardSigmoid | ✅ | ✅ | f32 f64 | |
| HardSwish | ✅ | ✅ | f32 f64 | |
| LeakyRelu | ✅ | ✅ | f32 f64 | alpha attr |
| LogSoftmax | ✅ | ✅ | f32 | |
| Mish | ✅ | ✅ | f32 | opset 18 |
| PRelu | ✅ | ✅ | f32 | per-channel slope |
| Relu | ✅ | ✅ | f32 f64 | |
| Selu | ✅ | ✅ | f32 | |
| Sigmoid | ✅ | ✅ | f32 f64 | |
| Softmax | ✅ | ✅ | f32 f64 | opset < 13 / >= 13 |
| Softplus | ✅ | ✅ | f32 | |
| Softsign | ✅ | ✅ | f32 | |
| Tanh | ✅ | ✅ | f32 f64 | |
| Celu | ✅ | | f32 | alpha |
| Hardmax | ✅ | | f32 | axis |
| Shrink | ✅ | | f32 | bias / lambd |
| Swish | ✅ | | f32 f64 | SiLU (x·sigmoid(x)) |
| ThresholdedRelu | ✅ | | f32 | alpha threshold / alpha 閾値 |

## Trigonometric / Transcendental

> **Note:** i32/i64 support for Cos, Sin, Exp, Log, Erf, Sqrt is a Go implementation extension beyond the ONNX spec. Values are converted to float64 internally and rounded back to the original integer type.<br>
> **注:** Cos, Sin, Exp, Log, Erf, Sqrt 等の i32/i64 対応は ONNX 仕様外の Go 実装独自の拡張。内部で float64 に変換して計算し、元の整数型に丸めて返す。

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Acos | ✅ | ✅ | f32 | |
| Asin | ✅ | ✅ | f32 | |
| Atan | ✅ | ✅ | f32 | |
| Cos | ✅ | ✅ | f32 f64 i32 i64 | |
| Cosh | ✅ | ✅ | f32 | |
| Erf | ✅ | ✅ | f32 f64 i32 i64 | Used in GELU / GELU 計算に使用 |
| Exp | ✅ | ✅ | f32 f64 i32 i64 | |
| Log | ✅ | ✅ | f32 f64 i32 i64 | |
| Sin | ✅ | ✅ | f32 f64 i32 i64 | |
| Sinh | ✅ | ✅ | f32 | |
| Tan | ✅ | ✅ | f32 | |
| Acosh | ✅ | | f32 | |
| Asinh | ✅ | | f32 | |
| Atanh | ✅ | | f32 | |

## Convolution / Pooling

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| AveragePool | ✅ | ✅ | f32 f64 | count_include_pad |
| Conv | ✅ | ✅ | f32 f64 | im2col+GEMM, depthwise specialized, auto_pad / depthwise 特化 |
| ConvTranspose | ✅ | ✅ | f32 f64 | |
| GlobalAveragePool | ✅ | ✅ | f32 f64 | |
| GlobalMaxPool | ✅ | ✅ | f32 | |
| MaxPool | ✅ | ✅ | f32 f64 | 2×2s2 / 3×3s2 fast path / 特化パス |
| Upsample | ✅ | ✅ | f32 | opset ≤ 9, nearest mode |
| GlobalLpPool | ❌ | | | |
| LpPool | ❌ | | | |
| MaxUnpool | ❌ | | | |

## Normalization

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| BatchNormalization | ✅ | ✅ | f32 f64 | Inference mode, Conv+BN fusion / 推論モード、Conv+BN 融合対応 |
| Dropout | ✅ | ✅ | f32 f64 i32 i64 u8 | Passthrough at inference / 推論時パススルー |
| GroupNormalization | ✅ | ✅ | f32 | num_groups / num_channels scale |
| InstanceNormalization | ✅ | ✅ | f32 f64 | |
| LayerNormalization | ✅ | ✅ | f32 | opset 17 |
| LRN | ✅ | ✅ | f32 | |
| RMSNormalization | ✅ | ✅ | f32 | SimplifiedLayerNormalization alias / エイリアス |
| LpNormalization | ✅ | | f32 | L1 / L2 |
| MeanVarianceNormalization | ✅ | | f32 | axes |

## Shape Operations

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Concat | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | Any axis / 任意軸 |
| DepthToSpace | ✅ | ✅ | f32 | DCR / CRD mode |
| Expand | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | |
| Flatten | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | axis attr |
| Reshape | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | zero-copy view, 0 / -1 |
| SpaceToDepth | ✅ | ✅ | f32 | |
| Split | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | attr / input both / 属性・入力の両方対応 |
| Squeeze | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | opset < 13 / ≥ 13 |
| Tile | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | |
| Transpose | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | |
| Unsqueeze | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | opset < 13 / ≥ 13 |
| Compress | ✅ | | f32 | axis / flatten |
| CenterCropPad | ❌ | | | |
| Col2Im | ❌ | | | |

## Slicing / Indexing

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Gather | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 i32, any axis / 任意軸 |
| GatherElements | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| GatherND | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| NonZero | ✅ | ✅ | f32 i64 u8 | output: i64 / 出力: i64 |
| Pad | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | constant / reflect / edge |
| ScatterElements | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| ScatterND | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | idx: i64 |
| Slice | ✅ | ✅ | f32 f64 i32 i64 u8 i8 | Negative index / 負インデックス対応 |
| TopK | ✅ | ✅ | f32 | largest / smallest |
| Unique | ✅ | | f32 | 4 outputs (values, indices, inverse, counts) / 4 出力 |
| Scatter | ❌ | | | Legacy API (use ScatterND/ScatterElements) / 旧 API |
| TensorScatter | ❌ | | | |

## Comparison / Logic

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| And | ✅ | ✅ | bool(u8) | |
| Equal | ✅ | ✅ | f32 i64 i32 u8 i8 | output: bool(u8) / 出力: bool(u8) |
| Greater | ✅ | ✅ | f32 f64 i32 i64 u8 | output: bool(u8) |
| GreaterOrEqual | ✅ | ✅ | f32 f64 i32 i64 u8 | output: bool(u8) |
| Less | ✅ | ✅ | f32 f64 i32 i64 u8 | output: bool(u8) |
| LessOrEqual | ✅ | ✅ | f32 f64 i32 i64 u8 | output: bool(u8) |
| Not | ✅ | ✅ | bool(u8) | |
| Or | ✅ | ✅ | bool(u8) | |
| Where | ✅ | ✅ | cond: u8 / data: f32 f64 i32 i64 u8 i8 | |
| Xor | ✅ | | bool(u8) | |

## Reduction

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| ArgMax | ✅ | ✅ | f32 f64 | output: i64 / 出力: i64 |
| ArgMin | ✅ | ✅ | f32 f64 | output: i64 / 出力: i64 |
| CumSum | ✅ | ✅ | f32 f64 i32 i64 | |
| ReduceL1 | ✅ | ✅ | f32 | |
| ReduceL2 | ✅ | ✅ | f32 | |
| ReduceMax | ✅ | ✅ | f32 f64 i32 i64 | keepdims |
| ReduceMean | ✅ | ✅ | f32 f64 | keepdims |
| ReduceMin | ✅ | ✅ | f32 f64 i32 i64 | keepdims |
| ReduceProd | ✅ | ✅ | f32 f64 i32 i64 | |
| ReduceSum | ✅ | ✅ | f32 f64 i32 i64 | keepdims / noop |
| ReduceSumSquare | ✅ | ✅ | f32 f64 | |
| ReduceLogSum | ✅ | | f32 | |
| ReduceLogSumExp | ✅ | | f32 | |

## Data Generation / Type Conversion

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Cast | ✅ | ✅ | f32↔f64↔i32↔i64↔u8↔i8 | 6-type mutual conversion / 6 型相互変換 |
| CastLike | ✅ | ✅ | (same as Cast) | target tensor dtype |
| Constant | ✅ | ✅ | f32 f64 i32 i64 | tensor / float / int attr |
| ConstantOfShape | ✅ | ✅ | f32 i32 i64 u8 bool | |
| Identity | ✅ | ✅ | all types / 全型 | Eliminated by graph optimization / グラフ最適化で除去 |
| OneHot | ✅ | ✅ | f32 i32 i64 | idx: i64 i32 f32 |
| Range | ✅ | ✅ | f32 i32 i64 | |
| Shape | ✅ | ✅ | all types / 全型 | output: i64 / 出力: i64 |
| Size | ✅ | ✅ | all types / 全型 | output: i64 / 出力: i64 |

## Resize / Sampling

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| GridSample | ✅ | ✅ | f32 | bilinear / nearest, align_corners |
| Resize | ✅ | ✅ | f32 | nearest / linear, coord_transform |
| AffineGrid | ❌ | | | |
| ImageDecoder | ❌ | | | |

## RNN

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| GRU | ✅ | ✅ | f32 | forward / bidirectional |
| LSTM | ✅ | ✅ | f32 | forward |
| RNN | ✅ | ✅ | f32 | Tanh activation / Tanh 活性化 |

## Control Flow

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| If | ✅ | | — | Subgraph execution / サブグラフ実行対応 |
| Loop | ✅ | | — | Engine runtime / エンジンランタイムで処理 |
| Scan | ✅ | | — | Engine runtime / エンジンランタイムで処理 |

## Object Detection

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| NonMaxSuppression | ✅ | ✅ | f32 | center_point_box |
| DeformConv | ✅ | | f32 | Bilinear interpolation, mask / バイリニア補間、mask 対応 |
| MaxRoiPool | ✅ | | f32 | pooled_shape / spatial_scale |
| RoiAlign | ✅ | | f32 | avg / max mode, bilinear |

## Transformer

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| Attention | ✅ | | f32 | com.microsoft, multi-head |
| RotaryEmbedding | ✅ | | f32 | BxSxNxH, cos / sin cache / cos / sin キャッシュ |

## Quantization

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| DequantizeLinear | ✅ | ✅ | in: u8 i8 f32 → out: f32 | per-axis |
| QuantizeLinear | ✅ | ✅ | in: f32 → out: u8 | |
| QLinearConv | ✅ | | u8→f32→u8 | dequant → conv → quant |
| QLinearMatMul | ✅ | | u8→f32→u8 | dequant → matmul → quant |
| DynamicQuantizeLinear | ✅ | | in: f32 → u8, f32, u8 | Auto min/max / min / max 自動計算 |
| ConvInteger | ❌ | | | |
| MatMulInteger | ❌ | | | |

## Inspection

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| IsInf | ✅ | ✅ | f32 f64 | output: bool(u8) / 出力: bool(u8) |
| IsNaN | ✅ | ✅ | f32 f64 | output: bool(u8) / 出力: bool(u8) |

## Signal Processing

| Op | Go | Test | Types | Notes / 備考 |
|---|---|---|---|---|
| DFT | ✅ | | f32 | onesided / inverse |
| MelWeightMatrix | ✅ | | f32 | Hz-Mel filter bank / Hz-Mel フィルタバンク生成 |
| STFT | ✅ | | f32 | onesided / window |
| BlackmanWindow | ❌ | | | |
| HammingWindow | ❌ | | | |
| HannWindow | ❌ | | | |

## Not Planned

Operators with low priority for an inference runtime.<br>
推論ランタイムとして優先度が低いオペレーター。

| Op | Go | Reason / 理由 |
|---|---|---|
| Bernoulli | ❌ | Random generation / 乱数生成 |
| Multinomial | ❌ | Random generation / 乱数生成 |
| RandomNormal | ❌ | Random generation / 乱数生成 |
| RandomNormalLike | ❌ | Random generation / 乱数生成 |
| RandomUniform | ❌ | Random generation / 乱数生成 |
| RandomUniformLike | ❌ | Random generation / 乱数生成 |
| ConcatFromSequence | ❌ | Sequence type / シーケンス型 |
| Optional | ❌ | Optional type / Optional 型 |
| OptionalGetElement | ❌ | Optional type / Optional 型 |
| OptionalHasElement | ❌ | Optional type / Optional 型 |
| ReverseSequence | ❌ | Sequence type / シーケンス型 |
| SequenceAt | ❌ | Sequence type / シーケンス型 |
| SequenceConstruct | ❌ | Sequence type / シーケンス型 |
| SequenceEmpty | ❌ | Sequence type / シーケンス型 |
| SequenceErase | ❌ | Sequence type / シーケンス型 |
| SequenceInsert | ❌ | Sequence type / シーケンス型 |
| SequenceLength | ❌ | Sequence type / シーケンス型 |
| SequenceMap | ❌ | Sequence type / シーケンス型 |
| SplitToSequence | ❌ | Sequence type / シーケンス型 |
| NegativeLogLikelihoodLoss | ❌ | Loss function / 損失関数 |
| SoftmaxCrossEntropyLoss | ❌ | Loss function / 損失関数 |
| RegexFullMatch | ❌ | String / NLP / 文字列 / NLP |
| StringConcat | ❌ | String / NLP / 文字列 / NLP |
| StringNormalizer | ❌ | String / NLP / 文字列 / NLP |
| StringSplit | ❌ | String / NLP / 文字列 / NLP |
| TfIdfVectorizer | ❌ | String / NLP / 文字列 / NLP |
