package ops

// depthwiseF32 implements depthwise convolution directly without im2col.
// (n,c) 平面ごとに独立なため maxWorkers>1 でチャネル並列実行する。
func depthwiseF32(X, W, bias []float32, N, C, H, Wi, KH, KW, OH, OW, strideH, strideW, padTop, padLeft, maxWorkers int) []float32 {
	out := make([]float32, N*C*OH*OW)
	OHOW := OH * OW
	HW := H * Wi
	KHKW := KH * KW

	// ow がこの範囲なら窓が水平方向に完全に内側(境界チェック不要)
	owLo := 0
	if padLeft > 0 {
		owLo = (padLeft + strideW - 1) / strideW
	}
	owHi := OW
	if hi := (Wi - KW + padLeft) / strideW; hi+1 < owHi {
		owHi = hi + 1
	}
	if owHi < owLo {
		owHi = owLo
	}

	forEachIndexParallel(N*C, maxWorkers, func(nc int) {
		c := nc % C
		xBase := nc * HW
		oBase := nc * OHOW

		bv := float32(0)
		if bias != nil {
			bv = bias[c]
		}
		wBase := c * KHKW

		if KH == 3 && KW == 3 {
			wSlice := W[wBase : wBase+9 : wBase+9] // BCE
			w00 := wSlice[0]
			w01 := wSlice[1]
			w02 := wSlice[2]
			w10 := wSlice[3]
			w11 := wSlice[4]
			w12 := wSlice[5]
			w20 := wSlice[6]
			w21 := wSlice[7]
			w22 := wSlice[8]

			for oh := 0; oh < OH; oh++ {
				ih0 := oh*strideH - padTop
				for ow := 0; ow < OW; ow++ {
					iw0 := ow*strideW - padLeft
					sum := bv

					if ih0 >= 0 && ih0+2 < H && iw0 >= 0 && iw0+2 < Wi {
						r0 := xBase + ih0*Wi + iw0
						r1 := r0 + Wi
						r2 := r1 + Wi
						_ = X[r2+2] // BCE hint
						sum += X[r0]*w00 + X[r0+1]*w01 + X[r0+2]*w02
						sum += X[r1]*w10 + X[r1+1]*w11 + X[r1+2]*w12
						sum += X[r2]*w20 + X[r2+1]*w21 + X[r2+2]*w22
					} else {
						for kh := 0; kh < 3; kh++ {
							ih := ih0 + kh
							if ih < 0 || ih >= H {
								continue
							}
							row := xBase + ih*Wi
							for kw := 0; kw < 3; kw++ {
								iw := iw0 + kw
								if iw >= 0 && iw < Wi {
									sum += X[row+iw] * W[wBase+kh*3+kw]
								}
							}
						}
					}

					out[oBase+oh*OW+ow] = sum
				}
			}
			return
		}

		// 汎用 K×K: 窓が完全に内側の領域は境界チェックなしのタイトループで回す
		for oh := 0; oh < OH; oh++ {
			ih0 := oh*strideH - padTop
			rowInside := ih0 >= 0 && ih0+KH <= H
			outRow := oBase + oh*OW

			if rowInside {
				for ow := owLo; ow < owHi; ow++ {
					iw0 := ow*strideW - padLeft
					sum := bv
					for kh := 0; kh < KH; kh++ {
						rs := xBase + (ih0+kh)*Wi + iw0
						xr := X[rs : rs+KW : rs+KW]                            // BCE
						wr := W[wBase+kh*KW : wBase+kh*KW+KW : wBase+kh*KW+KW] // BCE
						for k, wv := range wr {
							sum += xr[k] * wv
						}
					}
					out[outRow+ow] = sum
				}
			}

			// 境界(または行全体が境界にかかる場合)は従来どおり境界チェック付き
			for ow := 0; ow < OW; ow++ {
				if rowInside && ow >= owLo && ow < owHi {
					ow = owHi - 1
					continue
				}
				iw0 := ow*strideW - padLeft
				sum := bv
				for kh := 0; kh < KH; kh++ {
					ih := ih0 + kh
					if ih < 0 || ih >= H {
						continue
					}
					row := xBase + ih*Wi
					for kw := 0; kw < KW; kw++ {
						iw := iw0 + kw
						if iw >= 0 && iw < Wi {
							sum += X[row+iw] * W[wBase+kh*KW+kw]
						}
					}
				}
				out[outRow+ow] = sum
			}
		}
	})
	return out
}
