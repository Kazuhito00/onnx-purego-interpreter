package ops

// depthwiseF32 implements depthwise convolution directly without im2col.
func depthwiseF32(X, W, bias []float32, N, C, H, Wi, KH, KW, OH, OW, strideH, strideW, padTop, padLeft int) []float32 {
	out := make([]float32, N*C*OH*OW)
	OHOW := OH * OW
	HW := H * Wi
	KHKW := KH * KW

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			xBase := n*C*HW + c*HW
			oBase := n*C*OHOW + c*OHOW

			bv := float32(0)
			if bias != nil {
				bv = bias[c]
			}

			if KH == 3 && KW == 3 {
				wSlice := W[c*KHKW : c*KHKW+9 : c*KHKW+9] // BCE
				w00 := wSlice[0]; w01 := wSlice[1]; w02 := wSlice[2]
				w10 := wSlice[3]; w11 := wSlice[4]; w12 := wSlice[5]
				w20 := wSlice[6]; w21 := wSlice[7]; w22 := wSlice[8]

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
							wBase := c * KHKW
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
			} else {
				wBase := c * KHKW
				for oh := 0; oh < OH; oh++ {
					ih0 := oh*strideH - padTop
					for ow := 0; ow < OW; ow++ {
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
						out[oBase+oh*OW+ow] = sum
					}
				}
			}
		}
	}
	return out
}
