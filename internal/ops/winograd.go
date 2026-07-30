package ops

// Winograd F(2x2, 3x3) 畳み込み。
// 3x3 stride1 dilation1 group1 の float32 conv を、2x2 出力タイルあたり
// 16 回の乗算(直接法は 36 回)で計算する。理論 FLOP は 1/2.25。
//
//	Y = A^T [ (G g G^T) ⊙ (B^T d B) ] A
//
// 変換行列 (Lavin & Gray, 2016):
//
//	B^T = [1  0 -1  0; 0  1  1  0; 0 -1  1  0; 0  1  0 -1]
//	G   = [1 0 0; 1/2 1/2 1/2; 1/2 -1/2 1/2; 0 0 1]
//	A^T = [1 1 1 0; 0 1 -1 -1]
//
// 直接法と浮動小数の丸めが異なるため結果はビット一致しない(相対 ~1e-6 級)。
// UseWinograd で無効化できる。

// winogradApplicable は F(2x2,3x3) の適用条件を判定する。
// 変換コストと 16 分割 GEMM(K=C が短くなる)のオーバーヘッドがあるため、
// チャネル数とタイル数が十分大きい GEMM 律速の conv に限って適用する
// (小さい conv は直接法 = strip + packing の方が速い)。
func winogradApplicable(kc *KernelConfig, group, KH, KW, strideH, strideW, dilH, dilW, C, OC, OH, OW int) bool {
	if kc != nil && !kc.UseWinograd {
		return false
	}
	tiles := ((OH + 1) / 2) * ((OW + 1) / 2)
	return group == 1 && KH == 3 && KW == 3 &&
		strideH == 1 && strideW == 1 && dilH == 1 && dilW == 1 &&
		C >= 64 && OC >= 64 && tiles >= 900
}

// convWinogradF32 は 3x3 s1 conv を Winograd F(2x2,3x3) で計算する。
// bias は nil 可。出力バッファ out(長さ N*OC*OH*OW、ゼロ初期化済み)へ書き込む。
func convWinogradF32(xf, wf, bias, out []float32, N, C, H, W, OC, OH, OW, padTop, padLeft, maxWorkers int) {
	tilesH := (OH + 1) / 2
	tilesW := (OW + 1) / 2
	tilesPerImage := tilesH * tilesW
	P := N * tilesPerImage

	// 重み変換 U[u][oc][c] = (G g G^T)[u]
	U := getFloat32Scratch(16 * OC * C)
	forEachIndexParallel(OC, min(maxWorkers, OC), func(oc int) {
		for c := 0; c < C; c++ {
			g := wf[(oc*C+c)*9 : (oc*C+c)*9+9]
			// t = G g (4x3)
			var t [4][3]float32
			for j := 0; j < 3; j++ {
				g0, g1, g2 := g[j], g[3+j], g[6+j]
				t[0][j] = g0
				t[1][j] = 0.5 * (g0 + g1 + g2)
				t[2][j] = 0.5 * (g0 - g1 + g2)
				t[3][j] = g2
			}
			// u = t G^T (4x4)
			base := oc*C + c
			for i := 0; i < 4; i++ {
				a0, a1, a2 := t[i][0], t[i][1], t[i][2]
				U[(i*4+0)*OC*C+base] = a0
				U[(i*4+1)*OC*C+base] = 0.5 * (a0 + a1 + a2)
				U[(i*4+2)*OC*C+base] = 0.5 * (a0 - a1 + a2)
				U[(i*4+3)*OC*C+base] = a2
			}
		}
	})

	// タイルブロックサイズ: V+M が ~1.5MB に収まるように選ぶ
	pb := (3 << 18) / (16 * (C + OC)) // 1.5MB/4byte = 384K floats
	if pb < 64 {
		pb = 64
	}
	if pb > 1024 {
		pb = 1024
	}
	if pb > P {
		pb = P
	}
	numBlocks := (P + pb - 1) / pb

	// ブロックは互いに独立なので、ブロック単位で並列実行する
	forEachIndexParallel(numBlocks, min(maxWorkers, numBlocks), func(blk int) {
		p0 := blk * pb
		p1 := min(p0+pb, P)
		bLen := p1 - p0

		V := getFloat32Scratch(16 * C * bLen)
		M := getFloat32Scratch(16 * OC * bLen)

		// 入力変換: V[u][c][pi] = (B^T d B)[u]
		for p := p0; p < p1; p++ {
			pi := p - p0
			n := p / tilesPerImage
			tp := p % tilesPerImage
			ih0 := (tp/tilesW)*2 - padTop
			iw0 := (tp%tilesW)*2 - padLeft
			for c := 0; c < C; c++ {
				xBase := (n*C + c) * H * W
				// 4x4 パッチをゼロパディング込みで読む
				var d [4][4]float32
				for r := 0; r < 4; r++ {
					ih := ih0 + r
					if ih < 0 || ih >= H {
						continue
					}
					row := xBase + ih*W
					for q := 0; q < 4; q++ {
						iw := iw0 + q
						if iw >= 0 && iw < W {
							d[r][q] = xf[row+iw]
						}
					}
				}
				// 列方向: t = B^T d
				var t [4][4]float32
				for j := 0; j < 4; j++ {
					d0, d1, d2, d3 := d[0][j], d[1][j], d[2][j], d[3][j]
					t[0][j] = d0 - d2
					t[1][j] = d1 + d2
					t[2][j] = d2 - d1
					t[3][j] = d1 - d3
				}
				// 行方向: v = t B
				dst := c*bLen + pi
				for i := 0; i < 4; i++ {
					a0, a1, a2, a3 := t[i][0], t[i][1], t[i][2], t[i][3]
					V[(i*4+0)*C*bLen+dst] = a0 - a2
					V[(i*4+1)*C*bLen+dst] = a1 + a2
					V[(i*4+2)*C*bLen+dst] = a2 - a1
					V[(i*4+3)*C*bLen+dst] = a1 - a3
				}
			}
		}

		// 16 位置それぞれで GEMM: M[u] = U[u] (OC×C) × V[u] (C×bLen)
		clear(M[:16*OC*bLen])
		for u := 0; u < 16; u++ {
			convGemmF32(U[u*OC*C:(u+1)*OC*C], V[u*C*bLen:(u+1)*C*bLen],
				M[u*OC*bLen:(u+1)*OC*bLen], OC, bLen, C)
		}

		// 出力逆変換: Y = A^T m A(2x2)+ bias
		for p := p0; p < p1; p++ {
			pi := p - p0
			n := p / tilesPerImage
			tp := p % tilesPerImage
			oh0 := (tp / tilesW) * 2
			ow0 := (tp % tilesW) * 2
			for oc := 0; oc < OC; oc++ {
				src := oc*bLen + pi
				var m [4][4]float32
				for i := 0; i < 4; i++ {
					for j := 0; j < 4; j++ {
						m[i][j] = M[(i*4+j)*OC*bLen+src]
					}
				}
				// 列方向: t = A^T m (2x4)
				var t [2][4]float32
				for j := 0; j < 4; j++ {
					m0, m1, m2, m3 := m[0][j], m[1][j], m[2][j], m[3][j]
					t[0][j] = m0 + m1 + m2
					t[1][j] = m1 - m2 - m3
				}
				bv := float32(0)
				if bias != nil {
					bv = bias[oc]
				}
				y00 := t[0][0] + t[0][1] + t[0][2] + bv
				y01 := t[0][1] - t[0][2] - t[0][3] + bv
				y10 := t[1][0] + t[1][1] + t[1][2] + bv
				y11 := t[1][1] - t[1][2] - t[1][3] + bv

				oBase := (n*OC+oc)*OH*OW + oh0*OW + ow0
				out[oBase] = y00
				if ow0+1 < OW {
					out[oBase+1] = y01
				}
				if oh0+1 < OH {
					out[oBase+OW] = y10
					if ow0+1 < OW {
						out[oBase+OW+1] = y11
					}
				}
			}
		}

		putFloat32Scratch(M)
		putFloat32Scratch(V)
	})
	putFloat32Scratch(U)
}
