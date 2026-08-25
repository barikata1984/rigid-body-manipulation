# OLS による慣性パラメータ同定の実施手順

最終更新: 2026-08-25。データセット（単一 run またはマージ済み）から通常の最小二乗
(ordinary least squares, OLS) で対象物体の慣性パラメータ 10 成分を推定する手順。

## 前提

- 入力はデータセットディレクトリの `transforms.json`。各フレームに `regressor`
  （6x10 の回帰行列。剛体の運動方程式を慣性パラメータについて線形化した係数行列）と
  `wrench`（力覚センサのレンチ 6 成分）を含むこと。
- 推定対象は 10 成分ベクトル `[mass, mx, my, mz, ixx, iyy, izz, ixy, iyz, izx]`
  （mass: kg、mx..mz: 一次モーメント kg·m、ixx..izx: 慣性テンソル kg·m²）。
- 真値は同ファイル最上位の `global_gt`（無い run もある。その場合は対象物体の
  `xml_models/targets/<obj>/` の定義から別途取得）。
- ノイズ系列の確認: 現行 `main.py` はノイズなし系列を素の `transforms.json` に書く
  （`main.py:148-151`）。ノイズ入りで同定したい場合は `transforms.json.bak`
  （ノイズ入り）と取り違えないこと。マージ済みデータセットは `merge_sources` を持ち、
  マージ時に選ばれた系列が入っている。

## 手順

1. フレームを積み上げて連立系を作る。

   ```python
   import json, numpy as np
   t = json.load(open("<dataset>/transforms.json"))
   A = np.array([f["regressor"] for f in t["frames"]]).reshape(-1, 10)  # (6N, 10)
   b = np.array([f["wrench"]    for f in t["frames"]]).reshape(-1)     # (6N,)
   ```

2. OLS を解く。

   ```python
   x_ols = np.linalg.lstsq(A, b, rcond=None)[0]
   ```

   `main.py:124` の同定処理と同一（`numpy.linalg.lstsq`）。

3. 評価。真値があれば成分ごとの誤差（物理単位）と L2 ノルムを出す。

   ```python
   gt = np.array(t["global_gt"])
   err = x_ols - gt
   l2  = np.linalg.norm(err)
   ```

   L2 だけでなく成分別誤差を必ず見る。誤差は mass / mz / ixx / iyy に集中する
   傾向がある（下記の減衰バイアス）。

## 既知の注意点

- **減衰バイアス**: 回帰行列 A は関節観測（qpos/qvel/qacc）の関数で、観測ノイズが
  伝播して A 自体が汚れる。OLS は b 側のノイズしか想定しないため、A 側ノイズが
  あると推定値は真値より系統的に小さく出る（縮み率 ≈ 信号分散/(信号分散+ノイズ分散)）。
  サンプルを増やしても消えない。2026-08-25 の実測では、旧ノイズ設定
  （‖E‖_F/‖A‖_F ≈ 0.75）で mass が −38%、`sensors.py` の σ を実機準拠
  （並進 2e-5 m、回転 1e-4 rad）に下げた後（相対ノイズ 0.195）でも mz/ixx/iyy に
  −50% 級の縮みが残る。
- **バイアスへの対処**: ノイズ共分散 EᵀE が既知なら補正 LS
  `x = (AᵀA − EᵀE)⁻¹ Aᵀb` でバイアスを除去できる（Fuller のモーメント法補正。
  検証スクリプトは 2026-08-25 セッションの scratchpad `eiv_compare.py`、
  結果はセッション議事録参照）。正式実装は `regressions/` に未収載。
- **TLS との関係**: リポジトリの `regressions.total_lstsq`（scipy.odr）は A と b の
  誤差を等分散と仮定する。実際は A 側が 2 桁大きいため部分的にしか補正できない。
  OLS より良いが補正 LS には及ばない。
- 等間隔間引き（マージ時の 600→300 など）は OLS の結果をほぼ変えない
  （実測で L2 +1% 程度）。

## 実行例（マージ済みデータセット）

```sh
pixi run python - <<'EOF'
import json, numpy as np
t = json.load(open("datasets/hammer/merged_rotonly_nomain_lowsigma/transforms.json"))
A = np.array([f["regressor"] for f in t["frames"]]).reshape(-1, 10)
b = np.array([f["wrench"] for f in t["frames"]]).reshape(-1)
x = np.linalg.lstsq(A, b, rcond=None)[0]
gt = np.array(t["global_gt"])
for n, xi, gi in zip("mass mx my mz ixx iyy izz ixy iyz izx".split(), x, gt):
    print(f"{n:5s} est={xi:9.5f}  gt={gi:9.5f}  err={xi-gi:+9.5f}")
print("L2 =", np.linalg.norm(x - gt))
EOF
```
