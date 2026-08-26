# OLS による慣性パラメータ同定の実施手順

最終更新: 2026-08-26。データセット（単一 run またはマージ済み）から通常の最小二乗
(ordinary least squares, OLS) で対象物体の慣性パラメータ 10 成分を推定する手順。

## 前提と座標系

- 入力はデータセットディレクトリの `transforms.json`。各フレームに `regressor`
  （6x10 の回帰行列）と `wrench`（力覚センサのレンチ 6 成分）が必要である。
- 推定対象は 10 成分ベクトル `[mass, mx, my, mz, ixx, iyy, izz, ixy, iyz, izx]`
  （mass: kg、mx..mz: 一次モーメント kg·m、ixx..izx: 慣性テンソル kg·m²）である。
- フレーム内の `regressor` と `wrench` は力覚センサ座標系で記録される。したがって、
  それらを使った生の OLS 解 `x_sen` もセンサ座標系の量である。
- 最上位の `global_gt`、`ls`、`tls` は物体 AABB 座標系で記録される。`global_gt` と
  OLS 解を比較する前に、センサ系の解を物体 AABB 系へ移す必要がある。
- 各フレームの `pose_sen_obj` から `pose_obj_sen = pose_sen_obj.inv()` を作り、
  `transfer_iparams(pose_obj_sen, x_sen)` で座標変換する。今回のデータではセンサの
  物体に対する姿勢が z 軸 180 度回転 (`diag(-1,-1,1)`) なので、平行移動が無い場合は
  `mx`、`my`、`iyz`、`izx` の符号が反転する。
- 真値は同ファイル最上位の `global_gt`（無い run もある。その場合は対象物体の
  `xml_models/targets/<obj>/` の定義から別途取得）である。

## 手順

1. フレームを積み上げて、センサ座標系の連立系を作る。

   ```python
   import json
   import numpy as np

   t = json.load(open("<dataset>/transforms.json"))
   frames = t["frames"]
   A = np.array([f["regressor"] for f in frames]).reshape(-1, 10)  # (6N, 10)
   b = np.array([f["wrench"] for f in frames]).reshape(-1)         # (6N,)
   ```

2. センサ座標系で OLS を解く。

   ```python
   x_sen = np.linalg.lstsq(A, b, rcond=None)[0]
   ```

   これは `main.py` の `identify_inertial_params` が行う生の最小二乗解と同じである。

3. 解を物体 AABB 座標系へ移す。

   ```python
   from liegroups import SE3
   from dynamics import transfer_iparams

   pose_sen_obj = SE3.from_matrix(np.asarray(frames[0]["pose_sen_obj"], dtype=float))
   pose_obj_sen = pose_sen_obj.inv()
   x_ols = transfer_iparams(pose_obj_sen, x_sen)
   ```

   `x_ols` と `global_gt` を比較する。TLS も計算する場合は、まずセンサ系で
   `total_lstsq(A, b)[0]` を解き、同じ `transfer_iparams` を適用してから比較する。

4. 評価値を出す。

   ```python
   gt = np.array(t["global_gt"], dtype=float)
   err = x_ols - gt
   l2 = np.linalg.norm(err, 2)
   relative_residual = np.linalg.norm(A @ x_sen - b) / np.linalg.norm(b)
   ```

   `err` と `l2` は物体 AABB 系での GT との差である。一方、`relative_residual` は
   センサ系の方程式 `A @ x_sen ≈ b` の適合度であり、GT に対する精度ではない。
   したがって、座標変換漏れがある解を `global_gt` と比較した値や、低い相対残差だけを
   根拠に推定精度を判断してはならない。成分別の絶対相対誤差は
   `100 * abs(err[i]) / abs(gt[i])` で計算するが、`abs(gt[i]) < 1e-8` 程度の GT≈0
   成分は割合を算出せず `— (GT≈0)` と表示する。

## ノイズ実現の再現性 (seed)

同定結果はセンサノイズの実現ごとにばらつく。複数条件を比較するときは、乱数の種 (seed) を
固定して同じノイズ実現を再現できる状態にしてから測る。

- 実行時に `--seed <int>` を指定すると、センサノイズの乱数生成器がその種で初期化される。
  同じ種・同じ軌道・同じ設定なら、レンチと回帰行列はビット単位で一致する。

  ```sh
  pixi run -e dev python main.py --object <target> --target-trajectory <traj.json> --seed 42
  ```

- 種を指定しない場合も、実際に引かれた値が記録される。したがって過去の run も後から
  再現できる。記録先は目録ファイル最上位の `noise_seed` で、ノイズあり・なしの両系列と
  train / valid / test の各分割すべてに同じ値が載る。ファイル名は下の「系列と既知の
  注意点」にあるとおり run によって入れ替わるので、実在するファイルを確認して読む。

  ```python
  # 例: ノイズなし系列が素の名前を取っている run の場合
  seed = json.load(open("<dataset>/unperturbed_transforms.json"))["noise_seed"]
  ```

- 単一の種で測った差は、その 1 実現に固有のばらつきを含む。条件間の比較 (ノイズ源の分離、
  推定量の優劣など) を結論にする場合は、種を変えて反復し、平均と標準誤差を添える。
- `noise_seed` を持たないデータセットは、seed 対応 (2026-08-26) より前に生成されたもので、
  ノイズ実現を再現できない。

## 系列と既知の注意点

- ノイズあり・なしの系列を併記する run では、ファイル名は recorder の
  `primary_prefix` に依存する。`transforms.json` を無条件にノイズなし系列とみなさず、
  フレームの内容と `jointvars_clean` などの付加情報を確認する。今回の 600 フレーム
  run では `transforms.json` がノイズ入り、`unperturbed_transforms.json.bak` が同一 run
  のノイズなし系列である。
- マージ済みデータセットは `merge_sources` を持つ。画像・カメラ姿勢は画像側 run、
  動力学量（`pose_sen_obj`、`twist_sen`、`dtwist_sen`、`wrench`、`regressor`）は
  動力学側 run から採用される。フレーム数を揃えるための等間隔間引きの添字も同項目で
  確認する。
- **減衰バイアス**: 回帰行列 A は関節観測 (qpos/qvel/qacc) の関数で、観測ノイズが
  伝播して A 自体が汚れる。OLS は b 側のノイズしか想定しないため、A 側ノイズが
  あると推定値は真値より系統的に小さく出る（縮み率 ≈ 信号分散/(信号分散+ノイズ分散)）。
  サンプルを増やしても消えない。2026-08-25 の実測では、旧ノイズ設定
  （‖E‖_F/‖A‖_F ≈ 0.75）で mass が −38%、`sensors.py` の σ を実機準拠
  （並進 2e-5 m、回転 1e-4 rad）に下げた後（相対ノイズ 0.195）でも mz/ixx/iyy に
  −50% 級の縮みが残る。
- **バイアスへの対処**: ノイズ共分散 EᵀE が既知なら補正 LS
  `x = (AᵀA − EᵀE)⁻¹ Aᵀb` でバイアスを除去できる（Fuller のモーメント法補正。
  正式実装は `regressions/` に未収載）。
- **TLS との関係**: リポジトリの `regressions.total_lstsq` (scipy.odr) は A と b の
  誤差を等分散と仮定する。実際は A 側が 2 桁大きいため部分的にしか補正できない。
  OLS より良い場合があるが、補正 LS には及ばない。
- 等間隔間引き（マージ時の 600→300 など）は励起が十分な場合 OLS の結果を大きくは
  変えないが、必ず同じフレーム集合で再計算して確認する。

## 実行例（マージ済みデータセット）

次の例は、生のセンサ系解と、座標変換後の物体 AABB 系解を分けて計算する。

```sh
pixi run python - <<'EOF'
import json
import numpy as np
from liegroups import SE3
from dynamics import transfer_iparams

path = "datasets/hammer/merged_rotonly_nomain_lowsigma/transforms.json"
t = json.load(open(path))
frames = t["frames"]
A = np.array([f["regressor"] for f in frames], dtype=float).reshape(-1, 10)
b = np.array([f["wrench"] for f in frames], dtype=float).reshape(-1)

x_sen = np.linalg.lstsq(A, b, rcond=None)[0]
pose_sen_obj = SE3.from_matrix(np.asarray(frames[0]["pose_sen_obj"], dtype=float))
x = transfer_iparams(pose_sen_obj.inv(), x_sen)
gt = np.array(t["global_gt"], dtype=float)
err = x - gt

names = "mass mx my mz ixx iyy izz ixy iyz izx".split()
for n, xi, gi, ei in zip(names, x, gt, err):
    if abs(gi) < 1e-8:
        pct = "— (GT≈0)"
    else:
        pct = f"{100 * abs(ei) / abs(gi):.6g}%"
    print(f"{n:5s} est={xi: .12g}  gt={gi: .12g}  err={ei:+.12g}  rel={pct}")
print("L2 =", np.linalg.norm(err, 2))
print("relative residual (sensor equation) =", np.linalg.norm(A @ x_sen - b) / np.linalg.norm(b))
EOF
```
