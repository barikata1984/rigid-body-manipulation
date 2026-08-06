# 問い合わせ: `loaded_dice` データセットの `wrench` が自身の `regressor` / `global_gt` と整合しない

日付: 2026-08-05
対象: `datasets/neural-mass-fields/loaded_dice/transforms.json`（300 frames）
対照: `datasets/realworld_nemd/hammer_spline_20260731_113007_run1/unperturbed_transforms.json`（同 300 frames、実機）
宛先: データ生成側

## 結論

`loaded_dice` の各フレームの `wrench` は、**同じ JSON に同梱されている `regressor` (6×10)、`twist_sen` / `dtwist_sen`、
`global_gt` のどれとも整合しません**。剛体パラメータをどう選んでも force 信号の 84%、torque 信号の 95% が残ります。
実機の `hammer_spline_run1` は同じ検査で R² = 0.9999 / 0.9998 なので、検査手順とこちらの規約解釈の問題ではありません。

学習側にバグは見つかりませんでした。訓練は理論最適値に到達しています（後述）。データ側の再生成が必要と考えます。

## 検証内容と数値

すべて訓練を通さず、JSON 内の値だけで再現できます（`regressor`, `wrench`, `global_gt`, `labels` を使用）。

### 1. `regressor @ global_gt` と `wrench` の比較

| | loaded_dice | hammer（対照） |
|---|---|---|
| `wrench` force RMS | 2.653 N | 6.653 N |
| `regressor @ global_gt` との残差 | **2.445 N** | 0.065 N |
| 10 パラメータ最小二乗の最良残差 | **2.219 N**（force）/ **0.107 N·m**（torque） | 0.065 N / 0.011 N·m |
| 最良フィットの説明分散 R²（force） | **+0.3005** | +0.9999 |
| 最良フィットの説明分散 R²（torque） | **+0.0503** | +0.9998 |
| LS 推定質量 / `global_gt` 質量 | 0.2030 / 0.3310 = **0.613** | 1.1163 / 1.1161 = **1.000** |
| `cond(regressor)` | 6.30 | 7.14 |

`cond` が両者 6〜7 なので、軌道の幾何的縮退ではありません。**torque は R² = 0.05、つまり実質的に
自身の運動学と無相関**です。

### 2. torque の大きさが `global_gt` の物体で出せる範囲を超えている

| 成分 | データの範囲 / RMS | `regressor @ global_gt` の範囲 / RMS | 比 |
|---|---|---|---|
| tx | ±0.34 / 0.1036 N·m | ±0.10 / 0.0408 N·m | **2.5×** |
| ty | ±0.29 / 0.1002 N·m | ±0.09 / 0.0397 N·m | **2.5×** |
| tz | ±0.30 / 0.0986 N·m | ±0.11 / 0.0330 N·m | **3.0×** |
| fx | ±8.43 / 2.2604 N | ±6.0 / 1.8148 N | 1.25× |
| fz | ±8.55 / 3.3090 N | ±6.8 / 3.0731 N | 1.08× |

トルクが 2.5–3 倍大きいのは、フレーム対応・符号・成分順のどの入れ替えでも説明できません（下記 §3）。
**慣性テンソルまたはモーメントアームが、`global_gt` に書かれている値より大きい物体で生成された**可能性があります。

### 3. 規約の取り違えは棄却済み

| 検査 | loaded_dice の force 残差 | 判定 |
|---|---|---|
| そのまま | 2.2190 N | — |
| torque ブロックの符号反転 | 2.2190 N | 改善なし |
| force ブロックの符号反転 | 2.2190 N | 改善なし |
| 全体の符号反転 | 2.2190 N | 改善なし |
| force / torque ブロック入れ替え | （torque 残差 2.618 N·m に悪化） | 棄却 |
| フレームシフト −25…+25 | shift 0 が最良 | 棄却 |
| 定数オフセット 3 列を追加 | 2.2089 N | 改善なし（重力項では説明できない） |
| 世界座標重力 3 列を追加 | 2.2089 N | 改善なし |

対照として hammer で force/torque を入れ替えると残差が 0.046 → 4.69 と 100 倍悪化するので、
**現行の成分順 `[force(3); torque(3)]` が hammer では正しい**ことが確認できます。学習側の
`dynamics/dynamics.py::_get_wrench` は `cat([forces, torques])` を返し、この順と一致しています。

### 3-2. 長さスケール（`aabb_scale`）の取り違えも棄却済み

`loaded_dice` の `aabb_scale` は 0.0345、hammer は 0.195 で 5.7 倍違うため、正規化座標から物理長への
変換係数の取り違えを疑うのが自然です。これは以下の理由で原因になりえません。

1. **§1–§4 の検査は `aabb_scale` を一切使っていません。** データセット自身の `regressor` (6×10)、
   `wrench` (6)、`global_gt` (10) の 3 者だけで閉じており、学習側の座標生成を通していません。
2. **`aabb_scale` の誤りは慣性パラメータ側に吸収されます。** 物体を相似比 `s` で拡大縮小すると
   `m ∝ s⁰`、`mc ∝ s¹`、`I ∝ s²` としてパラメータが変わるだけで、`regressor` の列（`twist` / `dtwist` と
   基準フレームのみの関数）は変わりません。§1 の最小二乗は 10 パラメータを自由に当てているので、
   **任意の `aabb_scale` 誤りは既に探索空間に含まれています**。それでも force 残差 2.219 N
   （R² = 0.30、torque は R² = 0.05）が残りました。

したがって「単位・スケールの取り違え」では説明が付きません。相似変形は 10 パラメータの取り替えに
すぎず、最小二乗の張る空間の中にあるため、§2 で示した torque の 2.5–3 倍という超過は
相似変形では消せない形の不整合です。

参考として、学習側で `aabb_scale` が実際に使われている箇所（規約確認用）:

| 箇所 | 用途 |
|---|---|
| `wisp/trainers/md_multiview_trainer.py:569` | `get_coords_mp(pidx, scale=aabb_scale, homog=True, zup=True)` → wrench 計算に渡す質量点座標 |
| 同 `:736`, `:1032` | 慣性パラメータ・重心のログ用座標 |
| 同 `:753`, `:1013` | geometric center の座標 |
| 同 `:988-994` | GT とのスコアで `aabb_scale^0/1/2` による次元合わせ |

### 4. フレーム対応が壊れている兆候

`wrench[i]` に最も近い `regressor[j] @ global_gt` を探すと:

| | loaded_dice | hammer（対照） |
|---|---|---|
| 最近傍が自分自身のフレーム | **4 / 300** | **300 / 300** |
| 対角コスト vs 最近傍コスト | 1.731 vs 0.494 | 0.0466 vs 0.0466 |

最適割当（Hungarian）で並べ替えると force 残差が 2.219 → 0.849 N、LS 質量が 0.203 → 0.355（真値 0.331）
まで改善します。ただし **torque 残差は 0.1071 → 0.1002 とほぼ変わらず**、単なる行の並べ替えだけでは
説明が付きません。§2 の大きさの問題と併存していると見ています。

## 学習側に問題が無いことの根拠

`loaded_dice` での dyn loss は epoch 2 以降 `1.37E-02` から動きません。この値は
「`regressor` の 10 次元列空間内で huber 損失を最小化した理論最適値」と一致します:

```
訓練設定: dynamics_loss_type=huber, dynamics_weight=0.1
         force_scale=4.562441, torque_k=26.1766  (訓練 split の std から自動算出)

huber 最適な 10 パラメータフィット   = 0.13666  → ×0.1 = 0.01367
訓練で観測された張り付き値 [dyn]                      = 0.0137     ← 4 桁一致
その残差 RMS: force 2.224 N / torque 0.0983 N·m
訓練中に計測した残差 RMS: force 1.6–2.5 N / torque ~0.09 N·m   ← 一致
```

つまりネットワークは**到達可能な最小値に到達しており、詰まっていません**。床はデータ自身の既約残差です。
同一コード・同一設定で hammer は dyn loss が `2.95E-06` まで落ち、残差も単調に減少します。

なお §1–§4 の検査は `regressor` と `global_gt` を使っていますが、**訓練コードはこの 2 つを読んでいません**
（付録 A 参照）。にもかかわらず上の 4 桁一致が成立するのは、訓練側モデルが出せる wrench の集合と
`regressor` の 10 次元列空間が一致していることの実測的な裏付けになります。つまり
「`regressor` で説明できない = 訓練でも到達できない」が経験的に確認されています。

## お願いしたいこと

1. `loaded_dice` の `wrench` を生成したスクリプトと、そのときの物体パラメータ（質量・重心・慣性テンソル）を
   共有してください。`global_gt` に書かれた値と一致しているか確認したいです。
2. `wrench` と `regressor` / `twist_sen` / `dtwist_sen` が**同一フレーム・同一時刻**から書き出されているか
   確認してください（§4 の 4/300 は行の対応が崩れている兆候です）。
3. 生成時に `wrench` へノイズを注入しているなら、その振幅を教えてください。§1 の R²（force 0.30 / torque 0.05）
   と整合するか確認します。
4. 生成側で `torque = r × F` か `F × r` か、およびどの座標系（センサ / 物体 / 世界）で書き出しているかを
   明記してください。学習側は `r × F`、センサ座標系（`pose_sen_obj` で変換後）を前提にしています。
5. 再生成する場合は、出荷前チェックとして `‖regressor @ global_gt − wrench‖ / ‖wrench‖ < 1e-2`
   を通すことを提案します。hammer は 0.01 を満たし、loaded_dice は force 0.92 / torque 1.12 で落ちます。

## 再現手順

```python
import json, numpy as np
d = json.load(open("datasets/neural-mass-fields/loaded_dice/transforms.json"))
A  = np.array([f["regressor"] for f in d["frames"]])      # (300, 6, 10)
w  = np.array([f["wrench"]    for f in d["frames"]])      # (300, 6)
gt = np.array(d["global_gt"][:10])                        # labels 順

res = np.einsum("fij,j->fi", A, gt) - w
print("force res", np.sqrt((res[:, :3] ** 2).mean()))     # 2.4454  (信号 2.6531)
print("torque res", np.sqrt((res[:, 3:] ** 2).mean()))    # 0.1129  (信号 0.1008)

x = np.linalg.lstsq(A.reshape(-1, 10), w.reshape(-1), rcond=None)[0]
print("LS mass", x[0], "vs gt", gt[0])                    # 0.2030 vs 0.3310
```

---

# 付録 A: JSON エントリが訓練コードでどう受け取られ処理されるか

生成側で規約を突き合わせられるよう、`transforms.json` の各エントリが訓練でどう使われるかを
実際のコードとともに示します。行番号は 2026-08-05 時点の `add-gradient-diagnostics` ブランチのものです。

## A-0. 訓練が読むキーの一覧（重要）

`wisp/datasets/formats/nemd_standard_dataset.py:293-328`（`_load_single_entry`）が読むのは
**フレームごとの次の 9 キーだけ**です。

| JSON のキー | 訓練内部での名前 | 使われ方 |
|---|---|---|
| `file_path` | `img` / `basename` | 画像読み込み。rgb / mask 損失 |
| `transform_matrix` | `pose` | カメラ姿勢。レイ生成 |
| `pose_sen_obj` | `pose_sen_obj` | 質量点座標をセンサ座標系へ変換（wrench 計算） |
| `pose_sen_obji` | `pose_sen_obji` | dict には入るが dyn 損失の経路では未使用 |
| `twist_sen` | `twist_sen` | wrench 計算 |
| `dtwist_sen` | `dtwist_sen` | wrench 計算 |
| `linacc_sen_obji` | `linacc_sen_obji` | dict には入るが dyn 損失の経路では未使用 |
| `aabb_scale` | `aabb_scale` | 質量点座標の物理長スケール |
| **`wrench`** | **`ft_sen`** | **dyn 損失の教師値** |

```python
# nemd_standard_dataset.py:299-301
# The measured wrench is stored under 'wrench' in the current dataset format
# (older datasets used 'ft_sen'); fall back to NaN so plain NeRF datasets still load.
ft_sen = frame.get("ft_sen", frame.get("wrench", np.nan * np.empty(6)))
```

**訓練が読まないキー**: `regressor`、および トップレベルの `global_gt` / `labels` / `ls` / `tls`。
これらを参照するのは `wisp/trainers/tracker/wandb_add_reference.py:95-96`（wandb へ参照値テーブルを
上げる別ユーティリティ）だけで、学習経路には入りません。真値スコアは別ファイル
`ground_truth.csv` から読む設計で、`loaded_dice` にはそれが無いため
`No ground truth data — logging inertial parameters without score.` が出ます。

つまり **`regressor` / `global_gt` は訓練では一切使われず、`wrench` だけが教師値として効きます**。
本報告で両者を使ったのは、データセット単体の自己整合性を訓練抜きで判定するためです。

## A-1. バッチとして trainer に渡るまで

`wisp/datasets/formats/nemd_standard_dataset.py:147-162`（`__getitem__`）:

```python
out = MultiviewBatch(
    rays=self.data["rays"][idx],
    rgb=self.data["rgbs"][idx],
    masks=self.data["masks"][idx],
    # Added for NeMD
    pose_sen_obj=self.data["poses_sen_obj"][idx],
    pose_sen_obji=self.data["poses_sen_obji"][idx],
    linacc_sen_obji=self.data["linaccs_sen_obji"][idx],
    twist_sen=self.data["twists_sen"][idx],
    dtwist_sen=self.data["dtwists_sen"][idx],
    ft_sen=self.data["fts_sen"][idx],
    aabb_scale=self.data["aabb_scales"][idx],
    idx=idx,
    ray_traceables=["rays", "rgb", "masks"],
    ground_truth_path=self.data["ground_truth_path"],
)
```

`DataLoader` は `batch_size=1`, `shuffle=True`（`wisp/trainers/base_trainer.py:198-201`）。
1 ステップ = 1 フレームです。バッチ次元は `step()` 内で `.squeeze(0)` で落とされます。

## A-2. 正規化スケールの決定（訓練開始時に 1 回）

`wisp/trainers/md_multiview_trainer.py:113-181`（`_resolve_dynamics_wrench_scales`）。
**訓練 split 全体の `wrench` の標準偏差**から自動決定します。データを差し替えると値が変わります。

```python
component_std = wrenches.std(dim=0, correction=0)
measured_force_scale  = torch.linalg.vector_norm(component_std[:3]).item()
measured_torque_scale = torch.linalg.vector_norm(component_std[3:]).item()
...
force_scale = measured_force_scale if configured_force_scale is None else float(configured_force_scale)
torque_k    = force_scale / measured_torque_scale if configured_torque_k is None else float(configured_torque_k)
```

実測値: `loaded_dice` は `force_scale=4.562441, torque_k=26.1766`、
`hammer` は `force_scale=11.5196, torque_k=10.1731`。
**`wrench[:3]` を力、`wrench[3:]` をトルクと仮定している**点に注意してください。

## A-3. dyn 損失の計算（毎ステップ）

`wisp/trainers/md_multiview_trainer.py:557-580`:

```python
if self.cfg.dynamics_weight > 0.0 and self.prune_count > 0:
    self.aabb_scale = data["aabb_scale"].to(self.device).squeeze(0)
    gt_wrench  = data["ft_sen"].to(self.device).squeeze(0)      # ← JSON の 'wrench'
    twist_sen  = data["twist_sen"].to(self.device).squeeze(0)
    dtwist_sen = data["dtwist_sen"].to(self.device).squeeze(0)
    pose_sen_aabb = SE3.from_matrix(
        data["pose_sen_obj"].reshape(4, 4).to(self.device),
        normalize=True,
    )

    self._coords_aabb_sampled_mp = self._grid.get_coords_mp(
        pidx=pidx,
        scale=self.aabb_scale,
        homog=True,
        zup=True,
    )
    _coords_sen_sampled_mp = pose_sen_aabb.dot(self._coords_aabb_sampled_mp)

    wrench = get_wrench(
        sampled_mass_set,
        _coords_sen_sampled_mp[..., :3],
        twist_sen,
        dtwist_sen,
    )
```

続けて正規化と損失（同 `:582-596`）:

```python
    # Normalize both wrench blocks with fixed training-split statistics.
    wrench_scale = torch.ones_like(gt_wrench)
    wrench_scale[3:] = self.cfg.dynamics_torque_k
    wrench_scale /= self.cfg.dynamics_force_scale
    wrench    = wrench    * wrench_scale
    gt_wrench = gt_wrench * wrench_scale

    dynamics_loss += compute_loss(wrench, gt_wrench, self.cfg.dynamics_loss_type)
    dynamics_loss *= self.cfg.dynamics_weight
```

`compute_loss` は `wisp/trainers/loss_functions.py:5-16` で、既定 `reduction="mean"`、
`huber` は `F.huber_loss`（δ=1）です。**6 成分の平均**を取ります。

**注意点 1**: `self.prune_count > 0` が条件なので、最初の prune（`prune_every=100` なら 100 イテレーション目）
までは dyn 損失が 0 です。
**注意点 2**: 正規化後の残差が 1 を超えると huber が線形域に入ります。`loaded_dice` は
force 残差 2.224 N / 4.562 = 0.49、torque 0.0983 × 26.18 / 4.562 = 0.56 なので二乗域内です。

## A-4. 質量点座標の作り方（座標系の規約）

`wisp/models/grids/hash_grid_mass_distr.py:246-265`:

```python
def get_coords_mp(self, pidx=None, scale=None, homog=False, zup=False, lexsort=False):
    _coords_mp = self._yup_coords_ndevice_dense_mp
    if zup:
        _coords_mp = self._zup_coords_ndevice_dense_mp
    if pidx is None:
        pidx = self.dense_pidx
    _coords_mp = torch.index_select(_coords_mp, 0, pidx)
    if scale is not None:
        rot = SO3(scale * torch.eye(3, device=self.device))
        pose_scaled_ndevice = SE3(rot, torch.zeros(3, device=self.device))
        _coords_mp = pose_scaled_ndevice.dot(_coords_mp)
    ...
```

`scale` は 3 軸等方の相似スケール（`scale * I₃`）です。訓練は `zup=True` を渡します。
y-up → z-up の回転は同ファイル `:87-95` の `roll = -3π/2` で、実際の軸対応は:

```
R_zup_yup = [[1, 0,  0],
             [0, 0, -1],
             [0, 1,  0]]

  yup +x -> zup +x
  yup +y -> zup +z
  yup +z -> zup -y
```

**確認いただきたいのは、`wrench` / `twist_sen` / `dtwist_sen` がこの z-up 側の軸並びで
書き出されているか**です。`pose_sen_obj` はこの z-up・`aabb_scale` 適用後の座標に左から掛かります。

## A-5. wrench の計算式（`dynamics/dynamics.py`）

```python
# :164-174
def get_wrench(mass_set, mass_point_coords, twist, dtwist):
    w_twist  = SE3.wedge(twist)
    w_dtwist = SE3.wedge(dtwist)
    return _get_wrench(mass_set, mass_point_coords, w_twist, w_dtwist)

# :143-161
def _get_wrench(mass_set, mass_point_coords, w_twist, w_dtwist):
    linaccs = get_linear_accelerations(w_twist, w_dtwist, mass_point_coords)
    forces  = mass_set * linaccs
    torques = tla.cross(mass_point_coords.float(), forces.float(), dim=-1)   # ~ (cells, 3)
    wrench  = torch.cat([forces, torques], dim=-1).sum(dim=0)                # -> (6)
    return wrench
```

```python
# :109-138
def get_linear_accelerations(w_twist, w_dtwist, coords, homogeneous=False):
    _linvel = get_linear_velocity(w_twist, coords, homogeneous=True)
    _linacc = homogenize(coords) @ w_dtwist.T + _linvel @ w_twist.T
    return _linacc if homogeneous else _linacc[..., :3]

# :82-106
def get_linear_velocity(w_twist, coords, homogeneous=False):
    _linvel = homogenize(coords) @ w_twist.T   # batched version of w_twist @ _pos
    return _linvel if homogeneous else _linvel[:3]
```

ここから読み取れる訓練側の規約を明示します。**生成側と一致しているか確認してください。**

| 項目 | 訓練側の規約 |
|---|---|
| wrench の成分順 | **`[force(3); torque(3)]`**（`cat([forces, torques])`） |
| トルクの外積順 | **`τ = r × F`**（`cross(coords, forces)`） |
| twist / dtwist の成分順 | **`[線形(3); 角(3)]`**。`SE3.wedge` が `Xi[:3,:3] = SO3.wedge(xi[3:])`、`Xi[:3,3] = xi[:3]` とするため（liegroups） |
| 加速度の式 | `a_i = Ξ̇ r̃_i + Ξ ṽ_i`（= `a₀ + α×r_i + ω×(ω×r_i + v₀)`）。Modern Robotics 8.2.1 |
| 基準点 | `pose_sen_obj` 適用後の**センサ座標系原点**（`r_i` はセンサ原点からのベクトル） |
| **重力** | **式に重力項が無い**。したがって `dtwist_sen[:3]` は重力を含んだ固有加速度（proper acceleration）である必要があります |
| 単位 | 座標 m、力 N、トルク N·m を前提（`aabb_scale` が m 単位を与える） |

最後の重力の扱いが最も食い違いやすい箇所だと考えています。生成側が `dtwist_sen` を
純運動学加速度（重力抜き）として書き出し、`wrench` を力覚センサ相当（重力込み）で書き出していると、
一定の不整合が入ります。ただし §3 で定数オフセット 3 列と世界座標重力 3 列を追加しても残差が
2.2190 → 2.2089 しか改善しなかったので、**`loaded_dice` の不整合は重力項だけでは説明できません**。

同じコードを `hammer_spline_20260731_113007_run1/unperturbed_transforms.json` に当てると
force res 0.065 / LS mass 1.1163（真値 1.1161）になります。
