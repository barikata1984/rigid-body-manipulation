# `hammer_spline_20260731_113007_run1` データセット互換性調査と 100 epoch 検証

作成日: 2026-07-31
対象: `datasets/realworld_nemd/hammer_spline_20260731_113007_run1`
比較基準: `datasets/realworld_nemd/hammer-07-27-#2`（動作実績あり）
ブランチ: `add-geom-center-and-com-plot`

## 概要

新データセットは初期状態では起動不能だった。原因はルート直下の JSON 個数と `file_path` の絶対パスの 2 点で、いずれもデータセット側の問題である。加えて背景色規約の不一致があり、これは実行時フラグで回避できる。

これら 3 点に対処したうえで 100 epoch を実走させたところ完走し（exit 0、7 分 40 秒）、推定慣性パラメータはデータセットに埋め込まれた真値と 1–5% で一致した。学習パスが要求するデータエントリに過不足はない。

---

## 1. 修正項目

### 項目 1: ルート直下の `*.json` が 8 個ある（ハードエラー）

**要求元** — `wisp/datasets/formats/nemd_standard_dataset.py:238-247`

```python
transforms = sorted(glob.glob(os.path.join(dataset_path, "*.json")))
...
elif len(transforms) > 3 or len(transforms) == 2:
    raise RuntimeError(f"NeRF dataset folder has an unsupported number of splits, ...")
```

ルート直下の `.json` の**個数**で分岐する。ファイル名では絞り込まない。1 個か 3 個でなければ例外。

**期待される形** — ルート直下の `.json` は 1 個（train 単独）または厳密に 3 個。3 個の場合、`nemd_standard_dataset.py:255-258` が `for _split in ["test","train","val"]: if _split in fname` で部分一致マッピングするため、`transforms_train.json` / `transforms_valid.json` / `transforms_test.json` で成立する（`"val" in "transforms_valid.json"` は True）。

基準データセット `hammer-07-27-#2` の実際の形: ルート直下の `.json` は `transforms_train.json` の 1 個のみ。他は `.bak` 拡張子で glob から外してある。

**新データセットの現状** — 8 個。`transforms{,_train,_valid,_test}.json` と `unperturbed_transforms{,_train,_valid,_test}.json`。

**実際のエラー**

```
RuntimeError: NeRF dataset folder has an unsupported number of splits, there should be
['test', 'train', 'val'], but found: [... 8 files ...]
```

**対処** — `unperturbed_transforms.json` 以外の 7 個を `.json.bak` にリネームした（§2 参照）。生成側で対応する場合は、ルートに 3 分割 JSON のみを出力する。リポジトリ内に該当する変換スクリプトは存在しない。

### 項目 2: `file_path` が他マシンの絶対パス（ハードエラー、ただし無言で壊れる）

**要求元** — `wisp/datasets/formats/nemd_standard_dataset.py:277`

```python
fpath = os.path.join(root, frame["file_path"].replace("\\", "/"))
```

`os.path.join` は第 2 引数が絶対パスだと `root` を捨てるため、外部マシンのパスがそのまま残る。同 286 行の `os.path.exists(fpath)` が False のとき 330-332 行で `return None` となり、**エラーも警告も出さずにフレームが捨てられる**（331 行の `log.info` はコメントアウト済み）。

**期待される形** — データセットルートからの相対パス。基準データセットの実値は `"file_path": "complete/0000.png"`。拡張子は省略可（280-282 行で `.png` を補完）。

**新データセットの現状** — `transforms*.json` / `unperturbed_transforms*.json` の全ファイル・全フレームが絶対パス。実値:

```
"/home/ak/workspace/rigid-body-manipulation/datasets/hammer/spline_20260731_113007_run1/complete/0000.png"
```

**実際のエラー**

```
File "wisp/datasets/formats/nemd_standard_dataset.py", line 494, in _collect_data_entries
  imgs = torch.stack(imgs)
RuntimeError: stack expects a non-empty TensorList
```

ローダは JSON パースとフレーム数 300 の認識までは到達する。tqdm が `300/300` を 450516 it/s で駆け抜けるのが全件スキップの証拠（実デコードなら約 165 it/s）。

**対処** — `complete/XXXX.png` 形式に書き換えた（§2 参照）。**生成側への依頼はこの項目に絞れる。** あわせて、生成側でパス実在チェックを入れることを推奨する。300 枚中 1 枚だけ誤っていても無言で学習から消えるため。

### 項目 3: alpha=0 画素の RGB が黒（エラーにならないが学習が破綻）

**要求元** — `wisp/datasets/formats/nemd_standard_dataset.py:608-620`

```python
has_external_masks = any(m is not None for m in external_masks)
if has_external_masks:
    masks = torch.stack(external_masks).unsqueeze(-1)   # RGB はそのまま
else:
    alpha = imgs[..., 3:4]
    masks = (alpha > 0.5).bool()
    rgbs = rgbs[..., :3] * alpha + (1 - alpha) * np.array(self.bg_color)  # 合成が走る
```

tracer 側は `app/nemd/configs/nemd_hash_fused.yaml:47` で `bg_color: [1.0, 1.0, 1.0]`（白）。

**基準データセット** — `complete/*.png` が RGB 3ch（alpha なし）で `masks/` が実在するため外部マスク分岐に入り、合成は走らない。GT 背景は白（画像平均 RGB ≈ 252）で tracer と一致。

**新データセットの現状** — `complete/*.png` が RGBA 4ch（800×800）、alpha はバイナリ（0/255）、**alpha=0 画素の RGB は実測 (0,0,0) の黒**。`masks/` が無いため alpha 分岐に入り、`dataset.bg_color` の既定値 `[0,0,0]` と合成されて GT 背景が黒になる。tracer は白を描くため、背景画素（全体の 98.3%）で誤差 1.0 が張り付く。

**実測**（いずれも 2 epoch、seed 7301、同一設定）

| データセット | rgb loss (ep1) | rgb loss (ep2) |
|---|---|---|
| `hammer-07-27-#2`（基準） | 1.13E-03 | 7.89E-04 |
| 新（項目 1・2 のみ修正） | 4.87E-01 | 4.89E-01 |
| 新 + `--dataset.bg-color 1.0 1.0 1.0` | 7.52E-04 | 2.28E-04 |

4.9E-01 は huber（delta=1）で誤差 1.0 のときの 0.5 とほぼ一致し、背景全面ミスマッチと整合する。

**対処** — **データセット変更は不要。** 実行時に `--dataset.bg-color 1.0 1.0 1.0` を付ける。`--bg-color` と短縮すると `tracer.bg-color` と衝突して `AmbiguousArgument` になるため `--dataset.` 前置が必須。生成側で揃えたい場合は alpha=0 画素の RGB を白にする案もあるが、実行時フラグの方が非破壊的である。

---

## 2. 適用した変更と復元手順

`datasets/` は `.gitignore:16` により git 管理外（`git ls-files datasets/` は 0 件）。リポジトリのトラック対象ファイルは一切変更していない。

### 2-1. JSON のリネーム（7 件）

`datasets/realworld_nemd/hammer_spline_20260731_113007_run1/` にて実行:

```sh
mv transforms.json                     transforms.json.bak
mv transforms_train.json               transforms_train.json.bak
mv transforms_valid.json               transforms_valid.json.bak
mv transforms_test.json                transforms_test.json.bak
mv unperturbed_transforms_train.json   unperturbed_transforms_train.json.bak
mv unperturbed_transforms_valid.json   unperturbed_transforms_valid.json.bak
mv unperturbed_transforms_test.json    unperturbed_transforms_test.json.bak
```

復元（同ディレクトリで実行）:

```sh
mv transforms.json.bak                     transforms.json
mv transforms_train.json.bak               transforms_train.json
mv transforms_valid.json.bak               transforms_valid.json
mv transforms_test.json.bak                transforms_test.json
mv unperturbed_transforms_train.json.bak   unperturbed_transforms_train.json
mv unperturbed_transforms_valid.json.bak   unperturbed_transforms_valid.json
mv unperturbed_transforms_test.json.bak    unperturbed_transforms_test.json
```

### 2-2. `file_path` の相対化

原本を `.abs` 拡張子でバックアップしたうえで書き換えた。**ルート直下に新たな `*.json` を作らないこと**が条件（ローダの glob 個数判定が壊れるため）。

```sh
cp -p unperturbed_transforms.json unperturbed_transforms.json.abs
```

変換スクリプト: `<scratchpad>/relativize.py`。`.abs` を読んで `frame["file_path"]` のみ `"complete/" + basename` に置換し `json.dump` で書き戻す。出力 `frames=300 rewritten=300`。

復元:

```sh
mv unperturbed_transforms.json.abs unperturbed_transforms.json
```

### 2-3. 変換後の事前検査

| 検査 | 結果 |
|---|---|
| 総フレーム数 | 300 |
| `os.path.isfile(os.path.join(root, file_path))` が True | **300 / 300** |
| 同 False | 0 |
| 全件が相対パス | True |
| 参照ディレクトリ | `complete` のみ |
| `file_path[0]` / `[-1]` | `complete/0000.png` / `complete/0299.png` |
| `file_path` 以外のペイロードが元と完全一致 | True |
| トップレベル / 各フレームのキー順序保存 | True |
| フレーム順序の保存 | True |
| ローダ glob が見る `*.json` | `['unperturbed_transforms.json']` の 1 個のみ |

---

## 3. データエントリの過不足

**結論: 学習パスが要求するキーに過不足はない。**

比較対象は新 `unperturbed_transforms.json`（300 frames）と旧 `hammer-07-27-#2/transforms_train.json`（300 frames）。

### 3-1. per-frame キーの差分

| 区分 | キー | 影響 |
|---|---|---|
| 共通・shape/型すべて一致 | `file_path`, `transform_matrix` (4,4), `pose_sen_obj` (4,4), `twist_sen` (6,), `dtwist_sen` (6,), `wrench` (6,), `regressor` (6,10) | — |
| 新のみ | `jointvars_clean` (list[3,6] float64, 300/300) | 参照ゼロ |
| 旧のみ | `cx`, `cy`, `fl_x`, `fl_y`, `w`, `h`, `k1`, `k2`, `p1`, `p2` | 旧でも実質未使用。`_load_single_entry` は読まず、`_collect_data_entries` はトップレベルのみ参照 |

### 3-2. トップレベルキーの差分

| 区分 | キー |
|---|---|
| 共通・型一致 | `aabb_scale`, `camera_angle_x`, `camera_angle_y`, `cx`, `cy`, `fl_x`, `fl_y`, `h`, `w` |
| 新のみ | `date_time`, `global_gt` (list[11]), `ls` (list[11]), `tls` (list[11]) |
| 旧のみ | `camera_model`, `excitation_identification_manifest`, `k1`, `k2`, `p1`, `p2`, `ols_bias` |

**`labels` は要素数と順序の両方が異なる。**

- 新（11 要素）: `[total_mass, mx, my, mz, ixx, iyy, izz, ixy, iyz, izx, aabb_scale]`
- 旧（10 要素）: `[total_mass, mx, my, mz, ixx, ixy, ixz, iyy, iyz, izz]`

`labels` / `global_gt` の使用箇所は `wisp/trainers/tracker/wandb_add_reference.py:95-96` のみで、これは `if __name__ == "__main__":` の独立スクリプト。学習ループからは呼ばれない。なお同ファイル 71 行は `transform_train.json`（アンダースコアの位置が実ファイル名と異なる）を開こうとするため、そもそも現状動かない。

### 3-3. ローダが実際に読むキーと充足状況

per-frame（`_load_single_entry`, `nemd_standard_dataset.py:265-332`）:

| 行 | キー | 新データセット | 欠損時の挙動 |
|---|---|---|---|
| 277 | `file_path` | あり | `KeyError`。存在しても実在しないパスなら 330-332 行で無言スキップ |
| 321 | `transform_matrix` | あり | `KeyError` |
| 293 | `pose_sen_obj` | あり | NaN 埋め |
| 294 | `pose_sen_obji` | **無し** | NaN 埋め（**旧も無し。既存挙動どおり**） |
| 295 | `twist_sen` | あり | NaN 埋め |
| 296 | `dtwist_sen` | あり | NaN 埋め |
| 297 | `linacc_sen_obji` | **無し** | NaN 埋め（**旧も無し**） |
| 298 | `aabb_scale` | per-frame は無し | トップレベルにフォールバック（202-212 行）。それも無ければ既定 1.25 |
| 301 | `ft_sen` → `wrench` | `wrench` あり | 2 段フォールバック後 NaN 埋め |
| 303-315 | `masks/<basename>.png` | **無し** | `None` → 608-620 行でアルファチャネル分岐（項目 3 の原因） |

トップレベル（`_collect_data_entries`, 494-640 行）:

| 行 | キー | 新 | 挙動 |
|---|---|---|---|
| 507/511 | `x_fov` / `y_fov` | 無し | 次の分岐へ |
| 516 | `fl_x` | あり | `elif "fl_x" in metadata and False:` の死んだ分岐。決して使われない |
| 522/527 | `camera_angle_x` / `camera_angle_y` | あり | fx=fy=965.69 を算出。`fl_x`/`fl_y` の値と一致するため実害なし |
| 559/562 | `cx` / `cy` | あり (400.0/400.0) | `x0 = 400 - 800//2 = 0`, `y0 = 0` |
| 564/565 | `offset` / `scale` | 無し | 既定 `[0,0,0]` / `1.0` |
| 543 | `k1` | 無し | 歪み補正の警告ログを出さないだけ（補正は未実装） |
| 622-625 | `ground_truth.csv` | 無し | `md_multiview_trainer.py:343-361` の `try/except FileNotFoundError` で継続（**旧にも無い**） |

### 3-4. split の扱い

リネーム後の実データセットに対し `_validate_and_find_transform` を直接呼んだ結果:

```
split='train' -> .../unperturbed_transforms.json
split='val'   -> None
split='test'  -> None
```

`nemd_standard_dataset.py:249-250` の `if len(transforms) == 1: transform_dict["train"] = transforms[0]` により、ファイル名に関係なくルートの JSON 1 個が無条件で train として採用される。300 フレーム全件が train。

valid/test が無くても trainer は回る。`main_nemd.py:136-139` の `if cfg.trainer.valid_every > -1 or cfg.trainer.mode == "validate":` が偽（`valid_every: -1`、`mode: 'train'`）のため `create_split` が呼ばれず `validation_dataset = None` になる。基準データセット `hammer-07-27-#2` もルート JSON は 1 個のみで、同じ構成である。

### 3-5. 誤解しやすい欠落（いずれも問題なし）

| 項目 | 判定 |
|---|---|
| `masks/` 欠落 | alpha から生成される（`nemd_standard_dataset.py:303-315` は無ければ `None` を返す設計） |
| `meta.yaml` 欠落 | Python コードから参照ゼロ（`grep -rn "meta\.yaml" --include="*.py" app/ wisp/` → 0 件） |
| `ground_truth.csv` 欠落 | `try/except FileNotFoundError` 済み。旧にも存在しない |
| カメラ別ディレクトリ (`d455_*`) 欠落 | コードから参照ゼロ |
| `train/` `valid/` `test/` ディレクトリ | JSON から参照されず、新旧とも実質未使用 |

---

## 4. 100 epoch 実走の結果

### 4-1. 実行コマンド

```sh
CUDA_VISIBLE_DEVICES=0 WISP_HEADLESS=1 python app/nemd/main_nemd.py \
  --config app/nemd/configs/nemd_hash_fused.yaml \
  --interactive False \
  --dataset-path 'datasets/realworld_nemd/hammer_spline_20260731_113007_run1' \
  --dataset.bg-color 1.0 1.0 1.0 \
  --seed 7301 --trainer.max-epochs 100 --trainer.prune-every 100 \
  --trainer.rgb-weight 1.0 --trainer.mask-weight 1.0 \
  --trainer.dynamics-weight 0.1 --trainer.tv-weight 1.0 --trainer.kd-weight 1.0 \
  --tracker.enable-wandb True --tracker.plot-mass-distr True \
  --tracker.mass-distr-overlay-opacity 0.9
```

### 4-2. 完走状況

| 項目 | 値 |
|---|---|
| exit code | **0** |
| 総所要時間 | 06:43:54 → 06:51:34 = **7 分 40 秒** |
| 内訳 | 起動+データロード 18s / 学習 7分7秒 / 後処理 14s |
| 1 epoch あたり | **4.044 s**（wandb `time/elapsed_ms_per_epoch = 4043.70`） |
| 400 epoch の外挿 | 約 **28 分** |
| wandb run | [`trim-meadow-254`](https://wandb.ai/barikata1984/nemd/runs/hpp8ovqf) |
| 出力ディレクトリ | `_results/nemd/test/20260731-064406/` |

トレースバックなし。`Epoch N/100` 行はちょうど 100 行。

### 4-3. 損失の推移

```
Epoch   1/100 | [ttl] 1.20E-02, [rgb] 5.63E-04, [dyn] 2.42E-04, [tv] 2.08E-05, [kd] 3.10E-05, [mask] 1.12E-02 | mps 19770
Epoch   2/100 | [ttl] 1.22E-03, [rgb] 1.66E-04, [dyn] 4.42E-05, [tv] 2.88E-05, [kd] 3.99E-05, [mask] 9.42E-04 | mps 21640
Epoch   5/100 | [ttl] 6.16E-04, [rgb] 1.18E-04, [dyn] 8.76E-06, [tv] 2.27E-05, [kd] 4.94E-05, [mask] 4.17E-04 | mps 26685
Epoch  50/100 | [ttl] 2.65E-04, [rgb] 3.95E-05, [dyn] 8.91E-06, [tv] 2.82E-05, [kd] 3.51E-05, [mask] 1.53E-04 | mps 15666
Epoch 100/100 | [ttl] 1.91E-04, [rgb] 1.45E-05, [dyn] 5.00E-06, [tv] 2.79E-05, [kd] 3.50E-05, [mask] 1.09E-04 | mps 15468
```

全 100 epoch を機械的にパースした集計:

| 項 | min | max | final | NaN | Inf | 負値 |
|---|---|---|---|---|---|---|
| ttl | 1.870E-04 | 1.200E-02 | 1.910E-04 | 0 | 0 | 0 |
| rgb | 1.440E-05 | 5.630E-04 | 1.450E-05 | 0 | 0 | 0 |
| dyn | 4.950E-06 | 2.420E-04 | 5.000E-06 | 0 | 0 | 0 |
| tv | 2.080E-05 | 3.000E-05 | 2.790E-05 | 0 | 0 | 0 |
| kd | 3.100E-05 | 4.990E-05 | 3.500E-05 | 0 | 0 | 0 |
| mask | 1.040E-04 | 1.120E-02 | 1.090E-04 | 0 | 0 | 0 |

ログ全文への `grep -ic nan` も 0 件。`pose_sen_obji` / `linacc_sen_obji` が NaN 埋めになるが、dynamics loss の計算経路はこれらを参照しないため NaN は伝播していない（2.42E-04 → 5.00E-06 と単調低下し、ゼロに張り付いてもいない）。

### 4-4. prune の挙動

**`prune_every` は epoch ではなく iteration 単位。** `md_multiview_trainer.py:44-45` のドキュメント文字列が `every "prune_every" iterations`、実装は同 243 行 `do_prune &= 0 == self.total_iterations % self.cfg.prune_every`。本データセットは 300 frames = 300 iterations/epoch なので、`prune-every 100` は **1 epoch あたり 3 回**発火する。設定の `start_prune: 1000` により開始はイテレーション 1000 ≈ epoch 4。

`Num. active mps` の推移:

| epoch | 1 | 4 (peak) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| active mps | 19770 | **27274** | 18886 | 16612 | 16183 | 16019 | 15666 | 15519 | 15533 | 15495 | 15505 | 15468 |

epoch 4 のピークが `start_prune: 1000` の到達点と一致し、以降は単調減少して epoch 60 以降 15,500 前後で安定。prune 後の損失スパイク・mps のゼロ落ち・振動はない（epoch 60-100 の ttl は 1.87E-04〜1.91E-04 で平坦）。

なお旧データセットの `transforms_train.json`（240 frames）を使う構成に戻すと、1 epoch あたりの発火回数が変わる点に注意。

### 4-5. mass distribution プロット

設定ダンプで `plot_mass_distr: True`, `mass_distr_overlay_opacity: 0.9`, `plot_center_markers: True`, `center_marker_scale: 0.08`, `num_mayavi_camera_turn: 2` の反映を確認。生成物:

| ファイル | 内容 |
|---|---|
| `_results/nemd/test/20260731-064406/mass_distr/0001.png`〜`0100.png` | mayavi 回転アニメ 100 フレーム |
| `mass_distr/mass_distr_20260731-064406.mp4` / `.gif` | 上記の動画化 |
| `mass_distr_centers.png` | geometric center / CoM マーカー重畳図 |
| `mass_distr.gif`, `rgb.gif` | 360 度シーン |

`mass_distr_centers.png` を目視確認: ハンマー形状の質量分布に対し、赤球（center of mass）がヘッド部、青立方体（geometric center）が柄の中ほどに描画され、凡例と Run name も出ている。overlay opacity 0.9 により質量場が半透明で重なる。数値でも `com_z = 0.14935` に対し `geom_z = 0.05139` と明確に分離し、マーカー位置と整合する。

mayavi / プロット系のエラー・警告はなし。360 度 RGB レンダリングも白背景にハンマーが浮遊物なしでシャープに再構成されており、背景色修正が効いていることの視覚的裏付けとなる。

### 4-6. GPU メモリ

5 秒間隔で 91 サンプル取得。

| 指標 | 値 |
|---|---|
| ピーク | **14884 MiB**（t≈5s、起動直後のデータロード/レイ生成時） |
| 学習中の定常値 | 約 9500-9620 MiB |
| 総容量 | 32607 MiB (RTX 5090) |
| ピーク時の残余 | 17723 MiB (54.4%) |
| GPU util | 定常 59-62%、平均 48.5% |

ピークは学習ループではなく起動時の一過性（`nemd_standard_dataset.py:595-604` が 300 view × 800×800 のレイを CUDA 上で生成してから CPU へ移す）。メモリ使用量は epoch 数に依存せず epoch 1 で確定するため、**400 epoch でも同じ 14.9 GB がピーク**。32 GB に対して 2 倍以上の余裕がある。

### 4-7. wandb

ログされたスカラー metric（22 個、すべて有限値）:

`loss/{total,rgb,dynamics,tv,kd,mask}_loss`, `mass/total_mass`, `first_moments/{mx,my,mz}`, `geometric_center/{geom_x,geom_y,geom_z}`, `moments_of_inertia/{ixx,iyy,izz,ixy,iyz,izx}`, `metrics/num_active_mes`, `time/elapsed_ms_per_epoch`, `360-Degree-Scene/step`

メディア/オブジェクト（9 個）: `mass_distr` (video), `mass_distr/centers` (image), `360-Degree-Scene/{RGB, Mass-Distr}` (image), 同 `/MaxLOD` (video), `occupancy` (histogram), `result` (table, 27 列)

同期実績: 5 W&B files, 45 media files, 2 artifact files。

**欠損**: `result` テーブルの `score` と `md_mse` が `None`。`ground_truth.csv` が無いためで、`md_multiview_trainer.py:686-687` が明示的に `None` を設定している。

### 4-8. WARNING / ERROR

`ERROR` / `CRITICAL` / `Traceback` は 0 件。WARNING は wandb SDK 由来の 4 件のみで、いずれも将来非互換の予告:

```
wandb: WARNING Step cannot be set when using tensorboard syncing. ...
wandb: WARNING `format` argument was not provided, defaulting to `gif`. ...  (×3)
```

UserWarning レベルでは `torch.meshgrid` の indexing 引数、`torch.cuda.amp` の deprecation、非タプル多次元インデックス（`hash_grid_mass_distr.py:293`, `md_multiview_trainer.py:584`, `dynamics/dynamics.py:45,52`）が出るが、いずれも旧データセットの実行でも同様に出る既存のもの。

### 4-9. 推定慣性パラメータ

新データセットのトップレベル `global_gt`（真値 11 要素）と突き合わせた。

| パラメータ | 推定値 | `global_gt`（真値） | 相対誤差 |
|---|---|---|---|
| total_mass | 1.12739 | 1.11610 | **+1.01%** |
| mx | 2.700e-07 | 5.463e-09 | 両者ほぼ 0 |
| my | 5.714e-05 | -5.372e-09 | 両者ほぼ 0 |
| mz | 0.168381 | 0.170299 | **-1.13%** |
| ixx | 0.0323809 | 0.0329496 | **-1.73%** |
| iyy | 0.0317222 | 0.0322190 | **-1.54%** |
| izz | 0.000952963 | 0.00100080 | **-4.78%** |
| ixy | -2.128e-06 | -4.737e-12 | 両者ほぼ 0 |
| iyz | -9.010e-06 | 8.796e-10 | 両者ほぼ 0 |
| izx | 1.403e-06 | -1.088e-09 | 両者ほぼ 0 |

物理的妥当性:

| 検査 | 結果 |
|---|---|
| 質量が正 | True (1.1274 kg) |
| 慣性テンソルが正定値 | True（固有値 = [9.530e-04, 3.172e-02, 3.238e-02]、すべて正） |
| 三角不等式 (Iᵢ+Iⱼ ≥ Iₖ) | True。izz が他 2 軸より 34 倍小さく、柄の長いハンマーと整合 |
| 非対角成分 | すべて \|·\| < 1e-5 で対角の 1/3000 以下。主軸がほぼ座標軸に一致 |

重心 `com` = (2.395e-07, 5.068e-05, 0.14935)、幾何中心 `geom` = (1.261e-05, 1.137e-04, 0.05139)。CoM が幾何中心より z 方向に大きくずれており、質量がヘッド側に偏るハンマーとして妥当。`geom_tau_sweep` も 4 段階（factor 0.5/1/2/4）すべて記録され、`geom_z` は 0.05136〜0.05170 と閾値に対して安定。

---

## 5. 発見したバグ（未修正）

`wisp/trainers/tracker/nemd_tracker.py` の 2 箇所で、同一の `moms_i` ベクトルに対する慣性テンソル非対角成分のラベル順序が食い違っている。

- `nemd_tracker.py:515`（`log_dashboards`、wandb スカラー用）: `names = ["ixx", "iyy", "izz", "ixy", "izx", "iyz"]`
- `nemd_tracker.py:562`（`log_result`、result テーブル用）: `names = ["ixx", "iyy", "izz", "ixy", "iyz", "izx"]`

末尾 2 要素が逆。`wisp/trainers/md_multiview_trainer.py:678-679` が `"iyz": self.moms_i[4]`, `"izx": self.moms_i[5]` と定義しているので、**562 行が正しく 515 行が誤り**。実測でも同一 run 内で値が入れ替わっている:

| キー | wandb スカラー (515 行経由) | result テーブル (562 行経由) |
|---|---|---|
| `iyz` | -9.010e-06 | 1.403e-06 |
| `izx` | 1.403e-06 | -9.010e-06 |

今回は両値とも 1e-5 以下で結論に影響しないが、非対角成分が有意になる物体では解釈を誤る。新データセットの `labels` は `[..., ixy, iyz, izx, ...]` で 562 行と同じ規約なので、515 行を合わせるのが筋。

---

## 6. 残課題

- **データセット生成側への依頼**: `file_path` をルート相対 `complete/XXXX.png` 形式に変更する。あわせて生成時のパス実在チェックを推奨（絶対パスは無言で壊れるため）
- **`nemd_tracker.py:515` のラベル順序修正**（§5）
- **`score` / `md_mse` の自動採点**: 真値は `global_gt` として JSON 内にあるが、スコアリング経路は `ground_truth.csv` しか見ない。CSV 化するか読み込み経路を追加すれば自動採点できる
- **`unperturbed_transforms*.json` の用途**: コードから参照ゼロ。学習にどちらを使う想定かを生成側に確認する必要がある
- **`ls` / `tls` キーの用途**: コード上参照ゼロで、用途はコードからは確定できない
- **`--dataset.bg-color 1.0 1.0 1.0` の常用**: 400 epoch の本走でも必須。付けないと rgb loss が 0.49 で張り付く

## 7. 新旧データセットのスケール差

400 epoch 本走や旧データセットとの比較時の注意点。

| 項目 | 新 (`spline_run1`) | 旧 (`hammer-07-27-#2`) | 比 |
|---|---|---|---|
| 学習フレーム数 | 300（単一 JSON 全件） | 300 | 1.00 |
| 解像度 | 800×800 | 640×480 | 画素数 2.08 倍 |
| `\|\|f\|\|` 平均 | 11.46 N | 6.76 N | 1.7 倍 |
| `force_scale` | 11.52 | 5.49 | 2.1 倍 |
| `dtwist` 絶対平均 | 13.56 | 3.30 | 4.1 倍 |
| `twist` 絶対平均 | 1.018 | 0.327 | 3.1 倍 |
| マスク占有率 | 2.07% | 1.26% | 1.6 倍 |
| `aabb_scale` | 0.195 | 0.19 | — |
| 正規化後カメラ距離 | 4.0 | 2.58 | — |

spline 励起の方が明らかに激しい軌道であり、`dynamics_weight 0.1` の効き方は旧データセットと同一にはならない。

---

## 参照

- 実行ログ: `<scratchpad>/e100.log`（100 epoch 全文）
- GPU 計測: `<scratchpad>/gpu_samples.csv`（91 サンプル）
- 相対化スクリプト: `<scratchpad>/relativize.py`
- 学習成果物: `_results/nemd/test/20260731-064406/`
- wandb: https://wandb.ai/barikata1984/nemd/runs/hpp8ovqf

`<scratchpad>` = `/tmp/claude-1000/-workspace/8f60908f-1136-4dce-8b51-1b0190b726c2/scratchpad`（セッション固有のため揮発する）
