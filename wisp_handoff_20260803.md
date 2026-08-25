# wisp/nemd 側セッションへの引き継ぎメモ

作成日: 2026-08-03
作成元: rigid-body-manipulation 側のセッション
前提: `hammer_spline_run1_dataset_report.md` (2026-07-31 の互換性調査レポート) への対応結果

## 1. データセット生成側の変更 (実装・検証済み)

rigid-body-manipulation 側で以下を変更した。今後生成されるデータセットはすべてこの形式になる。

| 変更 | 内容 |
|---|---|
| `file_path` の相対化 | 各 frame の `file_path` は `complete/0000.png` 形式 (データセットルート相対)。絶対パスは廃止 |
| ルート直下の JSON は 1 個 | `transforms.json` (ノイズあり、300 フレーム全件) のみ `.json`。他 7 種 (`transforms_train` 等、`unperturbed_*`) は `.json.bak` 拡張子で書き出す。ローダの glob には 1 個だけヒットし、全フレームが train になる |
| `masks/` を追加 | `masks/<basename>.png`。単一チャンネル uint8、値は {0, 255} の 2 値、解像度は画像と同一。MuJoCo の segmentation レンダリング由来 |
| 背景を白に | skybox を白に変更。背景画素の RGB は白 (アンチエイリアス縁 1 px を除き 255)。`complete/*.png` は RGBA 4ch のままで、alpha は segmentation マスクと同一 |

### wisp 側の運用への影響

- `--dataset.bg-color 1.0 1.0 1.0` は不要になった。そもそも `masks/` が存在すると `nemd_standard_dataset.py:606-620` の背景合成コードに到達しないため、このフラグは no-op になる
- 検証済みサンプル: `datasets/hammer/spline_20260803_173638_run2/` (300 frames。alpha とマスクの一致率 100%、前景率 0.65〜3.53%)
- 注意: hammer はヘッド素材が白いため、白背景とほぼ無コントラストのフレームがある (先頭フレームで前景の 67% が輝度 250 以上)。分離はマスクで行うこと

### 追加の変更 (実装・検証済み)

- `labels` から `aabb_scale` を除去した。`labels` / `global_gt` / `ls` / `tls` はすべて 10 要素 (`aabb_scale` はトップレベルキーとして存続)
- `ls` / `tls` (LS/TLS 同定結果) を `global_gt` と同じ物体 aabb 座標系で書き出すようにした。従来はセンサ座標系のままで、z 軸 180° 回転分 `mx, my, iyz, izx` の符号が逆だった (chair のノイズなし L2 が 0.065 → 0.0035 に改善。hammer は対称なので不変)

**新旧データセットの判別**: JSON に座標系を示すキーはない。`labels` が 11 要素なら旧形式 (`ls`/`tls` はセンサ座標系・末尾に `aabb_scale`/NaN)、10 要素なら新形式 (物体 aabb 座標系)。

## 2. wisp 側で直すべきバグ (実行検証済み)

### 2-1. 慣性テンソル非対角成分のラベル取り違え

**7/31 レポート §5 の正誤判定は逆だった。** wisp 環境で `get_moments_of_inertia` を実際に実行し、単位基底 (1 成分だけ立てた慣性テンソル) で列対応を確認した結果:

- 実装 (`dynamics/dynamics.py:37-60`) が返す並びは `[ixx, iyy, izz, ixy, izx, iyz]` (インデックス定義が `(xy, zx, yz)` の順)
- したがって:

| 場所 | 名前列 | 判定 |
|---|---|---|
| `nemd_tracker.py:515` (`log_dashboards`) | `..., ixy, izx, iyz` | **正しい** (レポートは誤りと判定していた) |
| `nemd_tracker.py:562` (`log_result`) | `..., ixy, iyz, izx` | **誤り** |
| `md_multiview_trainer.py:678-679` | `"iyz": moms_i[4]`, `"izx": moms_i[5]` | **誤り** (moms_i[4] は izx、[5] が iyz) |
| `md_multiview_trainer.py:842,855` (score) | CSV の `iyz`/`izx` と要素比較 | **誤り** (score が不当に悪化する) |

- 勾配には影響しない (dynamics loss は `get_wrench` 経由で、パックされた `moms_i` ベクトルを通らない)。影響は wandb / `result-ep*.json` のラベルと score のみ
- 修正は `:562` と trainer 側の 2 箇所を `[..., ixy, izx, iyz]` に合わせるのが最小。データセット側の `labels` (`..., ixy, iyz, izx`) と表記順が異なる点に注意 (どちらも値としては正しい。並び規約が違うだけ)

### 2-2. 参考: 検証方法

乱数の twist/dtwist に対し `regressor @ phi` が Newton-Euler の力・トルクを再現するかを、慣性 6 成分の全 720 順列で総当たり。生成側の回帰子は `[m, mcx, mcy, mcz, ixx, iyy, izz, ixy, iyz, izx]` の 1 通りだけが通ることを確認済み。wisp 側は `get_moments_of_inertia` に単位基底を与えて出力位置を直接観測した。

## 3. 任意の改善 (急がない)

- `wandb_add_reference.py:71` が `transform_train.json` (アンダースコア位置が実ファイル名 `transforms_train.json` と不一致) を開こうとするため現状動かない
- score / md_mse の真値は `ground_truth.csv` からしか読まれないが、同じ情報は JSON トップレベルの `global_gt` にある。そちらを読む経路を足せば CSV なしで自動採点できる (hammer の CSV は欠損中、Windows 側からのサルベージ待ち)
- `pixi-wisp-container/dynamics/dynamics.py:177-201` の `get_regressor` は正しい規約 `[..., ixy, iyz, izx]` だが、wisp 内から呼ばれていないデッドコード。2-1 を直す際に紛らわしければ削除候補
