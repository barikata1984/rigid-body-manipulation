# 2026-08-03 wisp データセット互換性対応

> 注記: 本議事録は 2026-08-25 に事後再構成したものである。対象セッション (2026-07-31〜08-03 頃) は議事録を残さず終了しており、本文は `hammer_spline_run1_dataset_report.md` (7/31 受領)、`wisp_handoff_20260803.md` (8/3 送付)、および当時の未コミット差分から復元した。会話記録が存在しないため、各判断の承認区分は不明である。

## Topic

wisp/nemd 側からの互換性調査レポートを受けたデータセット生成側の修正と、wisp 側バグの検証・引き継ぎ

## Summary

wisp/nemd 側の互換性調査レポート (7/31) が挙げたハードエラー 2 件 (ルート直下の JSON 8 個、`file_path` の絶対パス) と学習破綻 1 件 (背景の黒画素) を生成側で解消し、あわせて `ls`/`tls` の座標系を `global_gt` と同じ物体 aabb 系へ揃えた。座標系変換のために `transfer_iparams` 系 3 関数を `dynamics/dynamics.py` に追加した。7/31 レポート §5 の慣性テンソル非対角ラベルの正誤判定は実行検証の結果逆と判明し、wisp 側の要修正 3 箇所を特定して 8/3 の引き継ぎメモで伝達した。変更はすべて未コミットのまま残された。

## History

2026-07-31、wisp/nemd 側から `hammer_spline_20260731_113007_run1` データセットの互換性調査レポートを受領した (`hammer_spline_run1_dataset_report.md`)。指摘された修正項目は 3 件だった。

- ルート直下に `*.json` が 8 個あり、ローダの glob が想定する 1 個と衝突する (ハードエラー)
- 各 frame の `file_path` が生成マシンの絶対パスで、他マシンでは無言で壊れる
- alpha=0 画素の RGB が黒で、白背景合成の学習が破綻する

これを受けて生成側を修正した。JSON の命名は `recorders/standard_recorder.py` に `split_file_name` を新設し、`primary_prefix` に指定した系列の無 suffix 版だけが素の `.json`、他 7 種は `.json.bak` を取る方式にした。`main.py` は非摂動系列がある場合に `primary_prefix = "unperturbed_transforms"` へ切り替える。`file_path` は `simulators/simulator.py` でデータセットルート相対 (`complete/0000.png` 形式) に変更した。背景は `simulators/setup.py` で skybox を白にし、あわせて従来の「RGB が全部 0 なら背景」というヒューリスティックな alpha 生成を廃止して、MuJoCo の segmentation レンダリング由来の 2 値マスクを `masks/` に書き出し、alpha にも同じマスクを使うようにした。

追加で 2 件の仕様変更を行った。`labels` から `aabb_scale` を除去し、`labels`/`global_gt`/`ls`/`tls` を 10 要素に統一した (`aabb_scale` はトップレベルキーとして存続)。また `ls`/`tls` (LS/TLS 同定結果) を `global_gt` と同じ物体 aabb 座標系で書き出すようにした。従来はセンサ座標系のままで、z 軸 180° 回転分 `mx, my, iyz, izx` の符号が逆だった。この変換のために `dynamics/dynamics.py` へ `iparams_to_simat` / `simat_to_iparams` / `transfer_iparams` (10 次元慣性パラメータ ⇄ 空間慣性テンソルの相互変換と座標系移送) を追加した。効果は chair のノイズなし L2 が 0.065 → 0.0035。hammer は対称物体のため不変である。JSON に座標系を示すキーは設けず、`labels` の要素数 (11 なら旧形式、10 なら新形式) を判別基準とした。

検証は `datasets/hammer/spline_20260803_173638_run2` (300 frames) で行い、alpha とマスクの一致率 100% を確認した。この確認のためにスプライン軌道 `spline_20260803_173638` を生成しており、`catalog.json` への追記はこの run である。

7/31 レポート §5 が報告した wisp 側の慣性テンソル非対角ラベル取り違えは、wisp 環境で `get_moments_of_inertia` に単位基底を与えて列対応を実測した結果、レポートの正誤判定が逆だと判明した。実装の返す並びは `[ixx, iyy, izz, ixy, izx, iyz]` であり、`nemd_tracker.py:515` は正しく、`nemd_tracker.py:562`、`md_multiview_trainer.py:678-679`、同 `:842,855` (score 計算) の 3 箇所が誤りである。勾配には影響せず、影響は表示ラベルと score のみ。生成側回帰子の規約は乱数 twist に対する Newton-Euler 再現の全 720 順列総当たりで `[m, mcx, mcy, mcz, ixx, iyy, izz, ixy, iyz, izx]` の 1 通りだけが通ることを確認した。

以上を 2026-08-03 に `wisp_handoff_20260803.md` として wisp/nemd 側へ引き継いだ。生成側の変更はコミットせず、notes への記録も行われないままセッションが終了した。この記録欠落は 8/25 のセッションで発覚し、本議事録の事後再構成に至った。

なお `simulators/setup.py` にはトラッキングカメラ距離を `4*aabb_scale` から `5*aabb_scale` へ変える差分も含まれるが、この変更はレポートにも引き継ぎメモにも記載がなく、理由は復元できなかった。

## Decisions

以下はいずれも事後再構成であり、承認区分の記録はない。

- ルート直下の素の `.json` を 1 個に絞る命名規則 (`split_file_name` + `primary_prefix`) を採用 — 承認区分不明 (事後再構成)
- `ls`/`tls` を `global_gt` と同じ物体 aabb 座標系で書き出す — 承認区分不明 (事後再構成)
- alpha 生成を RGB ヒューリスティックから segmentation マスクへ置換し、`masks/` を新設 — 承認区分不明 (事後再構成)
- 新旧データセットの判別は `labels` の要素数 (11/10) で行い、座標系キーは追加しない — 承認区分不明 (事後再構成)

## Changes

- `dynamics/dynamics.py`, `dynamics/__init__.py`: `iparams_to_simat` / `simat_to_iparams` / `transfer_iparams` 追加 (未コミット)
- `main.py`: `identify_inertial_params` へ `pose_obj_sen` 追加、`ls`/`tls` の物体 aabb 系変換、`primary_prefix` 切替 (未コミット)
- `recorders/standard_recorder.py`: `split_file_name` 新設、`masks/` 生成、`labels` 10 要素化 (未コミット)
- `simulators/simulator.py`: `file_path` 相対化 (未コミット)
- `simulators/setup.py`: skybox 白化、カメラ距離 4→5 倍 (未コミット、後者は出所不明)
- `configurations/trajectories/spline_20260803_173638/` 生成、`catalog.json` 追記 (未コミット)
- `wisp_handoff_20260803.md` 作成 (未追跡)
- `tests/test_iparams_transfer.py`, `tests/test_split_file_name.py` 追加 (未追跡)

## Open Items

- 生成側変更一式のコミット
- カメラ距離 4→5 倍の変更の採否判断 (理由の記録なし)
- 座標系の統一方針の決定: 本セッションは `ls`/`tls` を物体 aabb 系へ寄せたが、8/6 の TODO は `global_gt` をセンサ系へ寄せる指示になっており、両方実施すると同一 JSON 内で座標系が割れる
- wisp 側バグ 3 箇所 (`nemd_tracker.py:562`, `md_multiview_trainer.py:678-679`, `:842,855`) の修正 (wisp 側リポジトリの作業)
- `wandb_add_reference.py:71` のファイル名不一致 (`transform_train.json`)、`global_gt` からの自動採点経路 (任意改善、handoff §3)
- hammer の `ground_truth.csv` 欠損、Windows 側からのサルベージ待ち
