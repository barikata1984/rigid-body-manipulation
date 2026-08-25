# 2026-08-03 wisp データセット互換性対応

> 注記: 本議事録は 2026-08-25 に事後再構成したものである。対象セッション (2026-07-31〜08-03 頃) は議事録を残さず終了しており、本文は `hammer_spline_run1_dataset_report.md` (7/31 受領)、`wisp_handoff_20260803.md` (8/3 送付)、および当時の未コミット差分から復元した。会話記録が存在しないため、各判断の承認区分は不明である。

## Topic

wisp/nemd 側からの互換性調査レポートを受けたデータセット生成側の修正と、wisp 側バグの検証・引き継ぎ

## Summary

wisp/nemd 側の互換性調査レポート (7/31) が挙げたハードエラー 2 件と学習破綻 1 件を生成側で解消した。ローダの glob が素の `.json` 1 個を前提とするため、命名規則で素の `.json` を 1 個に絞る方式を採った。背景分離は RGB ヒューリスティックでは黒背景と区別できないため、segmentation マスク方式へ置き換えた。LS/TLS 同定結果は真値 `global_gt` と符号規約を揃えるため物体 aabb 座標系へ統一した。7/31 レポート §5 の慣性テンソル非対角ラベルの正誤判定は実測の結果逆と判明し、wisp 側の要修正 3 箇所を特定して 8/3 の引き継ぎメモで伝達した。

## History

2026-07-31、wisp/nemd 側から `hammer_spline_20260731_113007_run1` データセットの互換性調査レポートを受領した (`hammer_spline_run1_dataset_report.md`)。指摘された修正項目は 3 件だった。

- ルート直下に `*.json` が 8 個あり、ローダの glob が想定する 1 個と衝突する (ハードエラー)
- 各 frame の `file_path` が生成マシンの絶対パスで、他マシンでは無言で壊れる
- alpha=0 画素の RGB が黒で、白背景合成の学習が破綻する

1 件目の JSON 数は、命名規則の変更で解消した。指定した主系列の無 suffix 版だけが素の `.json` を取り、他 7 種は `.json.bak` を取る。これによりローダの glob には 1 個だけヒットし、全フレームが train になる。差分には、非摂動系列がある場合に非摂動側を主系列へ切り替える変更も含まれる。この切替を入れた理由と時期の記録は復元できなかった。

2 件目の `file_path` は、データセットルート相対 (`complete/0000.png` 形式) に変更した。

3 件目の背景は、二段で対処した。skybox を白にした。あわせて従来の alpha 生成を廃止した。従来方式は「RGB が全部 0 なら背景」というヒューリスティックであり、黒い前景画素を背景と誤判定する。代わりに MuJoCo の segmentation レンダリング由来の 2 値マスクを `masks/` に書き出し、alpha にも同じマスクを使うようにした。検証は `datasets/hammer/spline_20260803_173638_run2` (300 frames) で行い、alpha とマスクの一致率 100% を確認した。

追加で 2 件の仕様変更を行った。1 件目として、`labels` から `aabb_scale` を除去し、`labels`/`global_gt`/`ls`/`tls` を 10 要素に統一した (`aabb_scale` はトップレベルキーとして存続)。2 件目として、`ls`/`tls` (LS/TLS 同定結果) を `global_gt` と同じ物体 aabb 座標系で書き出すようにした。従来はセンサ座標系のままだった。両座標系は z 軸 180° 回転の関係にあり、`mx, my, iyz, izx` の符号が逆になるため、真値との突き合わせが崩れていた。変換のために 10 次元慣性パラメータと空間慣性テンソルの相互変換・座標系移送を行う関数 3 つを追加した。効果は、重心が軸から外れた別対象物 chair のノイズなし同定誤差 L2 が 0.065 → 0.0035 である (`wisp_handoff_20260803.md` §1)。hammer は対称物体のため不変である。

JSON に座標系を示すキーは設けず、`labels` の要素数 (11 なら旧形式、10 なら新形式) を新旧判別の基準とした。キーを設けなかった理由の記録は復元できなかった。

7/31 レポート §5 は、wisp 側の慣性テンソル非対角成分のラベル取り違えを報告していた。wisp 環境で `get_moments_of_inertia` に単位基底を与え、列対応を実測した。その結果、レポートの正誤判定は逆だと判明した。実装の返す並びは `[ixx, iyy, izz, ixy, izx, iyz]` である。`nemd_tracker.py:515` は正しく、`nemd_tracker.py:562`、`md_multiview_trainer.py:678-679`、同 `:842,855` (score 計算) の 3 箇所が誤りである。勾配には影響せず、影響は表示ラベルと score のみである。

生成側回帰子の並び規約も独立に確認した。乱数 twist に対する Newton-Euler 再現を慣性 6 成分の全 720 順列で総当たりし、`[m, mcx, mcy, mcz, ixx, iyy, izz, ixy, iyz, izx]` の 1 通りだけが通ることを確認した。

以上を 2026-08-03 に `wisp_handoff_20260803.md` として wisp/nemd 側へ引き継いだ。生成側の変更はコミットされず、notes への記録も行われないままセッションが終了した。この記録欠落は 8/25 のセッションで発覚し、本議事録の事後再構成に至った。

なお当時の差分にはトラッキングカメラ距離を `4*aabb_scale` から `5*aabb_scale` へ変える変更も含まれるが、レポートにも引き継ぎメモにも記載がなく、理由は復元できなかった。

## Decisions

- ルート直下の素の `.json` を 1 個に絞る命名規則 (`split_file_name` + `primary_prefix`) を採用 — 承認区分不明 (事後再構成)
- `ls`/`tls` を `global_gt` と同じ物体 aabb 座標系で書き出す — 承認区分不明 (事後再構成)
- alpha 生成を RGB ヒューリスティックから segmentation マスクへ置換し、`masks/` を新設 — 承認区分不明 (事後再構成)
- 新旧データセットの判別は `labels` の要素数 (11/10) で行い、座標系キーは追加しない — 承認区分不明 (事後再構成)

## Changes

- `dynamics/dynamics.py`, `dynamics/__init__.py`: `iparams_to_simat` / `simat_to_iparams` / `transfer_iparams` 追加
- `main.py`: `identify_inertial_params` へ `pose_obj_sen` 追加、`ls`/`tls` の物体 aabb 系変換、`primary_prefix` 切替
- `recorders/standard_recorder.py`: `split_file_name` 新設、`masks/` 生成、`labels` 10 要素化
- `simulators/simulator.py`: `file_path` 相対化
- `simulators/setup.py`: skybox 白化、カメラ距離 4→5 倍 (後者は出所不明)
- `configurations/trajectories/spline_20260803_173638/` 生成、`catalog.json` 追記
- `wisp_handoff_20260803.md` 作成
- `tests/test_iparams_transfer.py`, `tests/test_split_file_name.py` 追加
- (以上はすべて当時未コミット。2026-08-25 にコミット: 7d6f863, 56da635, e7d7b72)

## Open Items

- カメラ距離 4→5 倍の変更の採否判断 (理由の記録なし)
- 座標系の統一方針の決定: 本セッションは `ls`/`tls` を物体 aabb 系へ寄せたが、8/6 の TODO は `global_gt` をセンサ系へ寄せる指示になっており、両方実施すると同一 JSON 内で座標系が割れる
- wisp 側バグ 3 箇所 (`nemd_tracker.py:562`, `md_multiview_trainer.py:678-679`, `:842,855`) の修正 (wisp 側リポジトリの作業)
- `wandb_add_reference.py:71` のファイル名不一致 (`transform_train.json`)、`global_gt` からの自動採点経路 (任意改善、handoff §3)
- hammer の `ground_truth.csv` 欠損、Windows 側からのサルベージ待ち
