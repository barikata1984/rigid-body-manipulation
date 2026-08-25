# Decisions

決定台帳. 1 行 = 日付 + 採用案 / 却下: 案名 — 承認区分 + 議事録リンク.
覆された決定は行を更新する (削除しない).

- 2026-08-03 ルート直下の素の `.json` を 1 個に絞る命名規則 (`split_file_name` + `primary_prefix`) を採用する — 承認区分不明 (2026-08-25 事後再構成) ([議事録](LOGS/2026-08-03_wisp-dataset-compatibility.md))
- 2026-08-03 `ls`/`tls` を `global_gt` と同じ物体 aabb 座標系で書き出す — 承認区分不明 (2026-08-25 事後再構成. 8/6 の `global_gt` センサ系変換の決定と方針が衝突しており統一未決定 → ISSUES.md) ([議事録](LOGS/2026-08-03_wisp-dataset-compatibility.md))
- 2026-08-03 alpha 生成を RGB ヒューリスティックから segmentation マスクへ置換し `masks/` を新設する — 承認区分不明 (2026-08-25 事後再構成) ([議事録](LOGS/2026-08-03_wisp-dataset-compatibility.md))
- 2026-08-03 新旧データセットの判別は `labels` の要素数 (11/10) で行い座標系キーは追加しない — 承認区分不明 (2026-08-25 事後再構成) ([議事録](LOGS/2026-08-03_wisp-dataset-compatibility.md))
- 2026-08-06 loaded_dice は再生成せず, 出荷先に同梱済みの `unperturbed_transforms.json.bak` への差し替えで対処する / 却下: データセットの再生成 — エージェント判断 (未実施) ([議事録](LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md))
- 2026-08-06 照会文書 §2 (慣性・モーメントアーム過大) と §4 (フレーム対応の破綻) の推論は取り下げ, いずれもノイズ振幅で説明する — エージェント判断 ([議事録](LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md))
- 2026-08-06 出荷前チェックは `global_gt` をセンサ系へ変換したうえで torque 閾値 2e-2 として採用する / 却下: 照会文書の原案 (変換なし・両ブロック 1e-2) — エージェント判断 (未実施) ([議事録](LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md))
