# Decisions

決定台帳. 1 行 = 日付 + 採用案 / 却下: 案名 — 承認区分 + 議事録リンク.
覆された決定は行を更新する (削除しない).

- 2026-08-06 loaded_dice は再生成せず, 出荷先に同梱済みの `unperturbed_transforms.json.bak` への差し替えで対処する / 却下: データセットの再生成 — エージェント判断 (未実施) ([議事録](LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md))
- 2026-08-06 照会文書 §2 (慣性・モーメントアーム過大) と §4 (フレーム対応の破綻) の推論は取り下げ, いずれもノイズ振幅で説明する — エージェント判断 ([議事録](LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md))
- 2026-08-06 出荷前チェックは `global_gt` をセンサ系へ変換したうえで torque 閾値 2e-2 として採用する / 却下: 照会文書の原案 (変換なし・両ブロック 1e-2) — エージェント判断 (未実施) ([議事録](LOGS/2026-08-06_loaded-dice-wrench-inconsistency.md))
