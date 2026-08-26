# 2026-08-26 MuJoCo 状態・力覚値の時刻同期修正

## Topic
MuJoCo の状態と力覚値の 2 ms 時刻同期バグの修正

## Summary
`mj_step` の積分後 state と積分前 derived data を同じ frame に保存する不整合を修正した。
記録 frame の直前だけ `mj_forward` を実行し、記録時刻と記録形式を維持したまま全量を同一時刻へそろえた。
loaded_dice の clean 条件では `regressor @ GT` と wrench が機械精度で一致した。
毎 step の追加 forward は約 91% の負荷増だったため退け、約 15% の負荷増となる記録時限定案を採用した。

## History
前セッションの診断では、`mj_step` 後の `qpos/qvel` と積分前の forward 計算で得た `qacc/sensordata` が次の反復で同じ frame に保存されていた。
積分刻みは 0.002 秒である。
詳細は `notes/LOGS/2026-08-26_dataset-merge-and-noise-model.md` と `notes/debug/2026-08-26_merged-dataset-and-ols-results.md` の 4.6 節に記録されている。

当初は修正方法として次の案を比較した。

- `mj_step` 後に `mj_forward` を実行し、積分後 state に対応する derived data を再計算する
- 記録位置を変更し、積分前 state と step 内で計算された derived data を組み合わせる

MuJoCo 3.10.0 の最小モデルでは、`mj_step` 後に `qpos/qvel` だけが進み、`qacc/sensordata` は古い値を保持した。
同じ state に `mj_forward` を適用すると、`qacc/sensordata` は時刻を進めずに更新された。
loaded_dice の高速回転状態では、同時刻の最大絶対誤差は 2.2e-16 だった。
step 直後の混在状態では最大絶対誤差が 2.6e-3 になった。
forward 後は 4.7e-16 に戻った。
この対照により、座標変換、GT、回帰式を原因から除外した。
今回の MuJoCo 3.10.0 環境では、API の実際の挙動が時刻ずれを説明した。

記録位置を変更する案では、積分前の `qpos/qvel` を退避する必要がある。
画像とカメラ姿勢も同じ積分前 state にそろえなければ、別の 2 ms ずれが生じる。
split step または手動積分を使う場合は MuJoCo の step 処理への依存が増える。
`mj_forward` 案は追加計算を要するが、frame 時刻、画像、カメラ姿勢、記録形式を変更しない。
この段階では、初期記録前と各 `mj_step` 後に `mj_forward` を実行する案を採用した。

各 step で forward dynamics を 2 回実行する理由が不明確だという指摘を受け、MuJoCo の公式資料と実装を調べ直した。
公式 simulation documentation は、`mj_step` が現在 state の forward dynamics を計算してから state を積分すると説明している。
公式実装も `mj_forward` の後に選択された integrator を呼んでいる。
したがって `mj_step` の戻り後は state だけが新時刻を表し、derived data は積分前時刻を表す。
公式 maintainer は [issue #498 のコメント](https://github.com/google-deepmind/mujoco/issues/498#issuecomment-1253051783) と [issue #1667 のコメント](https://github.com/google-deepmind/mujoco/issues/1667#issuecomment-2118840670) でこの契約を明示している。
issue #498 の同コメントでは、post-step `mj_forward` は高価な解決策とも説明している。
`mj_step1` と `mj_step2` を逆順に配置する dm_control の旧方式では、位置・速度依存値だけを新 state にそろえられる。
加速度・力依存 sensor は `mj_step2` 内で計算直後に積分されるため、新 state にはそろえられない。

一回の `mj_step` だけで記録する案を、全 integrator で追加検証した。
pre-step の `qpos/qvel` と step 後に残る `qacc/sensordata` を組み合わせると、Euler、implicit、implicitfast では最大絶対誤差 2.2e-16 になった。
RK4 では最大絶対誤差が 4.7e-4 残った。
RK4 では derived pose も pre-step から 5.8e-4 変化した。
一回化は現在の Euler 設定では成立するが、integrator に依存して記録規約を変える案だった。

一般性を維持したまま追加計算を減らす案として、記録 frame の直前だけ `mj_forward` を実行する方法を比較した。
loaded_dice で 5000 step を 5 回計測した。
標準 `mj_step` の中央値は 0.0110 秒だった。
毎 step で forward を追加した場合は 0.0210 秒だった。
60 Hz の記録時だけ追加した場合は 0.0127 秒だった。
記録時限定案は integrator と sensor noise profile の意味を変更しない。
以上から、毎 step の post-step forward と pre-step state の退避案を退け、記録 frame の直前だけ `mj_forward` を実行する案へ変更した。

参照した一次資料は、[公式 simulation documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html)、[MuJoCo の `engine_forward.c`](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_forward.c)、[issue #498 の maintainer コメント](https://github.com/google-deepmind/mujoco/issues/498#issuecomment-1253051783)、[issue #1667 の maintainer コメント](https://github.com/google-deepmind/mujoco/issues/1667#issuecomment-2118840670)、[dm_control の `Physics.step`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py) である。

旧ループで生成した通常データでは、画像、姿勢、`qpos/qvel` が時刻 t を表す。
同じ frame の `qacc` と wrench は時刻 t−2 ms を表す。
このため `dtwist_sen`、regressor、`jointvars_clean`、wrench の組と、それらから計算した LS、TLS、OLS 評価は無効である。
画像、マスク、カメラ軌道、GT、ノイズモデルの単体統計はこの同期バグでは無効にならない。
診断時に明示的な `mj_forward` を入れた `*_forward_diag` はこの失効対象から外れる。

## Decisions
- 記録 frame の直前だけ `mj_forward` を実行する / 却下: 各 `mj_step` 後の forward、積分前 state の退避 — エージェント判断
- データセットと配布 zip の再生成は同期修正とは別の後続作業とする — ユーザー確認済み

## Changes
- `simulators/simulator.py` — 記録 frame の MuJoCo derived data だけを `mj_forward` で同期
- `tests/test_simulator_time_sync.py` — loaded_dice の clean 条件で `regressor @ GT` と wrench の機械精度一致を検査する回帰試験を追加
- `notes/TODO.md`、`notes/ISSUES.md`、`notes/DECISIONS.md` — 同期修正の完了と後続作業を反映
- 検証: `pixi run -e dev pytest -q` は 38 passed、変更対象の Ruff は pass
- データセットと zip の変更は None

## Open Items
- 同期修正後に clean / control only / record joint only / wrench only / all を再生成し、旧配布 zip を置き換える
- 元の nomain 軌道と D-opt 8π 軌道をノイズ入りで再比較する
- 同期修正後のデータで FT300-S 実測 profile のトルクノイズ寄与と loaded_dice の 10 種評価をやり直す
