# Issues

## [2026-06-30] tyro MISSING デフォルト警告

`coefficients`/`q0` 等を `None` に変更したが, tyro が `str` 型として MISSING を検出する警告が残っている.
動作には影響しないが出力がうるさい.

## [2026-06-30] Excited 軌道最適化が遅い

6DOF で約 30 秒/反復. 実用上の問題になりうる.

## [2026-07-09] cond10/cond50/cond100.yaml が条件数を分化できていない

`excited_6dof_20s_cond10/50/100.yaml` の差別化機構を `target_condition_number`(10/50/100)による早期停止に変更したが, cond50 と cond100 が現状**同一の cond=16.05 で収束**し, 分化できていない (実行例: `configurations/trajectories/excited_20260709_154917`, `_155019`).

原因: q4 の coeff_bounds バグ修正 (本ファイル旧エントリ, 解決済み) と q5 の速度制約撤去により, この envelope で到達可能な条件数の範囲が大きく改善した (nh=5, base_freq=0.1 のフル収束で cond=6.44 程度まで到達可能, 詳細は `notes/LOGS/log_trajectory_optimization.md` 2026-07-09 エントリ参照). そのため target_condition_number=50/100 は共に「最初の1反復で既に上回っている」水準になり, 両者とも実質未最適化のまま停止する.

さらに調査した結果, num_harmonics・base_freq のいずれも, 最適化をフル収束させる限り確実に悪条件を生むレバーにはならないと判明した (nh=1: cond=7.45, base_freq=0.02: cond=7.30, base_freq=1.0: cond=2.40 — いずれも一桁〜低い二桁に収束). 意図的に悪条件な軌道を得るには, パラメータ選択ではなく最適化予算(反復数)を制約する早期停止方式への再設計が必要と考えられる.

対応候補: `target_condition_number` を経験的に到達可能な範囲(概ね 2〜17)を前提に再設計する. `notes/TODO.md` 参照.

