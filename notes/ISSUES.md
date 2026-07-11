# Issues

## [2026-06-30] tyro MISSING デフォルト警告

`coefficients`/`q0` 等を `None` に変更したが, tyro が `str` 型として MISSING を検出する警告が残っている.
動作には影響しないが出力がうるさい.

## [2026-06-30] Excited 軌道最適化が遅い

6DOF で約 30 秒/反復. 実用上の問題になりうる.

## [2026-07-11] cond10/cond50/cond100.yaml の差別化が envelope タイト化で逆方向に破綻

`excited_6dof_20s_cond10/50/100.yaml` の `dq_max`/`ddq_max` envelope を 2 段階でタイト化し, 到達可能な cond フロアを約 57 まで引き上げた (現行タイト envelope: 並進 dq/ddq=0.3/0.6, 回転(j4,j5) dq/ddq=0.5/1.0, 実測: `configurations/trajectories/excited_20260711_010719/`).
これにより cond50 の差別化 (target_condition_number=50 で早期停止) は意図通り機能するようになった.

しかし 3 YAML を同一 envelope にしたため, 差別化の破綻が逆方向で再発した: cond10(target=10) はこのフロア (57) より低い値を要求しており到達不能で, 事実上フル実行 (`max_iter × n_restarts`) になる. cond100(target=100) は 1 反復目の cond (57.85) が既に target を下回るため, 実質未最適化のまま即発火する.

対応候補: (a) 3 YAML それぞれに別 envelope を設定する (cond10 は緩め, cond50 は現行のタイト envelope, cond100 はさらにタイトに), (b) `target_condition_number` による早期停止という差別化機構自体を再設計する. 判断は未着手 (詳細: `notes/LOGS/log_trajectory_optimization.md` 2026-07-11 エントリ, `notes/TODO.md`).
