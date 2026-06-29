# Issues

## [2026-06-30] tyro MISSING デフォルト警告

`coefficients`/`q0` 等を `None` に変更したが, tyro が `str` 型として MISSING を検出する警告が残っている.
動作には影響しないが出力がうるさい.

## [2026-06-30] Excited 軌道最適化が遅い

6DOF で約 30 秒/反復. 実用上の問題になりうる.

## [2026-06-30] Excited 軌道で特異姿勢に入る

q4(pitch)が 0 を通過する軌道が生成されると LQR 追従が発散する.
最適化の制約として q4≈0 を除外する必要がある.

## [2026-06-30] start_pos と XML initial_state の不一致

`excited_6dof.yaml` の `start_pos=[0,0,0,0,0,0]` と manipulator XML の `initial_state`(`qpos=[1,1,1,0,0,0]`)が合っていない.
初期位置ずれによる追従誤差の原因となる.
