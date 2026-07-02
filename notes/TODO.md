# TODO

## Config / CLI

- [ ] tyro の MISSING デフォルト警告を抑制する(cosmetic だが出力がうるさい)

## Trajectory

- [x] ExcitedTrajectory の特異姿勢回避: q4≈0 を除外する制約を最適化に追加する (Stage 0 完了: ペナルティ法 + coeff_bound; 2026-07-02: `singularity_center`/`singularity_margin` として関節ごとに再実装, `q_min`/`q_max` とは別建て)
- [x] 関節ごとの coeff_bounds 実装 (Stage 1-3 完了: per-joint coeff_bounds, q_min/q_max 制約, compute_fourier_bounds 移植)
- [x] D-optimal 目的関数 (Stage 4 完了: objective_type で条件数/D-optimal 切替)
- [x] マルチスタート最適化 (Stage 5 完了: n_restarts, seed, early_stop_patience)
- [ ] Excited 軌道最適化の高速化(現状 ~30 秒/反復, 6DOF)
- [ ] `ExcitationTrajectory` クラス (`excitation.py`) がどこにもインポートされていないか確認し, 削除または復活させる
- [ ] singularity_margin と dq_max/ddq_max 由来の coeff_bounds の不整合を解消する (現状 88%(263/300 フレーム)が除外ゾーン内に留まる: coeff_bounds ~0.063 rad vs 必要マージン 0.15 rad)

## Simulation / Identification

- [x] `excited_6dof.yaml` の `start_pos` を manipulator XML の `initial_state` に合わせる
- [ ] TLS 慣性パラメータ同定の精度検証: L2 誤差 0.209(spline) vs 0.140(excited)の差を定量的に評価する

## Code Cleanup

- [ ] `_lqr.py`(旧 LQR 実装)を削除するか残すか判断する
