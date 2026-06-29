# TODO

## Config / CLI

- [ ] tyro の MISSING デフォルト警告を抑制する(cosmetic だが出力がうるさい)

## Trajectory

- [ ] ExcitedTrajectory の特異姿勢回避: q4≈0 を除外する制約を最適化に追加する
- [ ] Excited 軌道最適化の高速化(現状 ~30 秒/反復, 6DOF)
- [ ] `ExcitationTrajectory` クラス (`excitation.py`) がどこにもインポートされていないか確認し, 削除または復活させる

## Simulation / Identification

- [ ] `excited_6dof.yaml` の `start_pos` を manipulator XML の `initial_state`(`qpos=[1,1,1,0,0,0]`)に合わせる
- [ ] TLS 慣性パラメータ同定の精度検証: L2 誤差 0.209(spline) vs 0.140(excited)の差を定量的に評価する

## Code Cleanup

- [ ] `_lqr.py`(旧 LQR 実装)を削除するか残すか判断する
