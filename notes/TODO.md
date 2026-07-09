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
- [x] 条件数の列正規化(equilibration)を実装 (2026-07-07: `column_scale` 引数を追加, 既定 True. Van der Sluis/Swevers に基づく標準手法. 単位変換不変性を検証済み)
- [x] q4(pitch) の特異点回避が構造的に機能しない問題を解消する (2026-07-08: 真の特異点は pitch=±π/2 で, q_min/q_max=±π/4 が既に十分な安全マージンを確保していた. singularity_center/margin による q4=0 近傍の追加除外は冗長かつ有害だったため, 全 YAML の singularity_margin の q4 成分を 0 にして無効化. q_min/q_max による ±45° 制限は維持)
- [ ] 三角不等式の解析的バウンド(`compute_fourier_bounds`)が保守的すぎる問題への対処: dq/ddq を SLSQP の直接非線形制約にする方式を, 正しい初期値戦略(解析的バウンドを満たす解を初期値にする)で再挑戦する (2026-07-08: 過去の失敗はj5の実行不可能な設定が原因だった可能性が高いと判明. 実装方針は notes/LOGS/log_trajectory_optimization.md 参照)
- [x] `coeff_bounds[4]=0.13`(全 6 YAML で一律)を根拠のある値に見直す (2026-07-09: `excited.py` の `min(manual, analytical)` 併用ロジックを撤去し, `dq_max`/`ddq_max` かつ `use_analytical_bounds=True` の場合は解析的バウンドを唯一の情報源とするよう変更. この条件下で `coeff_bounds` も指定されていると `ValueError` を送出するガードを追加. 全 6 YAML から該当箇所の `coeff_bounds` を削除または q4=0.5(他関節と同一)に統一)
- [ ] `excited_6dof_20s_cond10/50/100.yaml` の `target_condition_number`(10/50/100)が実現可能域とかけ離れて陳腐化している問題への対処: nh=5/base_freq=0.1 のフル収束では cond50/cond100 が同一の cond=16.05 で早期停止し分化しない. num_harmonics・base_freq のどちらも完全収束下では悪条件化のレバーにならないと判明したため, 経験的に到達可能な範囲(概ね cond 2〜17)を踏まえ, 最適化予算(反復数)を制限した早期停止方式で target_condition_number を再設計する (詳細: notes/LOGS/log_trajectory_optimization.md 2026-07-09 エントリ, notes/ISSUES.md)
- [ ] `excited_6dof_strict.yaml` の nh スイープ(nh∈{2,3,5,10}, base_freq=0.1)を実行し, 単一調波(nh=1)の劣化を回避した非退化な本番用軌道を選定する (2026-07-09 のセッションで合意されたが未着手)

## Simulation / Identification

- [x] `excited_6dof.yaml` の `start_pos` を manipulator XML の `initial_state` に合わせる
- [ ] TLS 慣性パラメータ同定の精度検証: L2 誤差 0.209(spline) vs 0.140(excited)の差を定量的に評価する

## Code Cleanup

- [ ] `_lqr.py`(旧 LQR 実装)を削除するか残すか判断する
