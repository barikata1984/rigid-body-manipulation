# Trajectory Optimization Log

## 2026-07-01: 励起軌道最適化 Stage 0-3 完了

### Stage 0: ペナルティ法 + global coeff_bound

ペナルティ項 `weight * Σ max(0, margin - |q_j|)²` を目的関数に追加し, 全関節一律の `coeff_bound` で振幅を制限した.
`ExcitedTrajectoryConfig` に `singular_joint`, `singular_margin`, `coeff_bound` を追加.

所見:
- SLSQP は 6 反復で終了し収束しなかった. L-BFGS-B + ペナルティ法に切り替えた.
- ペナルティ法単体では制約充足を保証できないため, `coeff_bounds` との併用が必須.
- LQR 追従の発散は防止できた. 条件数は coeff_bound の制約で悪化 (75〜1700, 設定依存).

### Stage 1: 関節ごとの coeff_bounds

`coeff_bound: float` を `coeff_bounds: list[float] | float` に拡張.
`ExcitedTrajectoryConfig.coeff_bounds` と `ExcitedTrajectory._optimize()` のバウンド構築を変更.

所見:
- per-joint coeff_bounds により条件数が約 150 倍改善 (1693 → 11.1).
- YAML 例: `coeff_bounds: [0.5, 0.5, 0.5, 0.5, 0.13, 0.5]` — q4 のみ小さく設定.

### Stage 2: q_min / q_max 関節位置制約

`singular_joint` / `singular_margin` を廃止し, 全関節に対する汎用的な位置制約 `q_min`, `q_max` に置き換えた.
desktop-ur5e の `JointLimits` + `make_joint_position_constraint` に相当.

所見:
- 単一関節のペナルティから全関節への一般化により, 任意の関節への制約が可能になった.

### Stage 3: 解析的バウンド導出 (compute_fourier_bounds)

desktop-ur5e の `compute_fourier_bounds` を移植.
速度・加速度限界から三角不等式でフーリエ係数の安全なバウンドを自動計算.

所見:
- 三角不等式によるバウンドは保守的 (過度に厳しい).
- スライド関節で `dq_max=[0.2, 0.2, 0.2, ...]` を設定すると, 解析的バウンドが係数 ~0.003 と極端に小さくなる.
- 実用上は手動チューニングした `coeff_bounds` と組み合わせる運用が現実的.

### XML キーフレームと start_pos の整合

`excited_6dof.yaml` の `start_pos` を manipulator XML の `initial_state`(`qpos`) に合わせた.
初期位置ずれによる追従誤差が解消された.

### 複数パラメータ設定での軌道生成・シミュレーション実行

Stage 1-3 の各設定で軌道生成とシミュレーションを実施し, 条件数と追従精度を確認した.

## 2026-07-01: Stage 4-5 完了

### Stage 4: D-optimal 目的関数

`base_trajectory.py` に `_build_observation_matrix`, `compute_d_optimal`, `compute_objective_with_cond` を追加.
`ExcitedTrajectoryConfig.objective_type` で `"condition_number"` / `"d_optimal"` を切り替え可能にした.

D-optimal は `-log det(Y) = -Σ log(λ_i(Y))` を最小化する.
Y = W^T W の固有値から直接計算するため, リグレッサをスタックする必要がない.

### Stage 5: マルチスタート最適化

`_optimize()` をマルチスタートループに拡張.
単一リスタートを `_run_single_optimization()` に分離し, `_generate_random_x0()` で高調波ほど小さい振幅の初期値を生成.
`n_restarts`, `seed`, `early_stop_patience` を `ExcitedTrajectoryConfig` に追加.

### 比較検証 (max_iter=5, n_restarts=3)

| 指標 | condition_number | d_optimal |
|---|---|---|
| Best Cond | 3199 | 8856 |
| L2 (LS) | 0.311 | 0.329 |
| L2 (TLS) | 0.173 | 0.189 |
| 追従 | 良好 | 良好 |
| 最適化時間 | 221s | 41s |

所見:
- 両目的関数とも LQR 追従は良好で発散なし.
- 条件数最小化の方が同定精度 (TLS L2) が良い.
- D-optimal は L-BFGS-B との相性で各リスタートの収束が浅い (1-2 反復で停止).
- D-optimal の収束改善には, リスタート数・反復数の増加, または SLSQP への移行が有効と考えられる.

### ソルバー選択の追加と SLSQP + D-optimal 検証

`optimizer_method` を config に追加. SLSQP 使用時は q_min/q_max をペナルティではなく不等式制約として渡す.

比較結果 (max_iter=5, n_restarts=3):

| ソルバー | 目的関数 | Best Cond | L2 (TLS) | 時間 |
|---|---|---|---|---|
| L-BFGS-B | condition_number | 3199 | 0.173 | 221s |
| L-BFGS-B | d_optimal | 8856 | 0.189 | 41s |
| SLSQP | d_optimal | 3410 | 0.174 | 98s |

所見:
- SLSQP + D-optimal は 5 反復/リスタート完走し, Cond = 3410, L2 = 0.174 と条件数最小化とほぼ同等の性能.
- L-BFGS-B + D-optimal (Cond = 8856) との差は 2.6 倍. D-optimal の収束不足はソルバーの問題であり, 目的関数自体の劣等ではない.
- SLSQP は L-BFGS-B より 1 反復あたりが速い (6s vs 12s). 制約を不等式制約として直接扱える利点もある.
- 追従性能は 3 パターンとも良好で差異なし.
