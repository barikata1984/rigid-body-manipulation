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
