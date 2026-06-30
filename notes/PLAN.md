# 励起軌道最適化の段階的移行計画

desktop-ur5e の制約付き最適化体系を rigid-body-manipulation に段階的に移植する.

参照元: `barikata1984/desktop-ur5e` の `src/ur5e_sim/identification/` 以下
- `constraints.py` — JointLimits, _TrajectoryCache, per-joint constraints, compute_fourier_bounds
- `objective.py` — condition_number_objective, d_optimal_objective
- `optimizer.py` — ExcitationOptimizer (multi-start, parallel, wandb)
- `workspace.py` — EE 変位, ペイロード位置, EE 速度, 干渉回避

## 現状の差分

| 機能 | desktop-ur5e | rigid-body-manipulation |
|---|---|---|
| 目的関数 | 条件数 / D-optimal 選択可 | 条件数のみ |
| ソルバー | SLSQP | L-BFGS-B + ペナルティ |
| フーリエ係数バウンド | 関節ごと × ハーモニクスごとに解析的導出 | 全関節一律 coeff_bound |
| 関節位置制約 | JointLimits + make_joint_position_constraint | ペナルティで 1 関節のみ |
| 関節速度制約 | make_joint_velocity_constraint | なし |
| 関節加速度制約 | make_joint_acceleration_constraint | なし |
| ワークスペース制約 | EE 変位上限, ペイロード, EE 速度 | なし |
| 干渉回避 | CollisionChecker | なし |
| マルチスタート | モンテカルロ N 回 + early stopping | なし (1 回) |
| 並列化 | ProcessPoolExecutor | なし |
| ロギング | wandb | print のみ |
| メイン軌道 + 励起 | なし (フーリエ単体) | あり (スプライン + 窓付きフーリエ) |

## 段階的実装

### Stage 0 (2026-06-30 完了): 特異姿勢回避の最小実装

- ペナルティ項 `weight * Σ max(0, margin - |q_j|)²` を目的関数に加算
- 全関節一律の `coeff_bound` で振幅を制限
- `singular_joint`, `singular_margin`, `coeff_bound` を ExcitedTrajectoryConfig に追加
- generate.py の OmegaConf merge バグ修正 (CLI デフォルトが YAML を上書きする問題)

結果: LQR 発散を防止. 条件数は coeff_bound の制約で悪化 (75〜1700, 設定依存).

### Stage 1 (2026-07-01 完了): 関節ごとの coeff_bounds

`coeff_bound: float` を `coeff_bounds: list[float] | float` に拡張.
float なら全関節一律 (後方互換), リストなら関節ごとに設定.

変更箇所:
- `ExcitedTrajectoryConfig.coeff_bounds`
- `ExcitedTrajectory._optimize()` のバウンド構築

YAML 例:
```yaml
coeff_bounds: [0.5, 0.5, 0.5, 0.5, 0.13, 0.5]
```

効果: 条件数が約 150 倍改善 (1693 → 11.1).

### Stage 2: 関節位置の上下限制約

desktop-ur5e の `JointLimits` + `make_joint_position_constraint` に相当.
全関節に対して `q_min <= q_total <= q_max` をペナルティで課す.
`singular_joint` / `singular_margin` を置き換える汎用的な仕組み.

変更箇所:
- `ExcitedTrajectoryConfig` に `q_min`, `q_max` (リスト) を追加
- `_optimize()` のペナルティ計算を一般化

### Stage 3: 解析的バウンド導出

desktop-ur5e の `compute_fourier_bounds` を移植.
速度・加速度限界から三角不等式でフーリエ係数の安全なバウンドを自動計算.
手動の `coeff_bounds` チューニングが不要になる.

変更箇所:
- `constraints.py` を新設 (desktop-ur5e から移植)
- 窓関数の違い (64s³(1-s)³ → 256s⁴(1-s)⁴) に合わせて導出式を修正

注意: rigid-body-manipulation はメイン軌道 + 励起の構造なので, バウンドは励起成分のみに適用する.

### Stage 4: D-optimal 目的関数

desktop-ur5e の `d_optimal_objective` を移植.
`-log det(W^T W) = -2 * Σ log(σ_i)` を最小化.
条件数最小化より数値的に安定で, 局所最適解に陥りにくい.

変更箇所:
- `objective.py` を新設または `base_trajectory.py` に追加
- `ExcitedTrajectoryConfig.objective_type` で切り替え

### Stage 5: マルチスタート最適化

desktop-ur5e の `ExcitationOptimizer` のマルチスタートループを移植.
N 個のランダム初期値から最適化し, 最良の結果を選択.
early stopping (改善しなくなったら打ち切り) を含む.

変更箇所:
- `optimizer.py` を新設
- `ExcitedTrajectory._optimize()` から最適化ループを分離

### Stage 6: 並列化 + wandb

マルチスタートを ProcessPoolExecutor で並列実行.
wandb で各リスタートの条件数推移を記録.

前提: Stage 5 の完了.

### Stage 7: ワークスペース・干渉回避制約

desktop-ur5e の `workspace.py` + `collision.py` を移植.
FK でエンドエフェクタ位置を評価し, 変位上限・干渉を制約.

実機適用時に必要. シミュレーションのみなら優先度は低い.

## 推奨順序

Stage 1 が条件数改善への最短ルートで, 30 分程度で実装可能.
Stage 2-3 は制約の汎用化と自動化で, desktop-ur5e との構造的な差を埋める.
Stage 4-5 は最適化品質の改善で, 同定精度に直結する.
Stage 6-7 は運用品質で, 実機適用時に必要.
