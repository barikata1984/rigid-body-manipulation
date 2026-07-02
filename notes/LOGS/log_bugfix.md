# Bugfix Log

## 2026-06-29〜2026-06-30: tyro+OmegaConf config システムのバグ修正

### 背景

`dev-use_pure_tyro` から `dev-use_instantiate` ブランチへ移行後, Hydra→tyro マイグレーションの残骸による複数のバグを修正した.
並列エージェントチーム(worktree 分離)を用いて修正し, `dev-use_instantiate_fixes` ブランチを作成した.

### 修正内容

**Config クラス**
- `ExcitedTrajectoryConfig`: `guide_config` → `main_trajectory` にリネーム, None ガード追加, `max_iter` フィールド追加
- `fourier.py`: `coefficients`/`q0` のデフォルトを `MISSING` → `None` に変更

**参照の統一**
- `QuinticSplineTrajectory` → `SplineTrajectory` の参照をコードと YAML 全体で修正

**基底クラス**
- `base_trajectory.py`: `_generate(args, kwargs)` → `_generate(*args, **kwargs)` に修正, `return pos, vel, acc` を追加, `plt.close(fig)` 追加

**個別クラス**
- `spline.py`: `_generate` シグネチャを `*args, **kwargs` に統一
- `excitation.py`: 裸の `num_joints`/`num_harmonics` → `self.*` に修正, `generate()` → `_generate()` にリネーム

**エントリポイント**
- `main.py`: `target_trajectory = None` を条件分岐の前に初期化, fps ガード追加
- `simulator.py`: 裸の `except: pass` を `kwargs.get()` に置換, 軌道未設定時の明示的 `ValueError` 追加, `_visualize` をファイル保存に変更
- `lqr.py`: `except ValueError` → `except (ValueError, MissingMandatoryValue)` に変更, エラーメッセージ修正

**YAML**
- `excited_6dof.yaml`: `num_joints`, `type`, `target_class`, `start_pos` を修正
- `spline_6dof.yaml`: `type: quintic`, `config_class` を修正

### 結果

- spline 軌道および excited 軌道の生成に成功
- MuJoCo シミュレーション(両軌道タイプ)の実行に成功
- TLS 慣性パラメータ同定: L2 誤差 0.209(spline) → 0.140(excited)に改善(特異姿勢を踏まない場合)

### 発見した問題

- `start_pos=[0,0,0,0,0,0]` と manipulator XML の `initial_state`(`qpos=[1,1,1,0,0,0]`)が不一致
- excited 軌道で q4(pitch)が 0 を通過すると特異姿勢に入り, 追従が発散する
- `ExcitationTrajectory` クラス(`excitation.py`)はどこにもインポートされていないデッドコード
- `_lqr.py` は旧実装, `lqr.py` が現役

## 2026-07-01: generate.py OmegaConf merge バグ修正

### 背景

`generate.py` の CLI 引数処理で, tyro のデフォルト値が YAML に書かれた設定値を上書きする問題があった.
OmegaConf の merge 順序が逆になっており, YAML 側の意図した設定が無効化されていた.

### 修正内容

- `generate.py`: OmegaConf merge の順序を修正 (CLI デフォルトを base, YAML を override として適用)

### 結果

- YAML で指定した `coeff_bounds`, `q_min`, `q_max` 等が CLI 実行時に正しく反映されるようになった

## 2026-07-02: コードレビュー (bf017ea~1..be07e55) とバグ修正

### 背景

`/code-review` スキル (high effort, 8 finders × 14 verifications) で, 特異姿勢回避 + 制約付き励起軌道最適化 Stage 0-5 (commit range `bf017ea~1..be07e55`, 7 commits) をレビューした.
10 件の指摘 (CONFIRMED 6 件, PLAUSIBLE 4 件) が得られ, 3 エージェントの並列実装ですべて修正した.

### 修正内容

**`trajectories/excited.py`**
- q4(pitch) 近傍を除外する新しい制約を追加. `ExcitedTrajectoryConfig` に関節ごとの `singularity_center` / `singularity_margin` を追加した. 既存の `q_min`/`q_max` は範囲内に収める包含制約であり, q4≈0 を除外するという要求とは論理的に逆であるため, 別の仕組みとして実装した. L-BFGS-B / SLSQP 両バックエンドでペナルティ法により実装 (除外領域が非凸のため, SLSQP でも不等式制約ではなくペナルティ法を採用).
- `_generate_random_x0` のインデックス順序バグを修正: `x[k*nj+j]` → `x[j*nh+k]` (`_build_trajectory` が前提とする C-order reshape に合わせた).
- `seed` がリスタート 0 (`self.a`/`self.b` の初期値) に適用されていなかったバグを修正: `np.random.default_rng(cfg.seed)` を使うよう変更.
- 条件数が NaN のとき `best_x` が `None` のまま残りクラッシュするバグを修正.
- `objective_type` / `optimizer_method` の値検証を追加 (不正値で `ValueError`).
- `coeff_bounds` / `q_min` / `q_max` / `dq_max` / `ddq_max` / `singularity_center` / `singularity_margin` の長さ検証を追加.
- `_build_q_constraints` で `FourierTrajectory` が重複構築されキャッシュされていなかったバグを修正: `_run_single_optimization` と共有する `_get_trajectory` キャッシュを使うよう変更.

**`configurations/trajectory_generation/excited_6dof.yaml`**
- スライド関節の `dq_max`/`ddq_max` の typo を修正 (0.2→2.0 m/s, 0.4→8.0 m/s^2; コメントが元々記述していた値に合わせた).
- `singularity_center: [0,0,0,0,0,0]` と `singularity_margin: [0,0,0,0,0.15,0]` (q4 のみ有効) を追加した. margin=0.15 は暫定値で, 実験的なチューニングが必要とコメントに明記.

**`trajectories/generate.py`**
- CLI 上書き検出バグを修正. 旧実装は"CLI 値がデフォルトと異なる場合のみ上書きする"という値比較のヒューリスティックを使っており, ユーザーが明示的に指定した CLI 値がたまたまデフォルト値と一致する場合に YAML の値へフォールバックしてしまう欠陥があった. `sys.argv` 中のフラグトークンの有無を検出する方式に置き換えた (tyro には明示指定を検出するネイティブな仕組みがなく, `brentyi/tyro#254` で未実装のまま報告されていることを確認済み).

**`configurations/trajectory_generation/spline_6dof.yaml`**
- `start_pos` を `[1,1,1,0,0,0]` から `[0,0,0,0,0,0]` に修正した. 同じ diff 内で更新された `manipulator.xml` の新しい `initial_state` キーフレームに合わせた. 修正前は, 他所で直したばかりの初期位置不一致バグと同種のリグレッションが spline_6dof.yaml に残っていた.

### 結果

3 エージェントの並列実装ですべての指摘を修正した. 修正後の動作検証結果は `log_trajectory_optimization.md` の 2026-07-02 エントリを参照.
