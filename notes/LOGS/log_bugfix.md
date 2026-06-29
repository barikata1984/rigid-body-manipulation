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
