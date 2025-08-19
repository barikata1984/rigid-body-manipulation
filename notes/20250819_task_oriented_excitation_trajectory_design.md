# タスク指向の励起軌道設計（アプローチ2）

## 1. 目的

始点 `q_start` から終点 `q_end` への**移動タスク全体を、慣性パラメータ同定にとって最も情報量が多くなる（＝励起的な）一本の軌道として設計**します。これにより、物理的なタスクを実行しながら、同時に効率的なデータ収集を行うことを目指します。遷移区間と励起区間という区別をなくし、タスク全体を滑らかで最適化された一本の軌道として扱います。

## 2. 手法

軌道 `q(t)` を、**基本軌道 `q_base(t)`** と **励起軌道 `q_excitation(t)`** の和として表現します。

`q(t) = q_base(t) + q_excitation(t)`

### 基本軌道 `q_base(t)`
- **役割**: 始点と終点を滑らかに結び、タスクの基本的な移動を担います。
- **実装**: **7次多項式スプライン**を用います。これにより、始点と終点での位置・速度・加速度・ジャークがゼロであることを保証でき、静止状態からの滑らかな開始と終了が可能です。

### 励起軌道 `q_excitation(t)`
- **役割**: 基本軌道に重畳され、パラメータ同定に必要な周波数成分を軌道に加えます。
- **実装**: **フーリエ級数**（複数の正弦波・余弦波の和）で表現します。
- **重要**: 軌道全体の連続性を保つため、励起軌道は始点と終点でゼロになる必要があります。これは、フーリエ級数に**窓関数**（例: Tukey窓やHanning窓）を乗算することで実現します。
  `q_excitation(t) = window(t) * (フーリエ級数)`

> #### 補足: 窓関数とは？
>
> 窓関数とは、信号処理で使われる特殊な関数で、**中央部分が1で、両端に向かって滑らかに0に収束する、釣鐘や台形のような形**をしています。
>
> **目的と効果:** 元の信号（今回は励起のための振動）に窓関数を掛け合わせると、信号の両端が滑らかに減衰し、ゼロになります。これは、照明を急にON/OFFするのではなく、**調光スイッチでじわっと明るくし、じわっと暗くする**のと同じ効果です。
>
> このアプローチでは、窓関数を使うことで、励起成分が軌道の始点と終点で確実にゼロになり、静止状態の基本軌道と**滑らかに接続されることを保証**します。これにより、不自然な急発進や急停止を防ぎます。

### 最適化
- **決定変数**: 励起軌道を構成するフーリエ級数の係数。
- **目的関数**: 軌道全体 `q(t)` から計算される**情報行列の条件数を最小化**します。
- **制約条件**: 関節の可動域、速度、トルク制限などを制約として加え、物理的に実現可能な軌道を保証します。

## 3. 修正が必要となる処理

修正は `trajectories/optimal_excitation.py` ファイルが中心となります。

1.  **新規関数の作成**: `generate_task_oriented_excitation_trajectory` のような、このアプローチ専用の新しい軌道生成関数を作成します。
2.  **最適化ロジックの構築**: 新しい軌道モデル `q(t)` を用いて、目的関数（条件数計算）と制約関数（物理限界チェック）を定義し、`scipy.optimize.minimize` を使って最適化を実行するロジックを実装します。
3.  **ヘルパー関数の追加**: 窓関数付きのフーリエ級数を生成する関数を新たに追加します。

## 4. 具体的な修正提案

`trajectories/optimal_excitation.py` に以下のような骨格のコードを追加します。

```python
# trajectories/optimal_excitation.py に追加

import numpy as np
from scipy.optimize import minimize
from scipy.signal.windows import tukey  # 窓関数としてTukey窓を利用

from trajectories.spline_interpolation import BoundaryCondition, generate_spline_trajectory
from dynamics.dynamics import calculate_condition_number

def generate_task_oriented_excitation_trajectory(
    start_qpos: np.ndarray,
    end_qpos: np.ndarray,
    duration: float,
    fps: int,
    n_harmonics: int,
    base_frequency: float,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    ee_body_name: str,
    optimization_max_iter: int = 50,
) -> dict:
    """始点から終点まで、全体が最適化されたタスク指向の励起軌道を生成する。"""
    n_joints = m.njnt

    # 1. 基本軌道 q_base(t) を7次スプラインで生成
    start_cond = BoundaryCondition(qpos=start_qpos.tolist(), qvel=[0]*n_joints, qacc=[0]*n_joints, qjerk=[0]*n_joints)
    end_cond = BoundaryCondition(qpos=end_qpos.tolist(), qvel=[0]*n_joints, qacc=[0]*n_joints, qjerk=[0]*n_joints)
    base_traj_data = generate_spline_trajectory("seventh", duration, fps, start_cond, end_cond)
    q_base_pos = base_traj_data[:, 0, :].T
    q_base_vel = base_traj_data[:, 1, :].T
    q_base_acc = base_traj_data[:, 2, :].T

    # 2. 励起成分の係数を最適化
    initial_coeffs = np.random.rand(n_joints, n_harmonics, 2) * 0.1  # 初期振幅は小さめに
    opt_args = (q_base_pos, q_base_vel, q_base_acc, m, d, duration, fps, base_frequency, ee_body_name)

    result = minimize(
        fun=_task_oriented_objective,
        x0=initial_coeffs.flatten(),
        args=opt_args,
        method="SLSQP",
        constraints=[{'type': 'ineq', 'fun': _joint_limit_constraint, 'args': opt_args}],
        options={'maxiter': optimization_max_iter, 'disp': False}
    )
    optimal_coeffs = result.x.reshape(n_joints, n_harmonics, 2)

    # 3. 最適化された係数を用いて最終的な軌道を生成して返す
    return _generate_combined_trajectory(optimal_coeffs, *opt_args)

def _task_oriented_objective(coeffs_flat: np.ndarray, *opt_args) -> float:
    """最適化の目的関数。条件数を計算する。"""
    m, d, ee_body_name = opt_args[3], opt_args[4], opt_args[8]
    n_joints = m.njnt
    n_harmonics = coeffs_flat.shape[0] // (n_joints * 2)
    coeffs = coeffs_flat.reshape(n_joints, n_harmonics, 2)
    
    traj = _generate_combined_trajectory(coeffs, *opt_args)
    joint_traj = np.stack([traj["qpos"].T, traj["qvel"].T, traj["qacc"].T], axis=1)
    
    return calculate_condition_number(m, d, joint_traj, ee_body_name)

def _generate_combined_trajectory(coeffs: np.ndarray, *opt_args) -> dict:
    """基本軌道と窓関数付き励起軌道を結合するヘルパー関数。"""
    q_base_pos, q_base_vel, q_base_acc, _, _, duration, fps, base_frequency, *_ = opt_args
    n_frames = int(duration * fps)
    
    # 窓関数を適用した励起軌道を生成 (このヘルパー関数は別途定義)
    t, exc_pos, exc_vel, exc_acc, exc_jerk = _generate_sinusoidal_trajectory_windowed(
        duration, fps, coeffs, base_frequency, window_func=tukey, alpha=0.2
    )
    
    # 基本軌道と励起軌道を足し合わせる
    full_qpos = q_base_pos + exc_pos
    full_qvel = q_base_vel + exc_vel
    full_qacc = q_base_acc + exc_acc
    # (qjerkも同様に計算)

    return {"t": t, "qpos": full_qpos, "qvel": full_qvel, "qacc": full_qacc, ...}

# _joint_limit_constraint や _generate_sinusoidal_trajectory_windowed などの
# ヘルパー関数も同様に定義する必要があります。
```
