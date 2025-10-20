# 励起軌道最適化の再現性確保とシード値設定の提案

## 1. 背景と現状の課題

`trajectories/exciting_spline.py` に実装されている `generate_exciting_spline_trajectory` 関数は、実行するたびに異なる最適化結果（軌道、条件数）を生成する。

この挙動の原因は、最適化アルゴリズム（SLSQP）に渡される初期値 `initial_coeffs` が、`numpy.random.rand()` によってランダムに生成されているためである。SLSQP自体は決定論的なアルゴリズムだが、目的関数が非凸であるため、与えられる初期値によって異なる局所最適解に収束する。

このため、特定の条件下での結果を再現・検証することが困難になっている。

## 2. 修正案

最適化結果の再現性を確保するため、ユーザーが乱数生成のシード値を明示的に設定できる機能を追加する。

具体的には、`generate_exciting_spline_trajectory` 関数にオプショナルな `seed` 引数を追加し、それが指定された場合に `numpy.random.seed()` を呼び出すように変更する。

### 2.1. 関数のシグネチャ変更

関数の定義に、シード値を指定するための `seed: int | None = None` 引数を追加する。

**変更前:**
```python
def generate_exciting_spline_trajectory(
    start_conditions: BoundaryCondition,
    end_conditions: BoundaryCondition,
    duration: float,
    fps: int,
    n_harmonics: int,
    base_frequency: float,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    ee_body_name: str,
    optimization_max_iter: int = 10,
) -> dict:
```

**変更後:**
```python
def generate_exciting_spline_trajectory(
    start_conditions: BoundaryCondition,
    end_conditions: BoundaryCondition,
    duration: float,
    fps: int,
    n_harmonics: int,
    base_frequency: float,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    ee_body_name: str,
    optimization_max_iter: int = 10,
    seed: int | None = None,  # この行を追加
) -> dict:
```

### 2.2. シード設定ロジックの追加

最適化の初期値 `initial_coeffs` をランダムに生成する直前に、渡された `seed` 引数を使ってNumPyの乱数シードを固定する処理を追加する。

**変更前の該当箇所:**
```python
    # 2. Optimize the coefficients of the excitation component
    initial_coeffs = np.random.rand(n_joints, n_harmonics, 2) * 0.1  # Start with small random amplitudes
    opt_args = (
        q_base_pos,
        # ...
```

**変更後の該当箇所:**
```python
    # 2. Optimize the coefficients of the excitation component
    if seed is not None:
        np.random.seed(seed)
    initial_coeffs = np.random.rand(n_joints, n_harmonics, 2) * 0.1  # Start with small random amplitudes
    opt_args = (
        q_base_pos,
        # ...
```

## 3. 期待される効果

- **再現性の確保**: `generate_exciting_spline_trajectory` 関数を呼び出す際に `seed` 引数を整数で指定することで、最適化のプロセスが固定され、常に同じ結果が得られるようになる。
- **後方互換性の維持**: `seed` 引数を指定しない場合（デフォルトは `None`）は、これまで通り実行ごとに異なるランダムな初期値が使われるため、既存のコードの挙動に影響を与えない。
