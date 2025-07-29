# 励起軌道設計に関する議論のまとめ

## 論文の内容

本論文は、ロボットマニピュレータに取り付けられた剛体負荷の慣性パラメータをオンラインで推定するための再帰的トータル最小二乗（RTLS）アプローチを提案しています。従来の最小二乗法がデータ行列のノイズを無視するのに対し、RTLSはデータ行列（加速度、角速度）と力-トルク測定ベクトルの両方における誤差を明示的に考慮することで、よりロバストな推定を可能にします。

論文では、RTLSの理論的側面、推定方程式、そして特に**励起軌道設計**に焦点を当てています。また、提案手法の性能を評価するために、再帰的最小二乗（RLS）および再帰的インストゥルメンタル変数（RIV）手法との比較実験を行っています。実験結果は、RTLSが力制御や物体認識といったロボットアプリケーションにおいて、より高速かつ堅牢なオンライン推定性能を提供することを示しています。

## 励起軌道設計のコンセプト

論文で述べられている励起軌道設計プロセスは、慣性パラメータの正確かつロバストな推定を可能にするために、以下のコンセプトに基づいています。

1.  **基本正弦波軌道の生成**:
    *   各関節の軌道 `qi(t)` は、関節空間において、固定された数の重み付けされた正弦関数と余弦関数を重ね合わせることで構成されます（論文の式(10)および(11)）。
    *   これにより、パラメータ推定に必要な多様な動き（励起）が生成されます。

2.  **ジャーク制限多項式の適用**:
    *   軌道の開始部分と終了部分において、ジャーク（加速度の変化率）を制限するために、正弦波部分に適切な6次多項式 `f(t)` と `g(t)` を乗算します（論文の式(14)および(15)）。
    *   これらの多項式は、特定の境界条件（例：開始時の位置、速度、加速度、ジャークがゼロ）を満たすように設計され、機械構造への不要な励起を低減し、スムーズな動きを保証します。

3.  **最適化基準の定義**:
    *   推定のノイズとバイアス感度を評価するために、相関行列 `Y` の条件数 `κ(Y)` を最小化することを目的とします。
    *   `Y` は、全ての時間ステップにおけるデータ行列 `SA` を縦に結合したものの転置と、結合した `SA` の積として定義されます。条件数が小さいほど、推定のロバスト性が高まります。

4.  **最適化プロセスの実行**:
    *   まず、最適化に適した有望な候補軌道を見つけるためにモンテカルロ探索が実行されます。
    *   各候補軌道に対して、MATLABの最適化ツールボックスに含まれる`fmincon`関数に相当する最適化アルゴリズムが適用され、`κ(Y)` を最小化します。
    *   最適化の各イテレーションでは、環境的および動的な制約（例：関節の物理的な限界）が満たされているかどうかがチェックされます。

5.  **実験的検証の重視**:
    *   軌道の条件数は、関節角度のセットポイントに基づく計算だけでなく、実際のセンサーデータ（加速度センサーなど）に基づいて実験的に取得された条件数も考慮されます。論文では、センサーデータに基づく条件数の方が、関節角度のセットポイントに基づくものよりも高いことが示されており、実際の性能を考慮した軌道設計の重要性が強調されています。

---

## 実装戦略の検討と決定

論文では設計の数学的コンセプトは述べられているものの、具体的な実装手順（シミュレーションの利用有無など）は明記されていません。そこで、計算効率とシミュレーションの忠実度のトレードオフを考慮し、以下のハイブリッドアプローチを実装戦略として採用します。

### 設計思想: 計算効率と忠実度の両立

*   **課題**: 最適化ループの各イテレーションで完全な物理シミュレーションを実行すると、計算コストが膨大になり非現実的です。一方で、ノイズや誤差を無視した純粋な動力学計算だけでは、得られる軌道が現実環境でロバストであるか不明です。
*   **解決策**:
    1.  **最適化フェーズ**: 計算が高速な**純粋な動力学計算**に基づき、軌道の運動学的な「良さ」を示す条件数を最小化します。これにより、多数の候補の中から効率的に最適な軌道パラメータを探索します。
    2.  **検証フェーズ**: 最適化によって得られた**最終的な軌道**を、センサーノイズなどを含む**忠実な物理シミュレーション**で一度だけ実行します。これにより、理想環境で設計された軌道が、現実的な環境でも有効であるかを検証します。

このアプローチにより、開発の速度と得られる結果の信頼性を両立させます。

### 評価指標「条件数」の役割

条件数 `κ(Y)` は、回帰行列 `SA` の列ベクトル間の**線形独立性**を測る指標です。条件数が小さい軌道とは、慣性パラメータの各要素がロボットの動きに与える影響を、互いに分離・識別しやすくなるような多様な動きを持つ軌道を意味します。この運動学的な特性は、ノイズの有無よりも軌道そのものの形状に強く依存するため、純粋な動力学計算による評価は妥当かつ効果的です。

---

## 実装マイルストーン

上記の戦略に基づき、以下のステップで実装を進めます。各ステップはテストによって検証可能です。

### マイルストーン 1: 動力学計算のラッパー関数作成

*   **目的**: `trajectories`モジュールから、シミュレータ全体をインスタンス化することなく、条件数計算に必要な動力学計算を手軽に呼び出せるようにする。
*   **実装**: `dynamics`モジュールに、ロボットモデルと関節軌道を入力として条件数を返すラッパー関数 `calculate_condition_number(robot_model, joint_trajectory)` を作成します。この関数は内部で動力学パラメータのセットアップと回帰行列の計算をカプセル化します。
*   **テスト**: `tests/test_dynamics.py`にて、既知のモデルと軌道に対する条件数が正しく計算されることを確認します。

### マイルストーン 2: 最適化の目的関数実装

*   **目的**: `scipy.minimize`に渡すための目的関数を完成させる。
*   **実装**: `trajectories/optimal_excitation.py`に、最適化変数である軌道係数 `coeffs` を受け取り、マイルストーン1で作成したラッパー関数を呼び出して条件数を返す目的関数 `objective_function(coeffs, ...)` を作成します。
*   **テスト**: `tests/test_optimal_excitation.py`にて、異なる `coeffs` に対して目的関数が妥当な値を返すことを確認します。

### マイルストーン 3: 最適化ループの実装

*   **目的**: `scipy.minimize`を使って実際に軌道パラメータを最適化するメインの関数を実装する。
*   **実装**: `trajectories/optimal_excitation.py`のメイン関数内で、`coeffs`の初期値を設定し、`scipy.optimize.minimize`を呼び出して目的関数を最小化する最適化ループを実装します。
*   **テスト**: 最適化後の `coeffs` が初期値から変化し、かつ軌道の条件数が改善（低下）することを確認します。

### マイルストーン 4: 遷移軌道との統合

*   **目的**: 最適化された主軌道（励起部分）と、その前後をつなぐ滑らかな遷移軌道を結合する。
*   **実装**: 最適化された主軌道の開始・終了状態に基づき、`spline_interpolation.py`の機能を用いて遷移軌道を生成します。これら3つの軌道（遷移1, 主軌道, 遷移2）を連結し、一つの完全な軌道として出力するロジックを実装します。
*   **テスト**: 結合点において、位置・速度・加速度が滑らかに接続されていることを `np.testing.assert_allclose` などで検証します。

### マイルストーン 5: コマンドラインインターフェース (CLI) の最終化

*   **目的**: ユーザーがコマンドラインから全てのプロセスを実行できるようにする。
*   **実装**: `tyro`を用い、最適化のパラメータ（主軌道の時間、遷移時間、周波数など）を指定して、完全な励起軌道を生成・保存するCLIを構築します。
*   **手動テスト**: コマンドを実際に実行し、最終的な軌道データがJSONファイルとして正しく出力されることを確認します。

---

## マイルストーン達成記録

### マイルストーン 1: 動力学計算のラッパー関数作成

**目的**: `trajectories`モジュールから、シミュレータ全体をインスタンス化することなく、条件数計算に必要な動力学計算を手軽に呼び出せるようにする。

**達成内容**:

1.  **`dynamics/dynamics.py`の修正**: 
    *   `calculate_condition_number`関数を新規追加しました。この関数は、MuJoCoの`MjModel`と`MjData`オブジェクト、および関節軌道（位置、速度、加速度の時系列データ）を入力として受け取ります。
    *   関数内部では、`simulator/simulator.py`の`__init__`メソッドと`procoess_frame`メソッドのロジックを忠実に再現し、ロボットの動力学パラメータ（`uscrews_lj`, `simats_lj_l`, `hposes_lj_kj`など）をセットアップします。
    *   `functools.partial`を使用して`inverse`関数を部分適用し、軽量な逆動力学計算関数`inverse_dynamics`を作成しました。
    *   入力された関節軌道の各フレームに対して`inverse_dynamics`を呼び出し、エンドエフェクタのツイストとデツイストを計算します。
    *   計算されたツイストとデツイストを用いて`get_regressor_matrix`を呼び出し、各フレームの回帰行列を生成します。
    *   全ての回帰行列を縦に結合し、相関行列を計算した後、その条件数を`np.linalg.cond`で求め、結果として返します。
    *   当初ハードコードされていたエンドエフェクタのボディ名（`link6`）を`ee_body_name`引数として受け取れるように修正し、汎用性を高めました。

2.  **`tests/test_dynamics.py`の追加と修正**: 
    *   `unittest`フレームワークを使用したテストファイル`tests/test_dynamics.py`を新規作成しました。
    *   テストケース`test_calculate_condition_number`では、`mujoco`のシンプルなXMLモデルを動的に生成し、`spline_interpolation.py`で生成した軌道を用いて`calculate_condition_number`を呼び出します。
    *   返された条件数が`float`型であり、かつ正の値であることをアサートすることで、関数の基本的な動作を確認します。
    *   テストの実行時に、プロジェクトで実際に使用する`xml_models/manipulators/sequential/manipulator.xml`と`xml_models/targets/stanford-bunny`を結合したモデルをロードできるように、`tyro`を用いたコマンドライン引数解析を導入しました。
    *   `tyro`と`unittest`の併用における`sys.argv`の競合問題を解決するため、`tyro.cli`に`args`を明示的に渡し、`unittest.main`にも`argv`を渡すことで、両者の引数解析が干渉しないようにロジックを調整しました。
    *   これにより、`calculate_condition_number`が、プロジェクトの主要なマニピュレータとオブジェクトの組み合わせで正しく機能することが検証されました。

**検証結果**: 
*   `python3 tests/test_dynamics.py --manipulator_path xml_models/manipulators/sequential --object_path xml_models/targets/stanford-bunny` コマンドの実行により、テストが成功し、`calculate_condition_number`が`4.5042e+06`という条件数を返しました。この値は非常に大きいですが、これはテストに用いた単純な軌道が慣性パラメータを十分に励起できていないためであり、関数の正常な動作を示すものです。この大きな条件数を最小化することが、今後の最適化の目標となります。

### マイルストーン 2: 最適化の目的関数実装

**目的**: `scipy.optimize.minimize`に渡すための目的関数を完成させる。

**達成内容**:

1.  **`trajectories/optimal_excitation.py`の修正**: 
    *   `objective_function`関数を新規追加しました。この関数は、最適化の対象となる軌道係数`coeffs`と、軌道生成および条件数計算に必要な固定パラメータ（`m`, `d`, `duration`, `fps`, `jointpos_offset`, `base_frequency`, `ee_body_name`）を入力として受け取ります。
    *   関数内部では、まず`generate_sinusoidal_trajectory`を呼び出して、入力された`coeffs`に基づいた関節軌道（位置、速度、加速度の時系列データ）を生成します。
    *   生成された軌道データを`calculate_condition_number`が期待する形式に整形（`np.stack`と転置）した後、`dynamics.calculate_condition_number`を呼び出して回帰行列の条件数を計算します。
    *   計算された条件数を関数の戻り値として返します。この値が`scipy.optimize.minimize`が最小化しようとする「コスト」となります。
    *   `mujoco`モジュールと`dynamics.dynamics`モジュールからのインポートを追加しました。

2.  **`tests/test_optimal_excitation.py`の新規作成**: 
    *   `unittest`フレームワークを使用したテストファイル`tests/test_optimal_excitation.py`を新規作成しました。
    *   `setUpClass`メソッド内で、`test_dynamics.py`と同様に`tyro`を用いてコマンドライン引数を解析し、プロジェクトで実際に使用するマニピュレータとオブジェクトを結合したMuJoCoモデルをロードします。また、`objective_function`に渡す固定パラメータも設定します。
    *   テストケース`test_objective_function`では、ランダムな値を持つ`coeffs`と、全ての要素がゼロの`coeffs`の2種類のダミー係数配列を生成します。
    *   それぞれの`coeffs`に対して`objective_function`を呼び出し、返された条件数が`float`型であり、かつ正の値であることをアサートすることで、関数の基本的な動作を確認します。
    *   特に、ゼロ係数の場合はロボットが全く動かないため、条件数が無限大（`inf`）となることを確認し、`objective_function`が意味のない軌道に対しては非常に大きなコストを返すことを検証しました。

**検証結果**: 
*   `python3 tests/test_optimal_excitation.py --manipulator_path xml_models/manipulators/sequential --object_path xml_models/targets/stanford-bunny` コマンドの実行により、テストが成功しました。
*   ランダムな係数では`9.0862e+01`という条件数が、ゼロ係数では`inf`という条件数が返され、`objective_function`が期待通りに機能していることが確認されました。これは、最適化アルゴリズムが「より良い」軌道を見つけるための明確な評価基準が確立されたことを意味します。

### マイルストーン 3: 最適化ループの実装

**目的**: `scipy.minimize`を使って実際に軌道パラメータを最適化するメインの関数を実装する。

**達成内容**:

1.  **`trajectories/optimal_excitation.py`の修正**: 

    *   **変更点**: `generate_optimal_excitation_trajectory`関数を最適化ロジックを含むように大幅に修正しました。
    *   **追加インポート**: `from scipy.optimize import minimize`

    ```python
    # trajectories/optimal_excitation.py

    import numpy as np
    from numpy.typing import ArrayLike
    import mujoco
    from scipy.optimize import minimize # 追加

    from dynamics.dynamics import calculate_condition_number


    def generate_optimal_excitation_trajectory(
        duration: float,
        fps: int,
        n_harmonics: int, # 引数追加
        m: mujoco.MjModel, # 引数追加
        d: mujoco.MjData, # 引数追加
        base_frequency: float,
        jointpos_offset: ArrayLike = (0, 0, 0, 0, 0, 0),
        ee_body_name: str = "link6", # 引数追加
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates an optimal excitation trajectory by minimizing the condition number
        of the regressor matrix.
        """
        n_joints = m.njnt
        # 最適化の初期値 (coeffs): 小さなランダム値で初期化
        # 形状: (n_joints, n_harmonics, 2)
        initial_coeffs = np.random.rand(n_joints, n_harmonics, 2) * 0.01

        # objective_function に渡す固定引数
        objective_args = (
            m,
            d,
            duration,
            fps,
            jointpos_offset,
            base_frequency,
            ee_body_name,
        )

        # scipy.optimize.minimize は x0 (最適化変数) を1次元配列として期待するため、
        # initial_coeffs をフラット化します。
        initial_coeffs_flat = initial_coeffs.flatten()

        # objective_function は3次元配列の coeffs を期待するため、
        # 最適化器から渡される1次元配列を元の形状に戻すラッパー関数を定義します。
        def _objective_function_wrapper(coeffs_flat, *args):
            _m, _d, _duration, _fps, _jointpos_offset, _base_frequency, _ee_body_name = args
            _coeffs = coeffs_flat.reshape(n_joints, n_harmonics, 2) # ここで元の形状に戻す
            return objective_function(
                coeffs=_coeffs,
                m=_m,
                d=_d,
                duration=_duration,
                fps=_fps,
                jointpos_offset=_jointpos_offset,
                base_frequency=_base_frequency,
                ee_body_name=_ee_body_name,
            )

        # 最適化を実行
        result = minimize(
            fun=_objective_function_wrapper, # ラッパー関数を目的関数として指定
            x0=initial_coeffs_flat, # フラット化された初期値を渡す
            args=objective_args, # 固定引数を渡す
            method='Nelder-Mead', # 導関数不要のシンプルな最適化手法
            options={'maxiter': 100, 'disp': True} # テスト用にイテレーション回数を制限
        )

        # 最適化された係数も1次元配列で返されるため、元の形状に戻す
        optimized_coeffs = result.x.reshape(n_joints, n_harmonics, 2)

        # 最適化された係数で最終的な軌道を生成
        t_vec, qpos, qvel, qacc = generate_sinusoidal_trajectory(
            duration=duration,
            fps=fps,
            coeffs=optimized_coeffs, # 最適化された係数を使用
            base_frequency=base_frequency,
            jointpos_offset=jointpos_offset,
        )

        return t_vec, qpos, qvel, qacc

    # ... (generate_sinusoidal_trajectory 関数と objective_function 関数は省略) ...
    ```

    *   **実装内容**: 
        *   `generate_optimal_excitation_trajectory`関数は、最適化の対象となる`coeffs`の初期値をランダムに生成します。
        *   `scipy.optimize.minimize`は`x0`（最適化される初期値）を1次元配列として期待するため、`initial_coeffs`を`flatten()`して`initial_coeffs_flat`を作成しました。
        *   `objective_function`が3次元配列の`coeffs`を期待するため、`_objective_function_wrapper`という内部ラッパー関数を定義しました。このラッパー関数内で、`minimize`から渡される1次元配列を元の形状に`reshape`してから`objective_function`に渡すようにしました。
        *   `minimize`関数を呼び出し、`_objective_function_wrapper`を目的関数として、`initial_coeffs_flat`を初期値として渡します。`method='Nelder-Mead'`を選択し、`options={'maxiter': 100, 'disp': True}`でテスト用にイテレーション回数を制限しました。
        *   最適化結果`result.x`もフラット化されているため、`optimized_coeffs`として使用する前に元の形状に`reshape()`し直しました。
        *   最終的に、最適化された`optimized_coeffs`を用いて`generate_sinusoidal_trajectory`を呼び出し、最適化された軌道を生成して返します。

    *   **最適化対象パラメータ**: 論文の記述通り、`generate_sinusoidal_trajectory`の`coeffs`引数（`p_ik`と`d_ik`）が最適化の対象となります。これは、`n_joints * n_harmonics * 2`個の要素を持つ1次元配列として`minimize`に渡されます。

    *   **最適化ループの停止基準**: 現在の最適化ループの主な停止基準は、`minimize`関数の`options`引数で設定された`'maxiter': 100`です。これは、目的関数の評価回数が100回に達すると最適化が停止することを意味します。これはテストの実行時間を短縮するための設定であり、実際の最適化ではより多くのイテレーションが必要となる場合があります。`scipy.optimize.minimize`のNelder-Mead法には、デフォルトで`xatol`（最適化変数の変化の許容誤差）や`fatol`（目的関数の変化の許容誤差）といった停止基準も組み込まれていますが、現在の実装では明示的に上書きされていません。

2.  **`tests/test_optimal_excitation.py`の修正**: 

    *   **変更点**: `test_generate_optimal_excitation_trajectory`テストケースを追加しました。
    *   **追加インポート**: `from dynamics.dynamics import calculate_condition_number`

    ```python
    # tests/test_optimal_excitation.py

    # ... (既存のインポート、TestConfig、_test_config、TestOptimalExcitationクラスのsetUpClass、test_objective_function) ...

    def test_generate_optimal_excitation_trajectory(self):
        print("\nTesting generate_optimal_excitation_trajectory (optimization loop)...")
        # 比較のために、最適化前の初期条件数を計算
        coeffs_shape = (self.n_dof, self.n_harmonics, 2)
        initial_coeffs = np.random.rand(*coeffs_shape) * 0.01
        initial_cond_num = objective_function(
            coeffs=initial_coeffs,
            m=self.m,
            d=self.d,
            duration=self.duration,
            fps=self.fps,
            jointpos_offset=self.jointpos_offset,
            base_frequency=self.base_frequency,
            ee_body_name=self.ee_body_name,
        )
        print(f"  Initial Condition Number: {initial_cond_num:.4e}")

        # 最適化を実行
        t_vec, qpos, qvel, qacc = generate_optimal_excitation_trajectory(
            duration=self.duration,
            fps=self.fps,
            n_harmonics=self.n_harmonics,
            m=self.m,
            d=self.d,
            base_frequency=self.base_frequency,
            jointpos_offset=self.jointpos_offset,
            ee_body_name=self.ee_body_name,
        )

        # 最適化された軌道から条件数を再計算
        optimized_trajectory = np.stack([qpos.T, qvel.T, qacc.T], axis=1)
        optimized_cond_num = calculate_condition_number(
            m=self.m,
            d=self.d,
            joint_trajectory=optimized_trajectory,
            ee_body_name=self.ee_body_name,
        )
        print(f"  Optimized Condition Number: {optimized_cond_num:.4e}")

        # アサーション: 最適化された条件数が初期条件数よりも小さいことを確認
        self.assertLess(optimized_cond_num, initial_cond_num, 
                        "Optimized condition number should be less than initial.")
        # 返り値の型が正しいことを確認
        self.assertIsInstance(t_vec, np.ndarray)
        self.assertIsInstance(qpos, np.ndarray)
        self.assertIsInstance(qvel, np.ndarray)
        self.assertIsInstance(qacc, np.ndarray)

    # ... (if __name__ == '__main__': ブロックは省略) ...
    ```

    *   **実装内容**: 
        *   `test_generate_optimal_excitation_trajectory`テストケースを追加しました。
        *   このテストでは、`generate_optimal_excitation_trajectory`を呼び出す前に、ランダムな初期係数で`objective_function`を一度呼び出し、その条件数`initial_cond_num`を記録します。これにより、最適化の改善度を評価するためのベースラインを設定します。
        *   `generate_optimal_excitation_trajectory`を実行し、最適化された軌道を取得します。
        *   取得した最適化済み軌道を用いて、再度`calculate_condition_number`を呼び出し、`optimized_cond_num`を計算します。
        *   最も重要なアサーションとして、`self.assertLess(optimized_cond_num, initial_cond_num)`を用いて、最適化された軌道の条件数が、初期のランダムな軌道の条件数よりも**小さくなっていること**を確認します。これにより、最適化が実際に機能していることを検証します。
        *   また、返された軌道データの型が`np.ndarray`であることを確認するアサーションも追加しました。

**検証結果**: 
*   `python3 tests/test_optimal_excitation.py --manipulator_path xml_models/manipulators/sequential --object_path xml_models/targets/stanford-bunny` コマンドの実行により、テストが成功しました。
*   最適化後の条件数（例: `7.3578e+01`）が、初期の条件数（例: `1.1083e+02`）よりも小さくなっていることを確認しました。これは、`scipy.optimize.minimize`が目的関数を最小化しようと正しく機能していることを示しています。