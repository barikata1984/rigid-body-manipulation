# 動力学モジュールのリファクタリング

## 概要
本リファクタリングは、`dynamics/dynamics.py` と `simulator/simulator.py` 間で重複していたロボットの動的計算ロジックを解消し、コードのモジュール性と保守性を向上させることを目的として実施されました。特に、シミュレータの初期化フェーズと各フレーム処理フェーズにおける動的パラメータのセットアップおよび計算ロジックに重複が見られました。

## 変更内容

### 1. `dynamics/dynamics.py` へのヘルパー関数の導入
以下の2つのヘルパー関数を `dynamics/dynamics.py` に追加しました。

-   `_setup_robot_dynamics_parameters(m, d, ee_body_name)`:
    -   **目的**: ロボットの動的計算に必要な初期パラメータ（ユニバーサルスクリュー `uscrews_lj`、空間慣性行列 `simats_lj_l`、ホームポーズ `hposes_lj_kj`、および部分適用された逆動力学関数 `inverse_dynamics` など）のセットアップロジックをカプセル化します。
    -   **詳細**: `Simulator.__init__` と `calculate_condition_number` の両方で共通して行われていた、これらのパラメータの計算と初期化処理をこの関数に集約しました。計算されたパラメータはタプルとして返されます。

-   `_calculate_frame_dynamics(act_traj, inverse_dynamics_partial_func, id_ll, pose_x_ll, pose_ll_llj, pose_x_sen)`:
    -   **目的**: シミュレーションの各フレームにおける動的計算ロジックをカプセル化します。
    -   **詳細**: 関節軌道 `act_traj` と初期化された動的パラメータに基づいて、センサーフレームでのツイスト `twist_sen`、ツイストの微分 `dtwist_sen`、およびレグレッサ行列 `regressor` を計算します。これらの計算結果はタプルとして返されます。

### 2. `dynamics/dynamics.py` の `calculate_condition_number` の修正
-   `_setup_robot_dynamics_parameters` を呼び出して初期パラメータを取得し、ループ内で `_calculate_frame_dynamics` を呼び出してレグレッサ行列を計算するように変更しました。これにより、コードの重複が解消されました。

### 3. `simulator/simulator.py` の `Simulator` クラスの修正

-   **`__init__` メソッド**:
    -   `_setup_robot_dynamics_parameters` を呼び出すように変更し、返された動的パラメータを `self` の属性に割り当てました。これにより、重複していた初期化ロジックが置き換えられました。

-   **`procoess_frame` メソッド**:
    -   `_calculate_frame_dynamics` を呼び出すように変更し、フレームごとの動的計算を実行するようにしました。
    -   また、`dynamics` モジュールからの明示的なインポート (`from dynamics.dynamics import ...`) に変更したため、`dyn.` プレフィックスが残っていた箇所を修正しました。

## リファクタリングの利点

-   **コードの重複排除**: ロボットの動的計算に関するセットアップとフレームごとのロジックが `dynamics.py` に集約され、一貫性が保たれました。
-   **モジュール性の向上**: 動的計算ロジックが独立したヘルパー関数として定義されたことで、各モジュールの責任が明確になりました。
-   **保守性の向上**: 動的計算ロジックの変更が必要になった場合、`dynamics.py` 内のヘルパー関数のみを修正すればよくなり、変更の影響範囲が限定されます。
-   **後方互換性の維持**: `Simulator` クラスのパブリックなインターフェースは変更されていないため、既存のコードベースとの互換性が維持されています。

## 検証
リファクタリング後、以下のコマンドでシミュレーション実行とテストが正常に完了することを確認しました。

-   シミュレーション実行: `python main.py --object xml_models/targets/chair --target-trajectory experiment_setups/trajectories/spline.json`
-   テスト実行: `pytest`

これにより、リファクタリングされたコードが期待通りに機能していることが確認されました。
