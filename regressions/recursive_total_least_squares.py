import json
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameData:
    """rtls_eval.json の1フレーム分に対応するデータ構造"""
    frame_id: str                        # "0000", "0001" などのキー
    wrench: list[float]                  # 6要素: 力・トルク測定値
    regressor: list[list[float]]         # 6x10要素: 回帰行列
    # RIV用データは今回無視しますが、将来的な拡張のためOptionalで残します
    regressor_instrument: list[list[float]] | None = None

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """NumPy計算用に変換 (y, A) を返す"""
        # 観測ベクトル y: (6, 1)
        y = np.array(self.wrench, dtype=np.float64).reshape(6, 1)

        # 回帰行列 A: (6, 10)
        A = np.array(self.regressor, dtype=np.float64)

        return y, A


def load_rtls_eval_json(filepath: str) -> list[FrameData]:
    """
    rtls_eval.json 形式のファイルを読み込み、フレームID順にソートして返す
    """
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    frames_dict = raw_data.get("frames", {})

    # フレームID ("0000", "0001"...) でソートしてリスト化
    sorted_keys = sorted(frames_dict.keys())

    data_list = []
    for key in sorted_keys:
        frame_content = frames_dict[key]

        # JSONの内容をdataclassにマッピング
        frame_data = FrameData(
            frame_id=key,
            wrench=frame_content["wrench"],
            regressor=frame_content["regressor"]
        )
        data_list.append(frame_data)

    return data_list


class InertialParameterEstimatorNP:
    def __init__(self, method="rtls", lambda_forgetting=0.99, initial_p=100.0, n_params=10):
        self.method = method
        self.n_params = n_params

        # 推定パラメータ phi: (10, 1)
        self.phi = np.zeros((self.n_params, 1), dtype=np.float64)

        # RLS用 共分散行列
        self.P = np.eye(self.n_params, dtype=np.float64) * initial_p
        self.lambda_f = lambda_forgetting

        # RTLS用 SVD状態 (11次元: パラメータ10 + 観測1)
        self.feature_dim = self.n_params + 1
        self.U = np.eye(self.feature_dim, dtype=np.float64)
        self.S = np.ones(self.feature_dim, dtype=np.float64) * 1e-6

    def update(self, A: np.ndarray, y: np.ndarray):
        """手法に応じた更新メソッドを呼び出す"""
        if self.method == "rls":
            self._step_rls(A, y)
        elif self.method == "rtls":
            self._step_rtls(A, y)

    def _step_rls(self, A, y):
        # RLS Algorithm
        PA_T = self.P @ A.T
        APA_T = A @ PA_T
        Lambda = np.eye(A.shape[0])
        S_inv = np.linalg.inv(Lambda + APA_T)
        K = PA_T @ S_inv

        error = y - A @ self.phi
        self.phi = self.phi + K @ error

        I = np.eye(self.n_params)
        self.P = (I - K @ A) @ self.P / self.lambda_f

    def _step_rtls(self, A, y):
        # Incremental SVD RTLS Algorithm
        # [A | -y] をデータとして追加
        Z_batch = np.hstack((A, -y))

        # 6行分のデータを1行ずつSVD更新
        for i in range(Z_batch.shape[0]):
            vector = Z_batch[i].reshape(-1, 1)
            self._update_svd_brand(vector)

        # 最小特異値に対応する右特異ベクトル(ここではUの最後の列で近似)から解を構成
        min_idx = np.argmin(self.S)
        v_smallest = self.U[:, min_idx]
        v_last = v_smallest[-1]

        if abs(v_last) > 1e-9:
            # phi = - v_{1:n} / v_{n+1}
            phi_val = - v_smallest[:-1] / v_last
            self.phi = phi_val.reshape(-1, 1)

    def _update_svd_brand(self, c):
        # Brand's Rank-1 SVD Update
        m = self.U.T @ c
        p = c - self.U @ m
        p_norm = np.linalg.norm(p)

        size = self.S.shape[0]
        K = np.zeros((size + 1, size + 1))
        np.fill_diagonal(K[:size, :size], self.S)
        K[:size, size] = m.flatten()
        K[size, size] = p_norm

        u_prime, s_prime, _ = np.linalg.svd(K)

        if p_norm > 1e-9:
            P_vec = p / p_norm
        else:
            P_vec = np.zeros_like(p)

        U_expanded = np.hstack((self.U, P_vec))
        self.U = U_expanded @ u_prime

        # 次元維持
        self.S = s_prime[:self.feature_dim]
        self.U = self.U[:, :self.feature_dim]

if __name__ == "__main__":
    # ファイルパス (適宜変更してください)
    json_path = "./rtls_eval.json"

    try:
        # データの読み込み
        frames = load_rtls_eval_json(json_path)
        print(f"Loaded {len(frames)} frames from {json_path}")

        # 推定器の初期化
        estimator_rtls = InertialParameterEstimatorNP(method="rtls")
        estimator_rls = InertialParameterEstimatorNP(method="rls") # 比較用

        # オンライン推定ループ
        print("\n--- Starting Online Estimation ---")
        print(f"{'Frame':<6} | {'Mass (RTLS)':<12} | {'Mass (RLS)':<12}")
        print("-" * 36)

        for i, frame in enumerate(frames):
            y, A = frame.to_numpy()

            # 推定ステップ
            estimator_rtls.update(A, y)
            estimator_rls.update(A, y)

            # ログ出力 (10フレームごと)
            if i % 10 == 0 or i == len(frames) - 1:
                mass_rtls = estimator_rtls.phi[0, 0] # パラメータの最初の要素が質量m
                mass_rls = estimator_rls.phi[0, 0]
                print(f"{frame.frame_id:<6} | {mass_rtls:12.4f} | {mass_rls:12.4f}")

        params = estimator_rtls.phi.flatten()
        if params[0] < 0:
            params = -1 * params

        print("\n--- Final Estimated Parameters (RTLS) ---")
        labels = ["m", "mc_x", "mc_y", "mc_z", "Ixx", "Iyy", "Izz", "Ixy", "Iyz", "Izx"]
        for name, val in zip(labels, params):
            print(f"{name:<5}: {val:.6f}")

    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
