import numpy as np

# --- 1. 真の慣性パラメータとシミュレーション条件の定義 ---

# 求めたい「真の」慣性パラメータ (ベクトル φ) を仮定
phi_true = np.array([1.5, 0.01, 0.02, -0.015, 0.01, 0.001, -0.002, 0.012, 0.003, 0.015])

# データ収集のサンプル数
num_samples = 200
np.random.seed(0)

# --- 2. 測定データのシミュレーション (Aとwの両方にノイズ) ---

# --- ノイズのない理想的な A と w を生成 ---
# 理想的な観測行列 A
A_true = np.random.randn(6 * num_samples, 10)
# 理想的なレンチ w
w_true = A_true @ phi_true

# --- ノイズを追加して「測定値」をシミュレート ---
noise_level_A = 0.01  # A に加えるノイズのレベル
noise_level_w = 0.01  # w に加えるノイズのレベル

# ノイズを含んだ観測行列 A_measured を作成
A_measured = A_true + np.random.normal(0, noise_level_A, A_true.shape)
# ノイズを含んだレンチ w_measured を作成
w_measured = w_true + np.random.normal(0, noise_level_w, w_true.shape)


# --- 3. OLSによるパラメータ推定 (比較用) ---
# OLSは A_measured にノイズがあることを考慮できないが、比較のために計算する
phi_ols, _, _, _ = np.linalg.lstsq(A_measured, w_measured, rcond=None)


# --- 4. TLSによるパラメータ推定 ---

# === ステップ 4-1: 拡張行列の作成 ===
# TLSでは、全ての測定変数 (Aとw) を一つの行列にまとめる
# C = [A_measured | w_measured]
# w_measured を (N, 1) の2次元配列に変形して結合する
C_tls = np.hstack([A_measured, w_measured.reshape(-1, 1)])  # (n, 10), (n, 1) -> (n t, 11)

# === ステップ 4-2: 特異値分解 (SVD) の実行 ===
# 拡張行列CをSVDにかけることで、データの構造を明らかにする
# C = U * S * Vt
U, S, Vt = np.linalg.svd(C_tls)

# === ステップ 4-3: 解となるベクトルの抽出 ===
# TLSの解は、Vt の最後の行ベクトル（最小特異値に対応する右特異ベクトル）に含まれる
# このベクトルが、拡張されたデータ空間において最もばらつきの小さい方向を示しており、
# それが求める線形関係式に対応する
v_min = Vt[-1, :]

# === ステップ 4-4: パラメータへの変換 ===
# v_min は [φ; -1] に比例するベクトル。
# 最後の要素で全体を割り、符号を反転させることで φ を取り出す
last_element = v_min[-1]
phi_tls = -(1 / last_element) * v_min[:-1]


# --- 5. 結果の表示と比較 ---
print("--- Parameter Estimation Comparison (OLS vs. TLS) ---")
print("Noise added to both A and w.")
print("-" * 50)
print(f"{'Parameter':<10s} | {'True Value':>12s} | {'OLS Est.':>12s} | {'TLS Est.':>12s}")
print("-" * 50)

param_names = ["m", "mc_x", "mc_y", "mc_z", "I_xx", "I_xy", "I_xz", "I_yy", "I_yz", "I_zz"]
for i in range(len(phi_true)):
    print(f"{param_names[i]:<10s} | {phi_true[i]:12.4f} | {phi_ols[i]:12.4f} | {phi_tls[i]:12.4f}")

# 推定精度を評価
error_ols = np.linalg.norm(phi_true - phi_ols)
error_tls = np.linalg.norm(phi_true - phi_tls)
print("-" * 50)
print(f"OLS Estimation Error (Norm): {error_ols:.6f}")
print(f"TLS Estimation Error (Norm): {error_tls:.6f}")

if error_tls < error_ols:
    print("\n✅ As expected, TLS produced a more accurate estimate when both A and w are noisy.")
else:
    print("\nIn this specific run, OLS performed similarly or better.")
