---
Title: Inertial Parameter Identification in Robotics: A Survey
Authors:
  - Leboutet, Quentin
  - Roux, Julien
  - Janot, Alexandre
  - Guadarrama-Olvera, Julio Rogelio
  - Cheng, Gordon
Year: 2021
Venue: Applied Sciences
Tags:
  - "inertial-parameter-identification"
  - "robot-dynamics"
  - "benchmark"
  - "least-squares"
  - "physical-consistency"
  - "survey"
PDF: "[[papers/Leboutet-ApplSci2021-Inertial_Parameter_Identification/main.pdf|📃]]"
Import Date: "2026-07-15"
Read Date: 2026-07-15
Executive Summary: 6-DoF 産業用マニピュレータの慣性パラメータ同定手法を体系的にレビューし, 17 の代表的アルゴリズム (IDIM-OLS/WLS/IRLS/TLS, IDIM-IV, ML, DIDIM, CLOE, CLIE, DDIM-NKF, AdaNN, HTRNN, および LMI/SDP による PC-系) を単一の Matlab ツールボックス BIRDy 上で実装する. Staubli TX40 と Mitsubishi RV2SQ について Monte Carlo シミュレーション (各手法 M=1300 回) と実機検証を実施し, ノイズ耐性・推定精度・収束性・計算コストで比較する. DIDIM と IDIM-IV が最も良好な精度と収束速度を示し, IDIM-OLS/-WLS 系はデータフィルタリングとの併用で他手法と同等になるとの結論を得た. ただし excitation trajectory 設計は Fourier 級数と cond+σ_min コストによる短い 1 節 (§8.2) のみで扱い, task 追従下の drift 対応は議論外にとどまる.
Citekey: Leboutet-ApplSci2021-Inertial_Parameter_Identification
BibTeX Key: leboutet2021inertial
DOI: 10.3390/app11094303
Relevance: 4
Repository: https://github.com/TUM-ICS/BIRDy
Category: note
Template Version: v2.3
---

## Executive Summary
6-DoF 産業用マニピュレータの慣性パラメータ同定手法を体系的にレビューし, 17 の代表的アルゴリズム (IDIM-OLS/WLS/IRLS/TLS, IDIM-IV, ML, DIDIM, CLOE, CLIE, DDIM-NKF, AdaNN, HTRNN, および LMI/SDP による PC-系) を単一の Matlab ツールボックス BIRDy 上で実装する.
Staubli TX40 と Mitsubishi RV2SQ について Monte Carlo シミュレーション (各手法 M=1300 回) と実機検証を実施し, ノイズ耐性・推定精度・収束性・計算コストで比較する.
DIDIM と IDIM-IV が最も良好な精度と収束速度を示し, IDIM-OLS/-WLS 系はデータフィルタリングとの併用で他手法と同等になるとの結論を得た.
ただし excitation trajectory 設計は Fourier 級数と cond+σ_min コストによる短い 1 節 (§8.2) のみで扱い, task 追従下の drift 対応は議論外にとどまる.

---
## Summary

### この論文が答えた問い, あるいは解決した課題は何か？

固定ベース剛体シリアルマニピュレータの**慣性パラメータオフライン同定**について, これまで文献に分散していた多数のアルゴリズム (Least-Squares 系, Output/Input Error 系, Kalman filter 系, Neural Network 系, Physically Consistent 系) が「どういう条件でどれを使うべきか」を quantitative な argument で判断できる guideline は存在しなかった (§1.2).
著者らは (a) すべての手法を同一のフレームワーク下で実装した open-source Matlab toolbox **BIRDy** (Benchmark for Identification of Robot Dynamics) を提供し, (b) 6-DoF Staubli TX40 と Mitsubishi RV2SQ の Monte Carlo シミュレーションと実機実験によって手法間の関係と選択指針を確立することを目的とする (§1.2).

### 提案手法のアプローチと, その根幹をなす要素は何か？

BIRDy は「symbolic モデル生成 → excitation trajectory 生成 → 実験データ収集/前処理 → 同定アルゴリズム適用 → 後処理・比較」という 5 段階パイプラインを単一の Matlab toolbox として実装する (Figure 6, §8).
17 手法を同一の観測データに適用し, Monte Carlo で M=25·b=1300 runs (b は base parameter 数) を回して statistical に評価する.

- **統一 Inverse Dynamic Identification Model (IDIM) の regressor $Y_\beta$**: Euler-Lagrange 定式化で $\tau = Y_\chi(\ddot q, \dot q, q) \chi$ を組み, QR 分解で base parameter $\beta = \overline P^\top \chi$ に縮約する (§2, §8.1).
- **17 手法の実装**: IDIM-OLS/-WLS/-IRLS/-TLS (§3), IDIM-IV, IDIM-ML (§4), CLOE, CLIE, DIDIM (§5), EKF/UKF/CDKF/SREKF/SRUKF/SRCDKF/PF から構成される DDIM-NKF, AdaNN, HTRNN (§6), そして LMI を SDP で解く PC-IDIM-OLS/-WLS/-IRLS, PC-IDIM-IV, PC-DIDIM (§7).
- **Excitation trajectory の Fourier 級数パラメトリゼーション**: cost $J_t = k_1 \cdot \mathrm{cond}(W^\top W) + k_2/\sigma_{\min}$ を fmincon および ga で 10 秒の time horizon 上で最小化 ($k_1=1, k_2=100$) (Eq. 67, §8.2, §9.1).
- **Physical consistency の LMI/SDP 表現**: Huygens-Steiner theorem を用いて (56) を Schur complement 形式の $D_j(\chi) \succ 0$ に書き換え (Eq. 59), density realizability 制約 (60) を含む $D'_{Lj}(\chi) \succ 0$ を CVX+MOSEK で解く (§7.1, §9.3.9).
- **Figure of Merit (FOM)**: 平均相対角度差 $d_q$, 平均相対トルク差 $d_\tau$, 計算時間 $d_t$, 反復数 $d_{N_{it}}$, モデルシミュレーション数 $d_{N_{sim}}$ の 5 指標 (§9.2).

### 特に参考とした既存研究と, それらと比した提案手法の新規性は何か？

Gautier ら [[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] 以降の IDIM 系 (ref [2, 3, 6, 43, 52]), Janot らによる CLOE/DIDIM/IV の比較 (ref [5, 6, 12]), Sousa-Cortesão [20, 21] や Wensing [[papers/Wensing-RAL2017-LMIPhysicalConsistency/linear-matrix-inequalities-for-physically-consistent-inertial-parameter-identification-a-statistical-perspective-on-the-mass-distribution|Wensing+ 2017]] (ref [22]) の LMI/SDP による physical consistency, Urrea-Pascal (ref [13, 14]) による IDIM-OLS/AdaNN/HTRNN/EKF/GA の 5DoF SCARA 上での比較, Wu ら [29] の同定 survey, および TLS 系 (ref [[papers/Markovsky-SigProc2007-Overview_Total_Least-Squares/overview-of-total-least-squares-methods|Markovsky+ 2007]] [63], [54-56]) が主要な基盤である.
新規性は次の 3 点である (§1.2 最終段落).
第一に **17 手法を同一の robot model・同一の Monte Carlo 条件下で比較**した規模の benchmark はこれまで存在しなかった (ref [29] は survey だが benchmark は行っていない; ref [13, 14] は SCARA で 5 手法のみ).
第二に **手法間の conceptual relationship を確立**した (例: DIDIM は CLIE の Jacobian を近似したもの, IDIM-ML は IV の instrument matrix を $Z=W_{nf}$ 相当で構成することに対応, HTRNN は OLS 推定に漸近する).
第三に non-expert が使える **decision tree** (Figure 11) を提供する: 制御構造の既知性・エンコーダ分解能・サンプリングレートに応じて DIDIM/CLIE/CLOE, IDIM-IV, IDIM-ML/DDIM-NKF, IDIM-OLS/-WLS/-IRLS/AdaNN/HTRNN のいずれを選ぶかを分岐する.

### どのように訓練・最適化したのか？

- **損失関数 / 最適化目的**:
  - IDIM-WLS: $\hat\beta_{WLS} = (W^\top \Sigma^{-1} W)^{-1} W^\top \Sigma^{-1} y_\tau$ (Eq. 20).
  - IDIM-IV: $\hat\beta_{IV}^i = (W_s^{i\top} \Sigma^{-1} W)^{-1} W_s^{i\top} \Sigma^{-1} y_\tau$ を反復 (Eq. 34).
  - CLOE: $J(\beta) = \|y - y_s\|_2^2$ を Levenberg-Marquardt で最小化 (Eq. 37, §9.3.5).
  - DIDIM: CLIE の Jacobian を $G_{\tau_s} \approx Y_\beta(\ddot q_s, \dot q_s, q_s)$ で近似し, 各反復で $\hat\beta_{DIDIM}^i = (W_s^{i\top} \Sigma^{-1} W_s^i)^{-1} W_s^{i\top} \Sigma^{-1} y_\tau$ を解く (Eq. 45).
  - PC-IDIM-WLS: $\min_{\beta, \underline\chi} (W\beta - y_\tau)^\top \Sigma^{-1} (W\beta - y_\tau)$ s.t. $\chi = G^{-1}[\beta^\top \; \underline\chi^\top]^\top$, $D_j(\chi) \succ 0$ (Eq. 62). CVX+MOSEK で SDP を解く.
  - Excitation trajectory: $J_t = k_1 \cdot \mathrm{cond}(W^\top W) + k_2/\sigma_{\min}$ を fmincon/ga で $k_1=1, k_2=100$, 10 s horizon で最小化 (Eq. 67).
- **データセット**:
  - 反射的な意味での訓練データセットは無い. 各 Monte Carlo run で **excitation trajectory 上を 10 秒間 tracking したシミュレーションデータ** ($\ddot q, \dot q, q, \tau$ の時系列) を生成し, これを observation matrix $W$ にスタックして同定する.
  - **サンプリング条件**: TX40 では制御周波数 $f_c = 5$ kHz, サンプル $f \in \{500, 100\}$ Hz. RV2SQ は $f = 140$ Hz.
  - **ノイズ条件**: 5 レベル ($\sigma_q \in \{10^{-4}, 10^{-3}, 10^{-2}\}$ rad, $\sigma_\tau \in \{5\cdot 10^{-2}, 10^{-1}\}$ N·m) を組み合わせ, Table 1 に示す 15 条件で MCS.
  - **Run 数**: 各手法・各 run で 25 個の初期パラメータベクトル (CAD 値 ×15% 相対誤差) から初期化, $M = 25 \cdot b = 1300$ runs (b=52 は TX40 base parameter 数).

### どのように検証したか？指標と結果は？

**検証プロトコル**: (a) TX40 と RV2SQ の DDM を用いた MCS (15 実験条件 × 17 手法 × 1300 runs), (b) 実機 TX40 と実機 RV2SQ 上での single-run validation (base parameter 数は TX40 で 54 に増加, joints 5-6 の coupling を含む).
FOM は $d_q, d_\tau, d_t, d_{N_{it}}, d_{N_{sim}}$ の平均と標準偏差 (§9.2).

**主要結果** (§10.1):

- **Noise Immunity**: AdaNN, HTRNN, ML, IDIM-OLS/-WLS/-IRLS/-TLS は $\sigma_q = 10^{-4}$ rad より高いノイズで大きなバイアスを示す. しかし Butterworth zero-shift filter (50 Hz 帯域) の適用で W → W_{nf} に近づき, 結果が大幅に改善される. DIDIM, CLIE, CLOE は DDM/IDM シミュレーションを介するため joint position noise に本質的にロバスト (§10.1.1).
- **Estimation Accuracy**: **DIDIM と IDIM-IV が最も高い精度**を示す (MCS-RV2SQ-1-1, MCS-TX40-1-1, MCS-TX40-4-1 等). ただし real RV2SQ 上では DIDIM と IDIM-IV は制御法則を well known とする前提が崩れるため MCS ほど良好ではない. AdaNN は理論的には IDIM-OLS と漸近等価だが分散が **一桁以上大きい**. NKF は un-decimated TX40 データで良好だが, decimated data では急激に精度悪化 (§10.1.2).
- **Convergence and Computational Complexity**: DIDIM の平均計算時間は約 1 秒, IDIM-OLS の 5 倍程度 (DDM シミュレーションを含むため). PC-系 SDP は unconstrained の約 3 倍. CLIE/CLOE は Jacobian を有限差分で求めるため DIDIM/IDIM-IV の約一桁遅い. NKF (sigma-point 系) は 12 次元状態ベクトル×52 base parameter で 129 sigma-point を伝播させる必要があり最も高価 (§10.1.3).

### 検証結果に基づいた議論, 明らかになった課題はあるか？

著者は §10.2 で以下を明示的に議論する.

- 最適な手法選択は「how これらのパラメータが同定されるか」だけでなく「why それらを同定するか」にも依存する. black-and-white な答えは存在しない (§10.2 冒頭).
- **DIDIM/IDIM-IV は low-level controller の precise knowledge を必要**とする. これが崩れると同定 process にバイアスまたは divergence が生じうる. 実機で MCS 通りの精度が出ないのはこのため (§10.2).
- IDIM-OLS/-WLS/-TLS/ML/AdaNN/HTRNN は初期推定や tuning 係数を要さないため実用が容易だが精度は劣る (§10.2).
- NKF, IDIM-OLS/-WLS は recursive に書けるため adaptive control loop への実装が可能. Kalman filter の主な欠点は process/measurement covariance の tuning 感度が極端に高いこと (§10.2).
- サンプリングレートが 100 Hz 未満では joint velocity/acceleration の計算が unreliable になり, DDM シミュレーションを介する IDIM-IV や DIDIM を優先すべき (§10.2 末尾).
- **Future work**: (a) より詳細な noise 振幅・分布 (量子化誤差, センサ分解能低下) の study, (b) MuPad symbolic kernel が 7-DoF 超で遅い問題を SymPy/Mathematica や URDF/RBDL/KDL への移行で解決, (c) parallel robot (Stewart platform) や floating-base humanoid への拡張, (d) accelerometer/gyroscope/force sensor の追加モダリティ統合 (§10.2 末尾, §11).

---
## 自身の研究との関連

本論文は 6-DoF sequential manipulator の慣性パラメータ同定という本プロジェクトの中核領域と直接重なる survey であり, 手法選択の landscape を最新かつ包括的に整理する上での reference point として位置づけられる.
特に IDIM-OLS/-WLS/-TLS/-IV, DIDIM, および PC-系 SDP の関係を明示化した点は, 本プロジェクトが hammer 慣性 10 パラメータ同定でどの baseline を採用すべきかの議論を支える.

一方で本プロジェクトの中心課題である **task-required で non-optimized な base drift による regressor conditioning 悪化**は本論文の対象外である.
Section 8.2 (excitation trajectory design) は 41 ページの survey 全体で 1 節 (約 1 ページ) にとどまり, cost $J_t = k_1 \cdot \mathrm{cond}(W^\top W) + k_2/\sigma_{\min}$ の Fourier 級数最小化を提示するのみで, task 追従下での drift 対応や cond 悪化の緩和策には言及がない.
これは本プロジェクトが埋める gap の positioning を強化する材料になる: 41 ページの comprehensive survey ですら excitation を短い 1 節しか扱っていない.

未読で追跡が必要な文献:
- **[101] Abu-Dakka & Díaz-Rodríguez 2017 (IROS)**: 統計的分析付きの trajectory parametrization 比較. 本プロジェクトの excitation 設計比較の先行研究として重要.
- **[102] Ayusawa+ 2017 (ICRA)**: condition number optimization による persistently exciting trajectory 生成. 本プロジェクトの cond 最小化アプローチと直接対応する.
- **[38] Gautier & Khalil 1992 (IJRR)**: exciting trajectory の古典的原典.
- **[92] Janot & Wensing 2021 (CEP)**: PC-DIDIM の原論文.

本論文の BIRDy toolbox (https://github.com/TUM-ICS/BIRDy) は open-source であり, 本プロジェクトの baseline 実装として直接参照できる可能性がある.

---
## 追加議論


---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@article{leboutet2021inertial,
  title   = {Inertial Parameter Identification in Robotics: A Survey},
  author  = {Leboutet, Quentin and Roux, Julien and Janot, Alexandre and Guadarrama-Olvera, Julio Rogelio and Cheng, Gordon},
  journal = {Applied Sciences},
  volume  = {11},
  number  = {9},
  pages   = {4303},
  year    = {2021},
  doi     = {10.3390/app11094303},
  publisher = {MDPI}
}
```
</details>
