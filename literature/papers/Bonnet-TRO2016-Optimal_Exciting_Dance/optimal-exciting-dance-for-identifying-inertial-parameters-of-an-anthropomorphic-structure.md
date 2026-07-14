---
Title: Optimal Exciting Dance for Identifying Inertial Parameters of an Anthropomorphic Structure
Authors:
  - Bonnet, Vincent
  - Fraisse, Philippe
  - Crosnier, André
  - Gautier, Maxime
  - González, Alejandro
  - Venture, Gentiane
Year: 2016
Venue: TRO
Tags:
  - "inertial-parameter-identification"
  - "humanoid"
  - "exciting-trajectory"
  - "b-spline"
  - "zmp-balance"
  - "constrained-qp"
Ref: "[[references/main.md#Bonnet2016_dance]]"
PDF: "[[papers/Bonnet-TRO2016-Optimal_Exciting_Dance/main.pdf|📃]]"
Import Date: "2026-07-15"
Read Date: 2026-07-15
Executive Summary: HOAP-3 と人被験者の全身標準慣性パラメータ (SIP) を同定するため, 関節を励起用 q_DE とバランス用 q_DB に分解する多段最適化を提案する. 6 次 B-spline で表現した動作に対し, per-link mass-weighted condition number の和 J_exc = Σ M_i · cond(W_bi) を新規励起評価基準として定式化し, 静的姿勢・姿勢間動的遷移・ZMP バランス調整を階層的な非線形計画で解く. 同定は物理整合性 (質量正・CoM が bounding box 内・慣性行列半正定) を課した制約付き QP で行い, 腕への 0.5 kg 追加質量を検出できる精度が得られた.
Citekey: Bonnet-TRO2016-Optimal_Exciting_Dance
BibTeX Key: bonnet2016optimal
DOI: 10.1109/TRO.2016.2583062
Relevance: 4
Repository: none
Category: note
Template Version: v2.3
---

## Executive Summary
HOAP-3 と人被験者の全身標準慣性パラメータ (SIP) を同定するため, 関節を励起用 q_DE とバランス用 q_DB に分解する多段最適化を提案する.
6 次 B-spline で表現した動作に対し, per-link mass-weighted condition number の和 J_exc = Σ M_i · cond(W_bi) を新規励起評価基準として定式化し, 静的姿勢・姿勢間動的遷移・ZMP バランス調整を階層的な非線形計画で解く.
同定は物理整合性 (質量正・CoM が bounding box 内・慣性行列半正定) を課した制約付き QP で行い, 腕への 0.5 kg 追加質量を検出できる精度が得られた.

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
人型ロボットや人被験者のような**浮動ベース多体系**では, 関節トルクを直接測定できず, ルートリンクに作用する外部レンチ (床反力) のみが利用可能である.
このとき, SIP 同定用の**励起運動を自動生成する**には, (a) 20 以上の DoF, (b) 動的バランス (ZMP が支持多角形内) と機械的制約 (関節角・トルク・自己衝突), (c) 大きいリンク (体幹) と小さいリンク (手足) の質量比が回帰行列 W を著しく悪条件にする, という 3 つの困難が同時に立ちはだかる.
本論文は, これらを結合した非線形計画を単一で解こうとすると初期値依存の収束不良に陥る点を回避する分解定式化と, 質量差を吸収する新しい励起基準を提示する.

### 提案手法のアプローチと、その根幹をなす要素は何か？
全身動作を励起サブセット q_DE (質量・慣性を励起する関節) とバランスサブセット q_DB (ZMP を支持多角形内に保つ関節) に分割し (Table I), 4 つの足配置ごとに N_p=60 の静的姿勢と姿勢間動的遷移を階層的に最適化して "optimal exciting dance" を生成する.
同定側は base parameter Φ_b を最小二乗で先に求め, その後で標準慣性パラメータ Φ を物理整合性制約下の QP で回復する二段構成.

- **DoF 分解 (q_DE / q_DB)**: 足配置に応じて表 I の割り当てで励起用とバランス用に分ける. これにより高次元問題を低次元サブ問題に分解できる (§III.E.1)
- **6 次 B-spline パラメータ化**: N_K=2 の via points で関節軌道を表現, 加速度連続性と始終端速度・加速度零の境界条件を満たす (§III.A)
- **Mass-weighted excitation criterion J_exc = Σ_{i=1}^{N_L} M_i · cond(W_bi)**: 各リンクごとの base regressor W_bi の条件数を対応リンク質量で重み付けした和. 大きい質量リンクの励起を優先し, 全体 regressor cond(W_b) 最小化より高速に収束する (§III.B, (12))
- **CoM 用静的姿勢最適化 (§III.D)**: 静止姿勢では慣性列が自動的に消え static BP のみが残ることを利用し, 60 姿勢を J_exc_static で最適化
- **動的姿勢遷移最適化 (§III.E.1)**: 隣接静的姿勢 p, p+1 間の B-spline via points を J_exc_dynamic で最適化. 直前までの動作の regressor を累積して次を最適化する逐次拡張
- **ZMP バランス最適化 (§III.E.2)**: 励起に関与しない下肢関節 q_DB を J_ZMP = ‖ZMP - ZMP_Mid‖² 最小化で解き, 支持多角形中央付近に ZMP を寄せる
- **物理整合性制約付き QP (§II.C, (8))**: min ‖F̄ - W̄Φ‖² + ‖Φ_CAD - Φ‖², s.t. M_i≥0, CoM_i∈bounding box, vᵀI_iv≥ε (単位球上の v_j をサンプリングした線形近似)

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
浮動ベース系の SIP 同定は Ayusawa+ [6], Mistry+ [4], Yamane [5], Ayusawa+ [26] が floating-base regressor 定式化と physical consistency を提案し, 直列マニピュレータの励起軌道最適化は Swevers+ [10] ([[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]]) の Fourier 級数と cond(W) 最小化, Park [9] の B-spline, Rackl+ [12] の B-spline 最適化などが確立されている.
先行の全身励起研究のうち Baelemans+ [15] と Mayr+ [16] は**静的姿勢のみ**で CoM を同定し慣性は扱えず, Bonnet & Venture [14] は**平面 (2D) モデル**での visual biofeedback を提案していた.
新規性は 3 点: (1) 3D 全身動的動作で全 SIP (質量・CoM・慣性) を同定する初の枠組み, (2) mass-weighted per-link condition number 和 J_exc という新しい励起基準 (§III.B, (12)) — 通常の cond(W_b) 最小化と比べ, 3 DoF 平面数値例で cond が 105→62 に下がり収束も高速 (Fig. 4b), (3) DoF を q_DE / q_DB に分解して低次元 QP に落とし込む階層最適化 (§III.E).

### どのように訓練・最適化したのか？
理論・実験論文であり学習は含まない. 最適化のみ.

- **損失関数 / 最適化目的**:
  - 励起軌道生成: J_exc = Σ_{i=1}^{N_L} M_i · cond(W̄_bi) を SQP (Matlab) で最小化. 静的姿勢最適化 (14), 動的遷移最適化 (18) が主体で, いずれも B-spline via points が決定変数
  - バランス調整: J_ZMP = ‖ZMP - ZMP_Mid‖² を q_DB について最小化 (§III.E.2)
  - 同定 QP (8): min ‖F̄ - W̄Φ‖² + ‖Φ_CAD - Φ‖², s.t. M_i≥0, CoM_i ∈ oriented bounding box, vᵀI_iv ≥ ε=10⁻³ (単位球上の v_j サンプルによる線形化)
- **データセット**: N/A — 学習用データセットは無い. 生成される「データ」は各姿勢遷移で得た力プレート測定 (人: AMTI BP-400600 100 Hz, HOAP-3: Accugait AMTI 1000 Hz) と VICON マーカ (人 8 カメラ 100 Hz, HOAP-3 VICON Bonita 100 Hz) の同期記録. 人モデルは N_J=23 DoF, N_L=12 リンク; HOAP-3 は N_J=21 DoF, N_L=12. 静的姿勢は N_p=60 (4 足配置 × 15), 姿勢遷移時間は人 T_F=5s / HOAP-3 T_F=2s, サンプリング 50 Hz. 姿勢最適化 1 回あたり平均 8±4s, 全体解算は 15 分未満

### どのように検証したか？指標と結果は？
人被験者 (33 歳女性, 65 kg, 175 cm) と HOAP-3 (7.9 kg, 0.88 m) で実験を行った.

- **人検証 (§IV.A, IV.B)**: 追加 2.4 kg を左腕に取り付けた条件と無負荷条件で SIP を比較. 識別された SIP から予測した外部レンチと実測レンチの全実験区間 RMS 誤差は, 識別モデルで 7.8±2.1 N / 4.3±1.9 N·m と AT (人体計測表) モデルの 12±1.7 N / 9.67±3.65 N·m より小さい. スクワット動作の平均 RMS は Id 12.5N / 3.2N·m vs AT 22.6N / 7.9N·m で, 垂直力 F_Z は AT が 2.2 倍大きい誤差. 相関係数は Id CC=0.73 vs AT CC=0.64. 腕位置での質量差は 1.9 kg として検出され, 追加 2.4 kg との差分 0.5 kg が精度. 追加質量を除いた平均 SIP 差は 0.25±0.17 kg (体幹 0.42 kg 最大). 先行研究 Ayusawa+ [26] の 0.3 kg 精度と同水準
- **HOAP-3 検証 (§IV.D)**: 42 base parameter が相対標準偏差 <10% で信頼識別. CAD と識別 SIP の比較で外部レンチ推定 RMS は Id 側が概ね 2 倍以上小さい (Table II: F_X Id 1.1N vs CAD 1.9N, F_Z Id 0.9N vs CAD 3.5N など). 別動作 (half sitting → 右足片脚立ち) でのクロス検証でも Id が CAD より小さな RMS
- **数値解析 (§III.C, Fig. 4)**: 3 DoF 平面モデルで J_exc 最小化と J_cond=cond(W_b) 最小化を比較. 最終的な cond は J_exc 側が 62, J_cond 側が 105 で J_exc 優位. さらに 6 動作にわたる収束が J_exc は 2 回目以降ほぼ収束するのに対し J_cond は初期値悪化時に破綻しやすい

### 検証結果に基づいた議論、明らかになった課題はあるか？
著者は §V Conclusion で以下の限界を明示的に述べている.

- (§V Conclusion) 現在の手法は励起運動を **open-loop で再生**しており, ロボット/被験者パラメータの良い初期推定が必要. 動的バランスをリアルタイム ZMP コントローラで扱う pseudo-online 同定への拡張が今後の課題として挙げられる
- (§IV.C.2 Robot specific technical issue) HOAP-3 の pitch 足首アクチュエータの柔軟性と足-地面間摩擦が振動を招き, 足首関節可動範囲を強く制限せざるを得なかった. このため下肢リンクの慣性行列同定は困難 (§IV.D 最終段落で「脚 links の慣性はほぼ識別不能」と明示)
- (§V Conclusion) 手法は電気筋刺激的 (EMG) 信号や非線形筋モデルを取り込んだ生体力学応用には未対応. 外部レンチのみを入力とするため関節局所トルクや筋パラメータは同定できない
- (§V Conclusion 最終段落) ダンス様励起動作の視覚的フィードバック interface (被験者への提示) の**エルゴノミー**は要検討. Kinect ベース等のダンスビデオゲーム風提示が今後の方向
- (§V Conclusion) 病理的被験者への適用は将来課題. 筋力低下・可動域制限の反映は原理的に可能だが未実証

---
## 自身の研究との関連
本プロジェクトの 6 DoF 逐次マニピュレータ + ハンマー慣性 10 パラメータ同定と, 本論文の全身 humanoid SIP 同定は問題設定が本質的に異なる.
Bonnet 2016 では全身動作すべてが励起対象で task-required drift は存在せず, DoF を q_DE / q_DB に分けるのは (task 制約ではなく) 動的バランス制約 (ZMP ⊂ 支持多角形) を捌くためである.
本プロジェクトの base drift は task (grasp 姿勢維持や target 到達) が非最適化方向に joint を強制するために生じるものであり, 分解構造の形式的類似はあっても駆動する制約の種類が異なる.

流用可能・引用義務のある要素:

- **per-link mass-weighted condition number 目的関数 J_exc = Σ M_i · cond(W_bi)** — 本プロジェクトでハンマーリンクとロボット末端リンクの質量差が cond を悪化させる場合の先行例として引用義務. ただし本プロジェクトは全 base parameter がリンク統合されたハンマー慣性 10 変数に限定されるため, 「リンク単位重み付け」がそのまま活きるかは要検討
- **6 次 B-spline via-point 最適化** — 加速度連続と境界条件を課しつつ探索次元を絞る手段として, task 拘束下 excitation 軌道生成の参考になる
- **物理整合性 QP (質量正 / CoM ∈ bounding box / vᵀIv ≥ ε の球面サンプル線形化)** — 本プロジェクトの後段 SIP 回復ステップの参考になる. Wensing 2017 の LMI 版 ([[papers/Wensing-RAL2017-LMIPhysicalConsistency/lmi-physical-consistency|Wensing+ 2017]]) と比較して, 線形制約近似で QP に押し込む簡便版として位置づけられる

差分 (本プロジェクトが解くべき課題として残るもの):

- Bonnet はバランス側 q_DB を別 QP で解けば task 側の cond に影響しない (可分). 本プロジェクトは task-required drift が excitation 方向と結合しているため, 単純な DoF 分解では非最適化 DoF が残らない
- 本論文の per-link 目的関数は多リンク humanoid での質量比不均衡を吸収する狙い. 本プロジェクトの単一ハンマーリンクでは対応する構造がないため, 直接転用ではなく「task-aware な cond 重み付け」への発展が必要

判定: **部分的類似 (formal analogy) だが本質的に別問題**. per-link mass-weighted cond 目的関数と DoF 分解構造は先行例として引用する.

---
## 追加議論


---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@article{bonnet2016optimal,
  title={Optimal Exciting Dance for Identifying Inertial Parameters of an Anthropomorphic Structure},
  author={Bonnet, Vincent and Fraisse, Philippe and Crosnier, Andr{\'e} and Gautier, Maxime and Gonz{\'a}lez, Alejandro and Venture, Gentiane},
  journal={IEEE Transactions on Robotics},
  volume={32},
  number={4},
  pages={823--836},
  year={2016},
  publisher={IEEE},
  doi={10.1109/TRO.2016.2583062}
}
```

</details>
