---
Title: Adaptive Experiment Design for Nonlinear System Identification with Operational Constraints
Authors:
  - Hu, Jingwei
  - Zachariah, Dave
  - Wigren, Torbjörn
  - Stoica, Petre
Year: 2025
Venue: arXiv
Tags:
  - "adaptive-experiment-design"
  - "nonlinear-system-identification"
  - "fisher-information-matrix"
  - "receding-horizon"
  - "operational-constraints"
  - "online-parameter-estimation"
PDF: "[[papers/Hu-arXiv2025-Adaptive_Experiment_Design/main.pdf|📃]]"
Import Date: "2026-07-08"
Read Date: 2026-07-08
Executive Summary: 状態・パラメータとも未知な非線形離散時間システムに対し、運用制約(状態上下限)を満たしながら、その時点までの推定値に基づいて逐次的に入力を設計するreceding-horizon型の適応実験計画法。誤差共分散で標準化したL-criterion(Fisher情報行列に基づく)を提案し、直近ブロックのみのFIMと事前共分散を組み合わせた保守的近似で計算量を抑える。振り子系のシミュレーションで、PRBS入力やEKFより高精度な推定と制約遵守を実証。
Citekey: Hu-arXiv2025-Adaptive_Experiment_Design
BibTeX Key: hu2025adaptive
DOI: none (arXiv preprintのみ確認)
Relevance: 5
Repository: https://github.com/jingwei91hu/adaptive-input-design
Category: note
Template Version: v2.3
---

## Executive Summary
状態・パラメータとも未知な非線形離散時間システムに対し、運用制約(状態上下限)を満たしながら、その時点までの推定値に基づいて逐次的に入力を設計するreceding-horizon型の適応実験計画法。誤差共分散で標準化したL-criterion(Fisher情報行列に基づく)を提案し、直近ブロックのみのFIMと事前共分散を組み合わせた保守的近似で計算量を抑える。振り子系のシミュレーションで、PRBS入力やEKFより高精度な推定と制約遵守を実証。

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
状態・パラメータが共に未知な非線形離散時間システムに対して、運用上の状態制約(operational constraints)を守りながら、パラメータ推定に有効な入力をどのようにオンラインで設計するか、という問い。

### 提案手法のアプローチと、その根幹をなす要素は何か？
receding horizon方式で、直近の$b$サンプルのブロックから最尤推定によりパラメータ$\theta$・状態$x_t$・ノイズ分散$v$を更新し(式17)、その推定値のまわりで保守的なFisher情報行列を計算して(式13–14)、L-criterion(式8)を運用制約のペナルティ項(式15)と合わせて最小化する入力列を求める(式16)。状態推定はunscented Kalman filterの予測ステップ(式19–20)で逐次更新する。

- **L-criterion(式8)**: 現在の誤差共分散$C_t$で標準化したCramér-Rao下界のトレース $\mathcal J=\mathrm{tr}(C_{t+k}C_t^{-1})\le d_\theta$。$\theta$の各成分の単位が非整合な場合でも比較可能にする
- **保守的FIM近似(式13–14)**: 過去データを最尤推定量$(\hat\theta,\hat x_t,\hat v)$とその漸近共分散(式11)に要約し、状態遷移のJacobian $E_{t+i}=\partial f^i_\theta/\partial(\theta,x_t)$を通じて現ホライズンのFIMを計算する。真のFIMより保守的(誤差を大きめに見積もる)下界を与える
- **ブロック単位のオンライン推定(式17)**: $b\ge d_\theta+d_x+d_y$サンプルごとに、Basin Hopping+L-BFGSで非凸最尤推定を解く
- **運用制約ペナルティ(式15–16)**: 状態が許容範囲を超えた分を二乗ペナルティ化し、情報量規準と同じ次元(標準化された単位)で重み付け合成する
- **入力の無制約変換(式21–22)**: 逆シグモイド関数で有界入力を無制約空間に写像し、Adam+自動微分で最適化する

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
開ループの非線形入力設計(Goodwin 1971 [11])、閉ループ(receding horizon)の制約付き設計(Babar & Baglietto 2021 [3])と対比している。既存のペナルティ型設計は、単位が非整合な$\theta$の各次元を直接組み合わせる規準を使うため競合目的のバランスが難しいと指摘し(§I)、本研究は現在の誤差共分散で標準化したL-criterionにより単位を揃えた規準を導入し、かつ初期値に敏感でないオンライン推定器を新規に提案する点を新規性としている。

### どのように訓練・最適化したのか？
- **損失関数 / 最適化目的**: 式(16) $U_t^*=\arg\min_{U_t\in\mathcal U}\tilde{\mathcal J}(U_t;\theta,x_t,v)+\gamma\mathcal J_\mathcal X(U_t;\theta,x_t)$。パラメータ推定は式(17)の正則化付き最尤推定(非凸、Basin Hopping+quasi-Newton L-BFGSで解く)。
- **データセット**: N/A: 学習ベース手法ではなく、学習データセットは存在しない。数値実験は非線形振り子モデル(式23、$\theta_1=-24$、$\theta_2=1$)のシミュレーションで、100回のモンテカルロ試行により評価している。

### どのように検証したか？指標と結果は？
PRBS入力2種(PRBS1: 振幅制約のみ満たす、PRBS2: 制約遵守を人手でチューニング)と提案手法を比較した。指標は正規化MSEとCramér-Rao下界(近似)、および運用制約違反量OCV(式)。結果、提案手法は角度状態を$[-45°,+45°]$の制約内に保ちながら(Fig.2)、PRBS1条件下のEKF推定よりも低いMSEを達成し(Fig.3)、CRBも提案手法のほうが低い(=より情報量の多い実験設計になっている)。PRBS2は制約違反が抑えられているがCRBに基づく評価では大きな誤差を示し、プロットから除外されている。実行時間は標準ノートPC(シングルスレッドCPU)で1試行あたり0.2〜5秒(ホライズン長$k$とモデル複雑度に依存)。

### 検証結果に基づいた議論、明らかになった課題はあるか？
(§V Conclusionより) 著者は結論部で限界を明示的には論じておらず、「運用制約内で安全な実験を行いながら、推定精度の大幅な向上と実験時間短縮を達成した」という成果の要約のみで締めくくっている。著者は限界に明示的には言及していない。ただし本文中(§III-B)には、緩和された規準$\tilde{\mathcal J}$が真の規準$\mathcal J$に対して「$t$が小さいうちは緩い(loose)可能性があるが、最尤推定量の漸近有効性が成り立てばタイトになりうる」という条件付きの精度限界が記述されている。

---
## 自身の研究との関連
本論文の対象モデル $x_{t+1}=f_\theta(x_t,u_t)$(数値実験では振り子 $x^{(2)}_{t+1}=x^{(2)}_t+(\theta_1\sin x^{(1)}_t+\theta_2 u_t)\Delta T$)は、$\theta$について**非線形**である。Fisher情報行列のJacobian $E_{t+i}=\partial f^i_\theta/\partial(\theta,x_t)$ は現在の推定値 $\hat\theta_t$ のまわりで評価され(式13)、真の最適入力は未知の $\theta$ に依存する。これはP1が援用する逐次実験計画理論(Chernoff、Box-Hunter、Fedorov[7]のchicken-and-egg問題)がそのまま適用できる設定である。

対照的に、P1の対象(手首力覚センサで把持物体の慣性を推定する固定基盤マニピュレータ)は回帰行列が$\theta$について線形であり、本論文の設定とは構造的に異なる。この差は、P1提案がH3で本論文型の非線形システム向け逐次設計理論を援用することの妥当性を問い直す材料になる。

---
## 追加議論

---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@article{hu2025adaptive,
  title={Adaptive Experiment Design for Nonlinear System Identification with Operational Constraints},
  author={Hu, Jingwei and Zachariah, Dave and Wigren, Torbj{\"o}rn and Stoica, Petre},
  journal={arXiv preprint arXiv:2502.20941},
  year={2025}
}
```
</details>
