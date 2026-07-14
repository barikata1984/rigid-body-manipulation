---
Title: PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics
Authors:
  - Xie, Tianyi
  - Zong, Zeshun
  - Qiu, Yuxing
  - Li, Xuan
  - Feng, Yutao
  - Yang, Yin
  - Jiang, Chenfanfu
Year: 2024
Venue: CVPR
Tags:
  - "3d-gaussian-splatting"
  - "material-point-method"
  - "continuum-mechanics"
  - "physics-based-simulation"
  - "neural-rendering"
  - "generative-dynamics"
PDF: "[[papers/Xie-CVPR2024-PhysGaussian/main.pdf|📃]]"
Import Date: "2026-07-12"
Read Date: 2026-07-12
Executive Summary: 3D Gaussian Splatting（GS）で再構成した静的シーンに、後段の物理シミュレータやメッシュ化なしで妥当な動力学を付与できるかという課題に対し、GS の各ガウス核をMaterial Point Method（MPM）の物質点として直接扱い、変形勾配に応じてガウス核自身（位置・共分散・球面調和関数の向き）を更新する「what you see is what you simulate」設計を提案した。弾性体・塑性金属・粒状体・非ニュートン流体など多様な材質でのリアルタイム生成に成功した一方、影の再計算や材質パラメータの自動推定は今後の課題として残る。
Citekey: Xie-CVPR2024-PhysGaussian
BibTeX Key: xie2024physgaussian
DOI: 10.1109/CVPR52733.2024.00420
Relevance: 2
Repository: https://xpandora.github.io/PhysGaussian/
Category: note
Template Version: v2.3
---

## Executive Summary
3D Gaussian Splatting（GS）で再構成した静的シーンに、後段の物理シミュレータやメッシュ化なしで妥当な動力学を付与できるかという課題に対し、GS の各ガウス核をMaterial Point Method（MPM）の物質点として直接扱い、変形勾配に応じてガウス核自身（位置・共分散・球面調和関数の向き）を更新する「what you see is what you simulate」設計を提案した。弾性体・塑性金属・粒状体・非ニュートン流体など多様な材質でのリアルタイム生成に成功した一方、影の再計算や材質パラメータの自動推定は今後の課題として残る。

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
NeRF や 3D Gaussian Splatting（GS）で再構成した静的シーンに新規の動力学（generative dynamics）を与える従来手法は、シミュレーション用のジオメトリ（三角メッシュ・四面体・ケージメッシュ）を別途構築し、それを介してレンダリング用表現を変形させる多段階パイプラインに依存していた（§1 Introduction）。この「シミュレーションされる形状」と「レンダリングされる形状」の分離は、メッシュ化コストや解像度の不一致という問題を生む。本論文は、GS の3Dガウス核をそのままシミュレーションの離散表現として使い、レンダリングとの間に一切の中間ジオメトリを挟まない統一パイプラインを構築できるかという問いに答える。

### 提案手法のアプローチと、その根幹をなす要素は何か？
静的シーンをまず GS で再構成した後、その3Dガウス核の集合を連続体力学における物質点群とみなし、カスタムの Material Point Method（MPM）で時間発展させ、得られた変形をガウス核自身のパラメータ更新として直接レンダリングに反映させる（§3 Method Overview）。連続体の変形写像 $\phi(X,t)$ の一次近似（局所アフィン変形、式(6)）により、変形後もガウス核がガウス形を保つことを利用し、MPM とスプラッティングを同じ粒子表現で橋渡ししている。
- **局所アフィン近似による変形ガウス核の保存**（式(5)-(8)）: 各ガウス核の材質空間中心 $X_p$ 周りで変形写像を一次展開することで、変形後の核 $G_p(x,t)$ が閉形式のガウス分布であり続けることを保証する。これが GS のレンダリング式と MPM の粒子表現を破綻なく接続する数学的な要石。
- **カスタム MPM による運動学と力学の統合**（§3.3, 3.4; APIC 転送スキームの詳細は Appendix A）: 位置・速度・変形勾配をラグランジュ粒子（ガウス核）とオイラーグリッド間で往復させ、連続体力学の質量・運動量保存則（式(2)-(4)）に従って弾性エネルギー・応力・塑性を陽に解く。多様な構成則（固定コロテーショナル弾性、von Mises 塑性、Drucker-Prager 塑性、Herschel-Bulkley 塑性）を差し替え可能にしている。
- **球面調和関数の向きの回転更新**（§3.5, 式(9)）: 変形勾配の極分解から得られる回転成分 $R_p$ で球面調和基底を回転させ、視点依存の見た目（鏡面反射など）が変形後も物体表面の実回転と整合するようにする。
- **内部充填（internal filling）**（§3.7）: GS は物体表面近傍にのみガウス核を分布させるため、再構成直後のシーンは内部が空洞である。密度場（式(11)）を用いて内部にも粒子を補完し、体積を持つ物体としての力学的挙動（自重での崩壊回避など）を可能にする。

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
GS フレームワーク自体は Kerbl et al.（3D Gaussian Splatting, [16]）に基づき、シミュレーション手法は Stomakhin et al. の MPM（[39]）および Zong et al. の弾塑性構成則（[53]）を採用している。動的 NeRF/GS 分野の先行研究との対比として、Wu et al.（4D Gaussian Splatting, [45]）や Luiten et al.（Dynamic 3D Gaussians, [22]）はガウス核の形状を固定するか学習で変形させるのみで物理法則を組み込んでいない。Li et al.（PAC-NeRF, [18]）は物理シミュレータを NeRF に統合した点で近いが、物体形状をシステム同定向けの単純な形状に限定しており高忠実度レンダリングを主眼としない。Qiao et al.（Neuphysics, [31]）を含む従来の物理統合手法は、NeRF から抽出したメッシュ経由でシミュレーションと結び付けるため中間ジオメトリの解像度不一致が生じる。PhysGaussian の新規性は、シミュレーションとレンダリングの表現を完全に一致させる「what you see is what you simulate（WS²）」の実現にあり、変形勾配（一次情報）まで含めてガウス核自体を直接シミュレーション状態として扱う点で、ゼロ次情報（変形写像）のみを使う先行手法と異なる（§4.2 Comparisons）。

### どのように訓練・最適化したのか？
- **損失関数 / 最適化目的**: 静的シーン再構成段階（3D Gaussian Splatting のオリジナル最適化）では $L_1$ 損失と SSIM 損失を用いてレンダリング画像と入力画像の再構成誤差を最小化する（§3.1）。加えて、過度に扁平なガウス核による見た目の破綻を防ぐための異方性正則化損失（式(12)、$\mathcal{L}_{aniso}$）を任意で追加できる。物理シミュレーション自体（MPM の時間積分）は学習ベースの最適化ではなく、連続体力学の保存則（式(2)(3)）と構成則を陽的に前進オイラー積分するものであり、勾配降下による最適化対象ではない。
- **データセット**: 合成データとして BlenderNeRF による sofa suite（衝突）、Instant-NGP の fox、Nerfstudio の plane、DroneDeploy NeRF の ruins を使用。実世界データとして iPhone で撮影した toast・jam（各150枚の写真、COLMAP でカメラパラメータと初期点群を復元）を独自収集した（§4.1 Datasets）。なお vasedeck は Appendix C（Additional Evaluations, Fig. 9）で使われている別データセットで、NeRF dataset 由来である。ラティス変形ベンチマークでは BlenderNeRF により wolf・stool・plant の各シーンについて未変形状態100枚・変形状態（bend/twist）各100枚のマルチビュー画像を合成している（§4.2 Dataset）。

### どのように検証したか？指標と結果は？
評価は大きく2種類。(1) 多様な材質（弾性・塑性金属・破断・粒状体・粘塑性流体・衝突）での生成的ダイナミクスの定性評価（Fig. 3, Fig. 9）。(2) ラティス変形ベンチマークでの定量評価（Tab. 1）で、NeRF-Editing、Deforming-NeRF、PAC-NeRF をベースラインとし、bend/twist変形後のレンダリング画像に対する PSNR を wolf・stool・plant の3シーンで比較した。結果は提案手法が全ケースで最高 PSNR を達成（例: stool-bend 31.15 dB、ベースライン3手法は21.83-25.00 dB、アブレーション各種は26.77-30.87 dB）。アブレーション（Fixed Covariance、Rigid Covariance、Fixed Harmonics）でも、球面調和関数の回転更新を含むフル手法が最良のスコアを示した（Tab. 1下段）。ボリューム保存の定性比較（Fig. 7）では、幾何ベースの NeRF-Editing に対して物理ベースの提案手法がより現実的な体積挙動を示すことを示した。速度面では plane 30 FPS、toast 25 FPS、jam 36 FPS のリアルタイム性能を、24コア3.50GHz Intel i9-10920X + NVIDIA RTX 3090 で達成した（§4.1 Results）。

### 検証結果に基づいた議論、明らかになった課題はあるか？
(§5 Discussion, Limitation より) 本フレームワークでは影の再計算（evolution of shadows）が考慮されておらず、材質パラメータ（ヤング率・ポアソン比等）は手動設定である。GS セグメンテーションと微分可能 MPM シミュレータを組み合わせることで、動画からの材質パラメータ自動推定が可能になりうるとしている。また、今後の課題として、より多様な材質（液体）への対応、ユーザー操作をより直感的にすること、大規模言語モデル（LLM）の活用可能性が挙げられている（§5 Discussion 末尾）。加えて（§4.1 Results 末尾）、FEM を用いればより高精度な弾性シミュレーションが可能かもしれないが、メッシュ抽出という追加ステップが必要になり、MPM が持つ非弾性挙動への汎用性を失うとも述べている。

---
## 自身の研究との関連
本プロジェクトは、マニピュレータが手首装着 FT センサで把持物体の慣性パラメータ（質量・重心・慣性テンソル）をオンラインに同定する研究（励振軌道のリアルタイム最適化、蓄積型フィッシャー情報行列による逐次計画）を主題としており、PhysGaussian とは問題設定・扱う物理量ともに直接の重なりは薄い。PhysGaussian が最適化・推定するのは見た目の再現に十分な弾性率・降伏応力等の連続体構成則パラメータであり、剛体の質量・重心・慣性テンソルという「6自由度剛体力学のパラメータ」ではない。またリアルタイム性の意味も異なり、本プロジェクトが問題にする「オンラインでの逐次情報蓄積・励振計画」ではなく、シミュレーション自体の計算速度（FPS）を指す。

接点として言及できるのは以下の2点にとどまる。第一に、PhysGaussian の internal filling（§3.7）が示すように、表面のみを観測する視覚的表現（GS）は物体内部の質量分布に関する情報を欠き、内部密度分布の何らかの仮定（一様密度）を補わない限り力学的に妥当な挙動を再現できない。これは [[papers/Nadeau-ICRA2023-SumOfItsParts/the-sum-of-its-parts-visual-part-segmentation-for-inertial-parameter-identification-of-manipulated-objects|Nadeau+ 2023]] が採用する「パーツごとの均質密度」仮定と同型の制約であり、視覚情報だけからは剛体の質量分布（≒慣性パラメータ）を一意に決定できないという、本プロジェクトが扱う同定問題の根本的な難しさを別の角度から裏付ける。第二に、著者らが Limitation で述べる「材質パラメータの自動推定には微分可能 MPM シミュレータとの組み合わせが必要」という指摘は、視覚的再構成だけでは動力学パラメータの同定が閉じず、何らかの実測（本プロジェクトの文脈では FT センサによる力学的観測）または微分可能シミュレーションによる勾配情報が別途必要になるという点で、本プロジェクトの問題意識（視覚情報のみでは不十分、実測に基づく逐次更新が必要）と方向性が一致する。

一方、[[papers/Zhang-arXiv2025-ProvablySafe/provably-safe-online-system-identification|Zhang+ 2025]] や [[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation|Albee+ 2022]] が扱う「オンラインでの逐次パラメータ更新と計画への反映」というループは、PhysGaussian には存在しない（材質パラメータは一度手動設定されると固定）。MPM 自体をロボット把持物体の同定に応用する可能性は理論上あり得るが、剛体力学の閉形式レグレッサ（本プロジェクトが依拠する $6\times10$ の回帰行列）に比べて計算コストが著しく高く、リアルタイム励振計画のループに組み込む現実性は低いと考えられる。総じて、関連は限定的であり、本論文は直接の手法的示唆よりも、視覚的表現から物理パラメータを推定する際の構造的な限界（内部情報の欠如）を確認する参考事例としての価値が主となる。

---
## 追加議論


---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@inproceedings{xie2024physgaussian,
  title     = {PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics},
  author    = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4389--4398},
  year      = {2024},
  doi       = {10.1109/CVPR52733.2024.00420}
}
```
</details>
