---
Title: Provably-Safe, Online System Identification
Authors:
  - Zhang, Bohao
  - Zhou, Zichang
  - Vasudevan, Ram
Year: 2025
Venue: arXiv
Tags:
  - "system-identification"
  - "safe-motion-planning"
  - "exciting-trajectory"
  - "inertial-parameter-estimation"
  - "reachability-analysis"
  - "manipulator-control"
PDF: "[[papers/Zhang-arXiv2025-ProvablySafe/main.pdf|📃]]"
Import Date: "2026-07-11"
Read Date: 2026-07-11
Executive Summary: 未知ペイロードを把持したマニピュレータが、関節・トルク制約と障害物回避を厳守しながらオンラインでペイロードの慣性パラメータを同定する枠組みを提案。運動量ベース回帰とlog-Choleskyパラメータ化により物理的整合性を保った非線形最小二乗を定式化し、摂動解析でセンサノイズを含む保守的な区間推定を導出。ARMOURの安全軌道計画のコスト関数を標準ダイナミクス回帰行列の条件数最小化に置き換えることで、安全性を保証したまま同定を加速する励起軌道を生成する。Kinova-gen3実機実験で、比較手法が衝突やトルク超過で失敗する中、提案法のみ全タスクに成功。
Citekey: Zhang-arXiv2025-ProvablySafe
BibTeX Key: zhang2025provably
DOI: arXiv:2504.21486
Relevance: 5
Repository: https://roahmlab.github.io/OnlineSafeSysID/
Category: note
Template Version: v2.3
---

## Executive Summary
未知ペイロードを把持したマニピュレータが、関節・トルク制約と障害物回避を厳守しながらオンラインでペイロードの慣性パラメータを同定する枠組みを提案。運動量ベース回帰とlog-Choleskyパラメータ化により物理的整合性を保った非線形最小二乗を定式化し、摂動解析でセンサノイズを含む保守的な区間推定を導出。ARMOURの安全軌道計画のコスト関数を標準ダイナミクス回帰行列の条件数最小化に置き換えることで、安全性を保証したまま同定を加速する励起軌道を生成する。Kinova-gen3実機実験で、比較手法が衝突やトルク超過で失敗する中、提案法のみ全タスクに成功。

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
未知ペイロードを把持したロボットマニピュレータにおいて、(1) 安全な計画と安全な制御のギャップ（モデル不確実性によるトラッキング誤差が衝突やトルク限界違反を招く）、(2) モデル不確実性（エンドエフェクタの慣性パラメータ範囲）の正確な推定、(3) システム同定中の安全保証（衝突回避・トルク限界遵守）、という3つの課題を統合的に解決することを目指した（§I Introduction）。従来研究は励起軌道設計と同定精度に注力する一方、同定プロセス中の障害物回避やトルク限界遵守をほとんど扱っていなかったと指摘されている（§I Introduction）。

### 提案手法のアプローチと、その根幹をなす要素は何か？
エンドエフェクタの慣性パラメータの区間上界（overapproximated bound）を初期値として持ち、その区間に基づき安全性を保証しつつ局所的に励起性の高い（locally exciting）軌道を実時間生成して追従し、得られたデータからより厳密な区間上界を再計算する、という反復ループ（Fig. 2）が全体戦略である。同定と軌道計画は「逆順」に提示されており、同定手法が先に定式化され、その後にそれを支える励起軌道計画が説明される構成になっている（§I）。
- 運動量ベースダイナミクス（Theorem 5, 6, 8）により、ノイズの大きい加速度計測を回避しつつ、システム運動量の時間積分という形で線形回帰問題（Corollary 10, 式(20)-(22)）を構成する
- log-Choleskyパラメータ化（Theorem 16, 式(14)-(16)）により、慣性パラメータの物理的整合性（LMI制約、Definition 15）を、制約付き最適化ではなく非線形最小二乗問題（Theorem 11, 式(25)）として非拘束化する
- パラメータ回復の写像 P が微分同相（diffeomorphism）であること（Corollary 17）により、変換後の非凸最適化に見せかけの局所最小解が生じないことを保証する
- 摂動解析（Theorem 18, Appendix B）に基づく感度解析により、計測ベクトルの区間ノイズ上界 [m] から、真の慣性パラメータを含むことが保証された区間上界 θ*_e(m) + (∂θ*_e/∂m)([m]-m)（Theorem 13, 式(27)）を導出する
- ARMOUR（[28], 既存の到達可能性解析に基づく安全モーションプランナー）のコスト関数を、標準ダイナミクス回帰行列 W の2ノルム条件数最小化（式(30)）に置き換えることで、安全性の証明（Lemma 22 of [28]）を保ったまま励起性の高い軌道を生成する（§V-B）
- 上記を統合したAlgorithm 2（Provably-Safe Online System Identification）により、軌道追従・データ収集・パラメータ更新を反復し、Lemma 14により全過程を通じて衝突回避とすべての限界遵守が証明される

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
安全な軌道計画の基盤として ARMOUR [28]（到達可能性解析に基づく実時間安全モーション計画・制御）に依拠し、そのフィードバック制御と受動的な安全保証をそのまま利用している（§V-A）。運動量ベースの回帰は [31] や momentum-based adaptive control の系譜（[30] 等）を踏まえている（§II Related Work、Theorem 5, 8 の出典）。log-Cholesky物理整合性パラメータ化は Rucker & Wensing [35, Section V.A] の結果（Corollary 17 の微分同相性）に基づく。励起軌道設計自体（条件数最小化によるフィッシャー情報量最大化など）は Gautier & Khalil [13]、Ayusawa et al. [4]、Venture et al. [45]、Bonnet et al. [7] ら既存研究の系譜にある（§II、§V-B）。
新規性は、著者らの整理によれば以下の点にある。既存の励起軌道研究は「安全な励起軌道生成」と「安全な軌道追従」の両方を満たすものがほとんど存在せず（唯一 [11] が安全な励起軌道生成を試みるがサンプリングベースで衝突自由性しか保証できない）、また閉ループ状態/入力感度に基づくアプローチ（[15], [41]）はUAVやクアッドロータ、限定的なペイロード範囲のマニピュレータにしか適用されていなかった（§II）。本論文は、ARMOURの証明可能な安全軌道追従とlog-Cholesky回帰に基づく証明可能に保守的な区間パラメータ推定を統合し、障害物回避とトルク限界を厳守しながらオンラインでシステム同定を行う初めての枠組みであると位置づけられている（§I 末尾、"two-fold" contributions）。また、適応制御（adaptive control）との対比では、従来の適応制御が収束速度の保証や状態/入力制約の強制ができない点を明示的な限界として指摘し（§II 末尾）、本手法はこれに対して安全性の証明可能性を提供する。

### どのように訓練・最適化したのか？
- **損失関数 / 最適化目的**: システム同定は非線形最小二乗 min_{η_e∈R^10} ||Y(q,q̇)P(η_e) − U(q,q̇,τ,θ_{r,0})|| （式(25)）。η_e は log-Cholesky パラメータで、P によって慣性パラメータ θ_e = P(η_e) に写像される。励起軌道生成は ARMOUR の軌道最適化（式(28)）において、コスト関数を標準ダイナミクス回帰行列 W(k)（式(30)）の2ノルム条件数最小化に設定した拘束付き非線形計画として解く。両問題とも Ipopt（解析的勾配・ヘシアン提供）で解いている（§VI-D）
- **データセット**: シミュレーションデータセットは用いず、実機（Kinova-gen3、7自由度）でのオンライン収集データのみを使用。前進オイラー積分の積分区間（forward integration horizon）h=400（積分時間100-120ms）、励起軌道の長さ t_f=3.0s、計画時間 t_p=1.5s、Algorithm 2は4本の励起軌道生成で打ち切り（同定フェーズ合計 (4-1)×1.5+3.0=7.5秒）。回帰行列構成のサンプル数 N_s=128。計測ノイズ上界 [δm] はロボットダイナミクスパラメータ θ_r（エンドエフェクタ除く）について最大5%、印加トルク計測 τ について2.5%と設定（§VI-D）。実験は5種のダンベル（4, 5, 6, 7, 8 lb）を対象に、各設定を5回繰り返した（§VI-E）

### どのように検証したか？指標と結果は？
実機実験は3種類。(a) 5個のダンベルをロボット前方0.25mの3Dプリント台に軽い順に積み上げるタスク（低障害物あり）、(b) 同タスクを台を0.50mに配置し高障害物ありで実施、(c) 最重量8lbダンベルを同定後、障害物を避けながら搬送する高難度タスク（Fig. 3, 4）。比較手法は TABLE II に整理された10種（ours, wrong, conservative, random, adap-1, adap-1-excit, adap-2, adap-2-excit, grav-pid, grav-pid-ours, grav-pid-excit）で、コントローラ種別（ARMOUR robust／adaptive／gravity-compensated PID）、ロボットモデルの取得方法、励起軌道の有無で類型化されている。各実験は5回繰り返し、結果は5試行で一貫していたと報告（§VI-E）。
TABLE III の結果: 提案法（ours）のみ実験(a)(b)(c)すべてで成功（success）。他手法は例えば wrong は(a)(b)ともに8lbで失敗、conservative は(a)で8lbにて失敗するが(b)は成功、random は(a)は成功するが(b)で8lbにて失敗、adap-1 は(a)は成功するが(b)で8lbにて失敗、adap-1-excit は(a)(b)とも成功、adap-2 は(a)は8lbで失敗し(b)は4lbで失敗、grav-pid は(a)で成功したが(b)で6lbにて失敗、grav-pid-ours は(a)(b)とも成功、grav-pid-excit は(a)(b)とも4lbで失敗、いずれの比較手法も実験(c)では衝突（collide）に至った。加えて TABLE V（回帰行列 Y の2ノルム条件数）では、提案法（例: 4lbで274.030）が random（同479.573）より一貫して小さく、より励起性の高い軌道を生成できていることを示す。Fig. 5-10 では区間推定幅が random より小さく、真値（ground truth）を含みつつタイトな区間を得たことをも示している（Theorem 13の実証的検証）。

### 検証結果に基づいた議論、明らかになった課題はあるか？
(§VII Limitations より) 以下3点が著者により明示されている。
- 励起軌道はARMOURに基づき degree-5 Bezier曲線でレシーディングホライズン計画されるため局所的な励起特徴しか捉えられず、Fourierベースの手法と比べて一般に条件数の大きい回帰行列になる
- ノイズはトルク計測に支配されると仮定し、前進オイラー積分によりデータが生成されるとも仮定している。より精度の高い数値積分に拡張可能だが、それでも真のシステムダイナミクスへの近似に留まる
- log-Choleskyパラメータ化 P は正定対角要素を強制するため指数関数を含み、これが式(42)のヘシアン行列の逆行列計算において数値的問題を生じさせうる。このためオンライン推定においてよりタイトな区間を得ることが困難になる

(§VIII Conclusion 末尾でも同旨の要約) 全体として、証明可能な安全軌道最適化と摂動解析に基づく頑健なシステム同定を組み合わせることで、重量ペイロードを持つロボットアームの安全性を維持しつつ精密な操作性能を達成したと結論づけている。

---
## 自身の研究との関連
本論文はrealtime_excitation研究（実時間励起軌道設計・オンラインシステム同定・ロボットダイナミクスパラメータ推定）と直接的に重なる。特に以下の点で参照価値が高い。

- 励起軌道設計の観点で、[[papers/Swevers-TRA1997-OptimalExcitation/optimal-robot-excitation-and-identification|Swevers+ 1997]] が対象とする「条件数最小化による励起軌道生成」という古典的枠組みを、安全制約（衝突回避・トルク限界）付きの実時間レシーディングホライズン計画に統合した点は直接の発展形といえる。本論文自身も§II・§V-Bで Gautier & Khalil や Venture らの条件数最小化の系譜を明示的に踏襲しており、Swevers 系の手法との比較軸を持つ
- [[papers/Kubus-IROS2008-Recursive_Total_Least-Squares/on-line-estimation-of-inertial-parameters-using-a-recursive-total-least-squares-approach|Kubus+ 2008]] が扱う再帰的Total Least Squaresによるオンライン同定と対比すると、本論文は運動量ベース回帰＋log-Choleskyパラメータ化＋摂動解析という異なる経路で「オンライン性」と「物理的整合性（正定性）保証」を両立しており、TLSベースのノイズ処理とは異なる保守的区間推定（区間演算による証明可能な上界）を提供する点が差別化になる
- [[papers/Nadeau-ICRA2022-FastObjectInertial/fast-object-inertial-parameter-identification-for-collaborative-robots|Nadeau+ 2022]]・[[papers/Nadeau-ICRA2023-SumOfItsParts/the-sum-of-its-parts-visual-part-segmentation-for-inertial-parameter-identification-of-manipulated-objects|Nadeau+ 2023]] の物体慣性パラメータ推定（把持物体の慣性推定）とは対象が近い（エンドエフェクタ+ペイロードの慣性パラメータ推定）が、本論文は障害物回避・トルク限界の証明可能な安全性を同時に扱う点で一段上位の枠組みを提供しており、Nadeauらの手法にARMOURのような安全軌道計画層を組み合わせる拡張の着想を与えうる
- Lee-TRO2019-GeometricIdentification の幾何学的パラメータ化（正定性を保証する別アプローチ）とは、log-Cholesky（Rucker & Wensing系）対幾何学的手法という「物理的整合性の強制方法」の比較軸で直接対比可能。本論文がCorollary 17でP写像の微分同相性を利用し局所最小=大域最小を保証する議論は、幾何学的パラメータ化との理論的差異を検討する際の参照点になる
- [[papers/Albee-RAL2022-RATTLE/the-rattle-motion-planning-algorithm-for-robust-online-parametric-model-improvement-with-on-orbit-validation|Albee+ 2022]] とは「オンラインでの動的パラメータ推定と制御の統合」という設計思想を共有しており、ARMOURの到達可能性解析ベースの安全保証を他プラットフォーム（自由飛行ロボット等）に応用する際の比較対象となりうる
- 本プロジェクトが実時間励起軌道の生成そのものを主題とするなら、Algorithm 2のレシーディングホライズン型「同定→再計画」ループ、および式(30)（標準ダイナミクス回帰行列の条件数を運動量回帰行列の条件数の近似として使う設計、Appendix D）は、計算コストと励起性のトレードオフを扱う際の直接的な参考実装になる。一方、本論文はARMOURという特定の安全軌道計画基盤に強く依存しており、他の到達可能性解析・障害物表現（zonotope以外）への一般化可能性は論文中で検証されていない点は、相補的に検討すべき差分である

---
## 追加議論

---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@article{zhang2025provably,
  title={Provably-Safe, Online System Identification},
  author={Zhang, Bohao and Zhou, Zichang and Vasudevan, Ram},
  journal={arXiv preprint arXiv:2504.21486},
  year={2025}
}
```
</details>
