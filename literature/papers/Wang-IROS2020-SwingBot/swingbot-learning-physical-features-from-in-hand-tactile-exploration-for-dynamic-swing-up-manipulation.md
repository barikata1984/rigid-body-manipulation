---
Title: SwingBot: Learning Physical Features from In-hand Tactile Exploration for Dynamic Swing-up Manipulation
Authors:
  - Wang, Chen
  - Wang, Shaoxiong
  - Romero, Branden
  - Veiga, Filipe
  - Adelson, Edward
Year: 2020
Venue: IROS
Tags:
  - "tactile-sensing"
  - "in-hand-manipulation"
  - "dynamic-manipulation"
  - "self-supervised-learning"
  - "physical-parameter-estimation"
PDF: "[[papers/Wang-IROS2020-SwingBot/main.pdf|📃]]"
Import Date: "2026-07-11"
Read Date: 2026-07-11
Executive Summary: 未知物体の物理特性（摩擦・質量・重心・慣性モーメント）を明示的に推定せずにswing-up操作を正確に行うことが課題。GelSightタッチセンサでtilting（傾け）とshaking（揺すり）の2種類のin-hand探索を行い、CNN+MLPおよびCNN+MLP+LSTMでそれぞれ特徴を抽出、fusionモデルで40次元の物理特徴埋め込みに統合する。この埋め込みと制御パラメータ（グリッパー幅）から最終swing角度を予測するforward dynamicsモデルをend-to-endで自己教師あり学習し、目標角度に最も近い予測を与える制御パラメータを探索的に選択する。未見物体で17.2度の誤差を達成し、探索なしのベースラインを13度以上上回った。埋め込みは物理量に事後的にdisentangle可能である一方、単一タスク・単一ハードウェアに強く条件付けられている点が限界として述べられている。
Citekey: Wang-IROS2020-SwingBot
BibTeX Key: wang2020swingbot
DOI: 10.1109/IROS45743.2020.9341006
Relevance: 4
Repository: http://gelsight.csail.mit.edu/swingbot
Category: note
Template Version: v2.3
---

## Executive Summary
未知物体の物理特性（摩擦・質量・重心・慣性モーメント）を明示的に推定せずにswing-up操作を正確に行うことが課題。GelSightタッチセンサでtilting（傾け）とshaking（揺すり）の2種類のin-hand探索を行い、CNN+MLPおよびCNN+MLP+LSTMでそれぞれ特徴を抽出、fusionモデルで40次元の物理特徴埋め込みに統合する。この埋め込みと制御パラメータ（グリッパー幅）から最終swing角度を予測するforward dynamicsモデルをend-to-endで自己教師あり学習し、目標角度に最も近い予測を与える制御パラメータを探索的に選択する。未見物体で17.2度の誤差を達成し、探索なしのベースラインを13度以上上回った。埋め込みは物理量に事後的にdisentangle可能である一方、単一タスク・単一ハードウェアに強く条件付けられている点が限界として述べられている。

---
## Summary

### この論文が答えた問い、あるいは解決した課題は何か？
把持物体の質量・重心・慣性モーメント・表面摩擦といった物理特性は、swing-up（振り上げ）のような重力・慣性の影響を強く受ける動的操作の成否を大きく左右する（§I, Fig. 2）。従来手法は、タスクに重要な物理パラメータをあらかじめ専門家が特定し、力学モデルと正確な計測を要求するため実世界展開が難しい（§II）。本論文は、事前に物理パラメータを明示的に測定・回帰することなく、少数回のin-hand触覚探索から得たタッチ情報のみを使い、未見物体に対しても目標角度へのswing-up操作を精度良く実行できるかを問うている（§I, §III）。

### 提案手法のアプローチと、その根幹をなす要素は何か？
SwingBotは、GelSightタッチセンサを用いた2種類のin-hand探索行動（tiltingとshaking）から得たタッチ情報を、自己教師あり学習によって40次元の「物理特徴埋め込み（physical feature embedding）」に集約し、この埋め込みと制御パラメータ（グリッパー幅）を入力としてswing-up後の最終角度を予測するforward dynamicsモデルを学習する。推論時には、複数の制御パラメータ候補についてこのモデルで角度を予測し、目標角度に最も近い予測を与えるパラメータを選んで実行する、という探索的な最適制御パラメータ選択を行う（§III, Fig. 3）。

- **Tilting（傾け）**: 0度に保持した物体を20度・45度に傾け、各角度でのマーカー変位（W×H×2, W=14, H=12）をCNN（カーネルサイズ5×5, 3×3, 2×2）+全結合層で40次元embeddingに変換。質量と重心（トルク情報から）の情報を主に符号化する（§III-B, Fig. 4）。
- **Shaking（揺すり）**: グリッパー力を緩めて0度姿勢のまま前後回転を素早く切り替え（実験では5度）、60〜70フレームのマーカー変位系列を同一CNN+MLPで各フレーム40次元に変換した後、LSTM（隠れ状態hとセル状態cを結合し80次元）で時系列情報を集約。摩擦情報を主に符号化する（§III-B, Fig. 4）。
- **Fusionモデル**: tiltingとshakingそれぞれのembeddingを結合し、物理特徴embedding [1:40] を出力（Fig. 3）。
- **Prediction model（forward dynamics）**: 物理特徴embeddingと制御パラメータ（グリッパー幅）を入力し、最終swing角度を予測。訓練時は実際にswing-upした結果の最終角度を自己教師ラベルとして、パイプライン全体をend-to-endで学習する（§III-C）。
- **Impulse-momentum法によるswing-up制御**: 手首を回転させながら物体を上方・回転方向に加速して運動量を蓄積し、逆方向に急加速してインパルスを作った瞬間にグリッパーを緩め、慣性で物体を自由回転させた後、任意のタイミングでグリッパーを締めて停止させる（§III-C, 参考文献[27]の手法を利用）。

### 特に参考とした既存研究と、それらと比した提案手法の新規性は何か？
Sintov et al.（参考文献[6]）はswing-up操作の力学解析を行い、表面摩擦・質量・重心・慣性モーメントがswing-upダイナミクスに寄与することを明らかにしており、本研究の探索行動設計（tilting/shaking）はこの知見に基づく（§III-B）。ただし[6]を含む従来手法は、対象パラメータを専門家が事前指定し、力学モデルと正確な物理量計測を必要とする点で実世界展開が困難であるのに対し、本研究は物理パラメータを直接回帰せず、自己教師あり学習によりタスク遂行に有用な低次元embeddingをモデル自身に獲得させる点で異なる（§II, §III-B）。また、Xu et al.（参考文献[1]）などのビジョンベース物理特徴学習手法は、環境（ランプ等）を用いた構造化された動的相互作用の観測に依存するが、本研究はin-hand操作のみで物理特性を抽出する点で異なる（§I, §II）。関連研究として、接触フィードバック依存タスク（slip control [16][17][18], regrasping [19][20][21], contour following [22][23][24], ball manipulation [25]）は主に静的・準静的な相互作用を扱うのに対し、本研究はswing-upという動的操作における物理特性の役割に着目している点が新規性として述べられている（§II）。

### どのように訓練・最適化したのか？
- **損失関数 / 最適化目的**: 明示的な損失関数の数式は本文中に記載がない。パイプライン全体（fusionモデル＋prediction model）を、実際にswing-upを実行して得られた最終角度を教師信号として、end-to-endの自己教師あり学習で学習する（§III-C）。
- **データセット**: モジュール式テンプレート物体（handle・rack・weightの3部品を組み合わせ可能、Fig. 5）を用い、33種類の物体（表面摩擦3種：foam, slick tape, plastic／円盤質量3種：3.7g, 7.3g, 14.5g／棒状rackで重心77-134mm・慣性モーメント0.03-0.58 g·m²の変動）を作成した。
  - **Seen（既知）物体データセット**: 33物体×各50試行、90%/10%で分割。訓練1485サンプル（33物体）、テスト165試行（33物体）。
  - **Unseen（未見）物体データセット**: 33物体を27物体（訓練）と6物体（テスト、Table IIIのID 1-6）に分割。テストセットは2種類の摩擦と2箇所に配置した2種類の質量の組み合わせ。訓練1350サンプル、テスト300試行。
  - 各データ収集試行では、0度で把持→20度・45度にtilting（マーカー情報記録）→0度に戻しshaking（マーカー系列記録）→ランダムな制御パラメータでswing-up実行→最終角度を教師ラベルとして記録、というプロセスをとる（§IV-D）。

### どのように検証したか？指標と結果は？
- **実験プラットフォーム**: 5自由度ロボットアーム（ReactorX 150, Interbotix）にGelSightセンサを搭載したグリッパーを使用。DYNAMIXEL XM-430-W350Tサーボ、OpenCM9.04マイコンで制御（§IV-A）。
- **モデルバリアント比較**（Table I, 指標：最終角度予測誤差[度]）:
  - *None*（触覚情報なし、平均値のみ出力）: Seen 25.4度 / Unseen 26.8度
  - *PP*（物理量の真値を直接入力）: Seen 11.0度 / Unseen 18.5度
  - *Tilting*のみ: Seen 13.3度 / Unseen 17.6度
  - *Shaking*のみ: Seen 10.9度 / Unseen 15.0度
  - *Combined*（提案手法、tilting+shaking融合）: Seen **10.2度** / Unseen **12.9度**（最良）
  - *Random*（ベースライン）: Seen 66.7度 / Unseen 66.7度
  - CombinedはNoneに対し両データセットで13度以上改善し、PP（物理量の真値を使うベースライン）にも最大5度上回った（§IV-B）。著者はこれを「理想的物理モデルに基づくPPの真値情報が、ゲルの弾性や物体姿勢など他の物理的要因を見落としている可能性」によるものと考察している。
- **物理特徴のdisentanglement**（Table II, 3層MLPで埋め込みから質量・重心・慣性モーメントを回帰、摩擦は3クラス分類）:
  - Combined: Seen摩擦分類94.8% / Unseen摩擦分類**93.9%**、Unseen質量誤差0.200、重心誤差0.099、慣性モーメント誤差**0.117**（Tilting・Shakingより良好、Massのみ Tiltingの0.184がやや低い）
  - End-to-End（パイプライン全体を物理量回帰用に再学習した参照上限）: Unseen摩擦分類95.4%、質量誤差0.073、重心誤差0.110、慣性モーメント誤差0.095
  - 全バリアントがRandomベースライン（33.3%分類/0.333誤差）を上回り、学習された埋め込みに全物理特性の情報が含まれることを示した（§IV-C）。
  - Tiltingは質量・重心の推定に優れ（Unseenでshakingに対し質量+8%, 重心+4%良好）、shakingは摩擦分類に優れる（本文記述では93.9% vs tiltingを+15%上回るとされるが、Table IIのUnseen摩擦分類の実数値はTilting 75.6% / Shaking 90.1% / Combined 93.9%であり、本文の「+15%」はshaking対tiltingの単純な数値差（18.3pt）とは厳密には一致しない）ことが確認された（§IV-C, Table II）。
- **タスク指向物理特徴の可視化**（Fig. 7）: PCAで2次元投影した物理embeddingにおいて、類似した制御方策分布を持つ物体（例：物体5と6）は埋め込み空間上でも近接し、方策分布が大きく異なる物体（例：物体1と4）は埋め込み空間上でも離れていることを確認した（§IV-D）。
- **Swing-up実タスク性能**（Table III）: 6つの未見テスト物体それぞれについて、目標角度45°/90°/135°/180°の4種×各5回、計20試行を実施。平均誤差は物体ごとに8.3〜23.4度で、**全体平均17.2度**であった。著者は、質量が軽い物体ほど不確実性が小さく性能が良いと報告している（§IV-E）。

### 検証結果に基づいた議論、明らかになった課題はあるか？
（§V Discussion and Future Work より）
- 本研究の埋め込み分析は、単一タスク（swing-up）の性能に基づいており、使用したハードウェアに強く条件付けられている（"heavily conditioned by the available hardware"）という限界がある。
- 使用したロボットプラットフォームはアクチュエーションノイズが大きく、swing角度予測の誤差を増大させている。
- 現在のGelSightセンサはセンシングのレイテンシがあり、swing動作全体をリアルタイムに観測することを妨げているため、本研究では開ループ制御を採用した。より低レイテンシのGelSightセンサとロバストなロボットシステムを組み合わせれば、リアルタイムフィードバック制御が可能になりうると述べている。
- Future workとして、(1) 得られたembeddingの質を利用して最適な探索行動自体を学習すること、(2) あるタスクで学習したembeddingが他タスクに転用可能かを評価すること、の2方向が挙げられている。

---
## 自身の研究との関連
本論文は、物体を能動的に動かして（tilting/shaking）得られる触覚時系列から、物理パラメータを明示的に同定せずにタスク遂行に有用な低次元表現を自己教師ありで学習するという構図を提示しており、「リアルタイムでの物理特性推定・励起（excitation）」という観点からは、excitation行動の設計とその情報利得を扱った先行例として参照できる。特に、(1) tiltingとshakingという異なる励起入力がそれぞれ異なる物理量（質量・重心 vs 摩擦）に選択的に感度を持つという実験結果（Table II, §IV-C）は、励起信号の設計がどの物理パラメータの可観測性を高めるかを左右するという一般的な知見として関連する。(2) ただし本研究は、著者自身が明記する通りswing-upという単一の動的タスクの性能のみで埋め込みの良し悪しを評価しており、リアルタイム性についてはGelSightのセンシングレイテンシを理由に明示的に断念し開ループ制御にとどめている（§V）。したがって、リアルタイム性を志向する研究との差分は、(a) オンライン・閉ループでの励起・推定ではなく、探索フェーズと実行フェーズが分離された開ループパイプラインである点、(b) 物理パラメータの直接推定ではなくタスク特化した埋め込みを学習目的としている点、の2点に整理できる。物理パラメータの直接回帰が可能であることをdisentanglement実験で事後的に示している（Table II）点は、タスク指向embeddingと物理量ベース表現の橋渡しとして参考になりうる。

---
## 追加議論

---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>
```bibtex
@inproceedings{wang2020swingbot,
  author={Wang, Chen and Wang, Shaoxiong and Romero, Branden and Veiga, Filipe and Adelson, Edward},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={SwingBot: Learning Physical Features from In-hand Tactile Exploration for Dynamic Swing-up Manipulation},
  year={2020},
  pages={5633-5640},
  doi={10.1109/IROS45743.2020.9341006}
}
```
</details>
