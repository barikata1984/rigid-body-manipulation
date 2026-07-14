# Literature Survey 実行ログ

## 2026-07-15: task-drift-excitation-trajectory-design

**トピック**: task 由来の大振幅ベース運動が重畳した状況下での慣性パラメータ同定のための励起軌道設計（`notes/findings/20260714T152931Z_task_drift_excitation.md` の novelty gap 確定を目的とした調査）。

**実行モード**: Auto-execution Mode（無人 batch 実行、ユーザー確認を全フェーズでフォールバック）。

**検索プロセス**:
- 既知7論文（Swevers 1997, Park 2006, Kubus 2007/2008, Bonnet 2016, Lee-Lee-Park 2021, Leboutet 2021, Annual Reviews 2024）は再調査せず、既存の深読みノートを再利用
- 未読4件の重点確認: Yun 2023 → 実際は **Park, Shin & Kim 2023**（著者名の誤りを訂正）、In-Situ Excitation Trajectory Optimizer（正式には ICIRA 2025 Proceedings、LNEE ではなく LNCS 系列、paywall で全文未確認）、Abu-Dakka 2017 IROS、Ayusawa 2017 ICRA
- OpenAlex API による前方引用スノーボーリング（6論文起点、約150件から選別）で16本の非ハブ論文を追加
- Park, Shin & Kim (2023) の PDF を新規取得・精読し、深読みノートを新規作成（`literature/papers/Park-arXiv2023-Object-Aware_Impedance_Control/`）
- 3件の補完検索サブエージェント（task-priority/null-space、online/dual-control、persistent-excitation理論/drifting-base）を dispatch したが、セッションの実効作業時間内に1件（online/dual-control）が未返却。残り2件はメインループの補完 WebSearch で部分的に代替
- Park-arXiv2023 クロス検証エージェント、reference-verify バッチエージェントも未返却。メインループ自身の直接精読・OpenAlex 照合で代替

**成果物**:
- `literature/surveys/task-drift-excitation-trajectory-design.md`（Map 26本、Hub 10本）
- `literature/papers/Park-arXiv2023-Object-Aware_Impedance_Control/object-aware-impedance-control-for-human-robot-collaborative-task-with-online-object-parameter-estimation.md`（新規深読みノート）
- `literature/references/main.md`（新規作成、26論文の書誌エントリ）

**プロジェクトへの含意**:
- novelty gap は維持される。「task 由来の非設計的な大振幅ドリフトが励起のフーリエ調波帯と周波数領域で直接競合し、観測行列を構造的に劣化させる」という機構分析と、相対振幅比 $T^2/(\text{turn 数})$ のスケーリング則は、本サーベイの範囲では先行例が確認できなかった
- 最も近い publish 済み類例は Park, Shin & Kim (2023) だが、ヌルスペース分離により干渉そのものを回避する設計であり、干渉が不可避な場合の分析は行っていない
- related work のセクション構成案: (a) 古典的フーリエ励起最適化の系譜（Swevers, Park2006, Bonnet, Lee-Lee-Park）、(b) task 制約下のオンライン励起（Albee, Zhang, Park2023）、(c) 大振幅励起回避のアプローチ（Nadeau x2, Foster）を prior art として引用した上で、本プロジェクトの機構分析を novelty として位置づける
- 残作業: 3件の補完検索サブエージェントと2件の検証エージェントが未返却のまま。後日結果が得られれば Concept Matrix / Paper Catalogue への追記、および独立クロス検証の完了が必要
