# Infrastructure Log

## 2026-06-30: uv から pixi への環境管理移行

### 背景

uv による環境管理を pixi に移行した.
direnv との統合により, ディレクトリ移動時に自動で環境が有効化されるようにした.

### 実施内容

**パッケージマネージャ移行**
- `pixi init --format pyproject` で `pyproject.toml` に `[tool.pixi.*]` セクションを追加
- `uv.lock`, `requirements.txt`, `.venv/` を削除
- `pixi.lock` を生成

**pyproject.toml の修正**
- `[project.scripts]` セクションを追加(`generate-trajectory` エントリポイントが `egg-info` にしか存在していなかった)
- `[tool.setuptools.packages.find]` を追加(flat-layout での setuptools パッケージ検出を明示化)

**direnv 統合**
- `.envrc` を作成し, ディレクトリ移動時に pixi 環境が自動有効化されるよう設定

**README 更新**
- `pip install -r requirements.txt` / `uv run` の記述を `pixi install` / `pixi run` に置き換え

**自動生成ファイル**
- `.gitattributes`(pixi が `pixi.lock` のマージ設定のために自動生成)

### 結果

- `pixi run -e dev python -m pytest tests/` で全テスト通過
- `generate-trajectory` CLI が pixi 環境下で正常動作
