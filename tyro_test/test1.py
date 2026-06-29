import dataclasses
from pathlib import Path
from typing import Optional
import tyro
import yaml


@dataclasses.dataclass(frozen=True)
class CommandConfig:
    """コマンド設定"""
    show_log: int = 1  # 1. ハードコードされたデフォルト値
    target_name: str = "robot"
    config: Optional[Path] = None
    #config: Annotated[Optional[Path], tyro.conf.arg(name="config")] = None
    """Path to YAML config file"""


def main():
    # 1. 最初に一度パースして --config の値を取得
    initial_config = tyro.cli(CommandConfig)

    # 2. config が指定されていれば YAML を読み込んでデフォルト値として再パース
    if initial_config.config is not None and initial_config.config.exists():
        with open(initial_config.config, "r") as f:
            yaml_data = yaml.safe_load(f)
        # YAML の値でデフォルトインスタンスを作成
        yaml_defaults = CommandConfig(**yaml_data)
        # YAML のデフォルト値で再パース（CLI 引数で上書き可能）
        final_config = tyro.cli(CommandConfig, default=yaml_defaults)
    else:
        final_config = initial_config

    print(f"最終的な設定: {final_config}")


if __name__ == "__main__":
    main()