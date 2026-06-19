from pathlib import Path
import yaml
import os

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


def load_config(name: str) -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 支持环境变量替换
    target_cfg = cfg.get(name, {})
    if(name == "ocr" or name == "agent"):
        if "api_key" in target_cfg and isinstance(target_cfg["api_key"], str):
            target_cfg["api_key"] = os.path.expandvars(target_cfg["api_key"])

    return target_cfg
