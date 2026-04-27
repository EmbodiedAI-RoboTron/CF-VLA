import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = REPO_ROOT / "configs/pi05_2stg_pytorch_delta_actions/exp3_2_base_lr_pi05_b16_norm_false_horizon50.yaml"


def test_train_config_loads() -> None:
    assert TRAIN_CONFIG.exists()
    config = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert "model" in config
    assert "data" in config
    assert "batch_size" in config


def test_train_launcher_usage_message() -> None:
    cmd = ["bash", "scripts/run_train_pi0_pytorch.sh"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "Usage:" in output


def test_core_scripts_compile() -> None:
    for rel_path in (
        "scripts/train_pytorch.py",
        "scripts/run_train_pi0_pytorch.sh",
        "scripts/serve_policy.py",
    ):
        path = REPO_ROOT / rel_path
        if path.suffix == ".py":
            subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)
