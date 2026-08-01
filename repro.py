"""Reproduction entrypoint. Reads config.yaml and dispatches to the audit or DNN run.

Usage: python repro.py
"""
import json
import os
import subprocess
import sys


def _ensure_deps():
    """Bootstrap deps when the base image lacks them (keeps the run command fixed)."""
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import yaml  # noqa: F401
        import huggingface_hub  # noqa: F401
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    reqs = os.path.join(here, "requirements.txt")
    print("[repro] installing requirements (missing deps)...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", reqs])


def load_config(path="config.yaml"):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    _ensure_deps()
    cfg = load_config()
    mode = cfg.get("mode", "audit")
    print(f"[repro] mode={mode} config={json.dumps(cfg)}", flush=True)
    if mode == "audit":
        from repro import audit
        audit.main()
    elif mode == "cifar":
        from repro import cifar
        cifar.main(cfg)
    elif mode == "timing":
        from repro import smoke
        smoke.main(cfg)
    else:
        print(f"[repro] unknown mode: {mode}")
        sys.exit(1)
    print("[repro] OK")


if __name__ == "__main__":
    main()
