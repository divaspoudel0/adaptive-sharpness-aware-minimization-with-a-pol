"""Reproduction entrypoint. Reads config.yaml and dispatches to the audit or DNN run.

Usage: python repro.py
"""
import json
import sys

import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
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
