from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Returns the VNB-Grasp repository root (directory containing README.md)"""
    # vnb_grasp/robosuite_ext/paths.py -> vnb_grasp/robosuite_ext -> vnb_grasp -> repo root
    return Path(__file__).resolve().parents[2]
