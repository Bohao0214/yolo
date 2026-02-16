from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d3.py
# Purpose: enhance241 d3 canonical entry; keeps backward compatibility through d1 implementation.

from .yolo11_241d1 import ENHANCE241_AUDIT_KEYS, P3LogitTemperature, apply

__all__ = ["ENHANCE241_AUDIT_KEYS", "P3LogitTemperature", "apply"]
