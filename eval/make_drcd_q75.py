#!/usr/bin/env python3
"""
Build the DRC-D quality-75 testbed: every DRC-D train and test image re-encoded
once as JPEG at quality 75, so the source block grid is strong. The paper's
DRC-D models never learned the (faint, quality ~95) grid from 1,474 images; if
a plain model learns the strong one, the quality-75 sets can screen
trace-removal recipes for shortcut removal in ten-minute runs.

Writes data/datasets/train_drcd_q75 and data/datasets/test_drcd_q75.
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
QUALITY = int(sys.argv[1]) if len(sys.argv) > 1 else 75

for split in ("train_drcd", "test_drcd"):
    src = ROOT / "data/datasets" / split
    dst = ROOT / "data/datasets" / f"{split}_q{QUALITY}"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.jpg")):
        out = dst / p.name
        if not out.exists():
            Image.open(p).convert("RGB").save(out, "JPEG", quality=QUALITY)
        n += 1
    print(f"{dst}: {n} images at quality {QUALITY}")
