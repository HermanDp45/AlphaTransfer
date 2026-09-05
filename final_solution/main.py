#!/usr/bin/env python3
"""Single entry point for the AlphaTransfer reference solution."""

import sys
from pathlib import Path

if "--research-v3" in sys.argv:
    sys.argv.remove("--research-v3")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_v3.preview import main
else:
    from alphatransfer_final.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
