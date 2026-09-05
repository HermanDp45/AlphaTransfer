#!/usr/bin/env python3
"""Default standalone TabM H3, with explicit legacy/research entry points."""
import sys
from pathlib import Path

SOLUTION_ROOT=Path(__file__).resolve().parent
if "--research-v3" in sys.argv:
    sys.argv.remove("--research-v3")
    sys.path.insert(0,str(SOLUTION_ROOT.parent))
    from research_v3.preview import main
elif "--legacy" in sys.argv:
    sys.argv.remove("--legacy")
    if not any(x=="--config" or x.startswith("--config=") for x in sys.argv[1:]):
        sys.argv.extend(["--config",str(SOLUTION_ROOT/"config.legacy.toml")])
    from alphatransfer_final.cli import main
else:
    sys.path.insert(0,str(SOLUTION_ROOT.parent))
    from final_solution.tabm_h3.predict import main

if __name__=="__main__":
    raise SystemExit(main())
