#!/usr/bin/env python3
"""
为已有 Stage3 JSON（含 wealth_trace、尚无 evalue_cap_timing）回填检测延迟字段。

  python scripts/stage3_backfill_evalue_cap_timing.py [json_path ...]

未传参时默认处理 results/stage3_evalue_{hotpotqa,musique,2wiki}.json。
加 `--force` 时覆盖已有 `evalue_cap_timing` 字段。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage3.cap_timing import evalue_cap_timing


def backfill_one(path: Path) -> bool:
    data = json.loads(path.read_text())
    st = str(data.get("shift_type", "none"))
    sf = float(data.get("shift_fraction", 0.5))
    changed = False
    for gk, gv in list(data.get("per_gamma", {}).items()):
        for ak, av in list(gv.get("per_alpha", {}).items()):
            wt = av.get("wealth_trace")
            if not wt or (not force and "evalue_cap_timing" in av):
                continue
            cap = float(av.get("wealth_cap", 0.0))
            if cap <= 0:
                cap = 1.0 / float(ak)
            n = len(wt) - 1
            av["evalue_cap_timing"] = evalue_cap_timing(
                [float(x) for x in wt],
                cap=cap,
                n_samples=n,
                shift_type=st,
                shift_fraction=sf,
            )
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]
    if args:
        paths = [Path(p) for p in args]
    else:
        paths = [root / "results" / f"stage3_evalue_{ds}.json" for ds in ("hotpotqa", "musique", "2wiki")]
    for p in paths:
        if not p.exists():
            print("skip missing", p)
            continue
        if backfill_one(p):
            print("updated", p)
        else:
            print("unchanged", p)


if __name__ == "__main__":
    main()
