"""Replay a lidraughts Brazilian PDN export through dq.rules.

    .venv/bin/python scripts/replay_pdn.py data/lidraughts_brazilian.pdn

Reports: games parsed, games fully replayed, first failures, and how many
decisive 'Normal' terminations ended in a no-moves position whose loser
matches the recorded result (a check on square numbering / orientation).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from dq.pdn import consistent_result, read_pdn, replay


def main(path: str) -> int:
    games = read_pdn(Path(path).read_text(encoding="utf-8"))
    c: Counter[str] = Counter()
    fails = []
    for g in games:
        c["games"] += 1
        c["plies"] += len(g.moves)
        rep = replay(g)
        if not rep.ok:
            c["replay_fail"] += 1
            if len(fails) < 5:
                fails.append((g.site, rep.error))
            continue
        c["replay_ok"] += 1
        cr = consistent_result(g, rep)
        if cr is True:
            c["result_checked_ok"] += 1
        elif cr is False:
            c["result_checked_BAD"] += 1
            if len(fails) < 5:
                fails.append((g.site, f"result mismatch: recorded {g.result}, final {rep.final.fen()}"))
    for k, v in sorted(c.items()):
        print(f"{k:22s} {v}")
    for site, err in fails:
        print("FAIL", site, err)
    return 1 if (c["replay_fail"] or c["result_checked_BAD"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/lidraughts_brazilian.pdn"))
