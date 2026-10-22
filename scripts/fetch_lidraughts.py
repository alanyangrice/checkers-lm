"""Download rated Brazilian games of the top lidraughts players (public API, polite pacing).

    python scripts/fetch_lidraughts.py --out data/lidraughts_brazilian_top.pdn --max-per-user 1500

Usernames come from the public leaderboard page; games from /api/games/user/{name} (PDN).
One request at a time, 2 s between users, 65 s back-off on HTTP 429 (lichess/lidraughts rule).
Games are de-duplicated by their Site URL, including against --merge files.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
import urllib.error

UA = "checkers-lm research fetch (contact: github.com/alanyangrice/checkers-lm)"


def get(url: str, accept: str) -> str:
    import http.client
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8", "replace")
        except http.client.IncompleteRead as e:
            # the server closes long streams early; keep what arrived (whole games are re-split by the caller)
            print(f"  stream cut after {len(e.partial)} bytes; keeping partial", flush=True)
            return e.partial.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("  429: backing off 65 s", flush=True); time.sleep(65); continue
            if e.code == 404:
                return ""
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  {type(e).__name__}: retrying in 20 s", flush=True); time.sleep(20)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lidraughts_brazilian_top.pdn")
    ap.add_argument("--merge", nargs="*", default=["data/lidraughts_brazilian.pdn"], help="existing PDN files to de-dup against")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--max-per-user", type=int, default=1500)
    ap.add_argument("--min-rating", type=int, default=0, help="skip games where either player is below this")
    ap.add_argument("--skip", type=int, default=0, help="resume: skip the first N players of the leaderboard")
    a = ap.parse_args()

    html = get(f"https://lidraughts.org/player/top/{a.top}/brazilian", "text/html")
    names = list(dict.fromkeys(re.findall(r'href="/@/([^"]+)"', html)))
    print(f"{len(names)} players", flush=True)

    seen: set[str] = set()
    for f in list(a.merge) + [a.out]:          # never re-add games already in the output file
        try:
            seen |= set(re.findall(r'\[Site "([^"]+)"\]', open(f, encoding="utf-8").read()))
        except FileNotFoundError:
            pass
    print(f"{len(seen)} games already on disk", flush=True)

    n_new = 0
    with open(a.out, "a", encoding="utf-8") as out:
        for i, u in enumerate(names):
            if i < a.skip:
                continue
            url = (f"https://lidraughts.org/api/games/user/{u}?perfType=brazilian&rated=true"
                   f"&max={a.max_per_user}&moves=true&tags=true&opening=false")
            txt = get(url, "application/x-chess-pgn")
            games = [g for g in re.split(r"\n\n(?=\[Event )", txt.strip()) if g.strip()]
            added = 0
            for g in games:
                m = re.search(r'\[Site "([^"]+)"\]', g)
                if not m or m.group(1) in seen:
                    continue
                if a.min_rating:
                    elos = [int(x) for x in re.findall(r'\[(?:White|Black)Elo "(\d+)"\]', g)]
                    if len(elos) < 2 or min(elos) < a.min_rating:
                        continue
                seen.add(m.group(1)); out.write(g.strip() + "\n\n"); added += 1
            out.flush(); n_new += added
            print(f"[{i+1}/{len(names)}] {u}: {len(games)} fetched, {added} new (total new {n_new})", flush=True)
            time.sleep(2.0)
    print(f"done: {n_new} new games -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
