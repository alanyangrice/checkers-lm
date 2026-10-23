"""Keep the lidraughts BOT busy: when it is idle, challenge an online bot to a rated Brazilian game.

    python bot/matchmaker.py --token-file ../lidraughts-bot/config.yml [--clock 300 3] [--every 120]

lidraughts-bot (the client) accepts challenges and plays the games; this fork has no
matchmaking of its own, so this loop issues outgoing challenges.  Candidate opponents are the
accounts listed on https://lidraughts.org/player/bots that are online and have a Brazilian
rating.  Humans can still challenge the bot directly at any time.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time

import requests

BASE = "https://lidraughts.org"


def token_from_config(path: str) -> str:
    m = re.search(r'token:\s*"([^"]+)"', open(path, encoding="utf-8").read())
    if not m:
        raise SystemExit("no token in " + path)
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", default="../lidraughts-bot/config.yml")
    ap.add_argument("--clock", type=int, nargs=2, default=(300, 3), metavar=("SECONDS", "INCREMENT"))
    ap.add_argument("--every", type=int, default=120, help="seconds between checks")
    ap.add_argument("--rated", action="store_true", default=True)
    a = ap.parse_args()
    tok = token_from_config(a.token_file)
    s = requests.Session(); s.headers["Authorization"] = f"Bearer {tok}"
    me = s.get(f"{BASE}/api/account", timeout=30).json()["username"]
    print(f"matchmaker for {me}: {a.clock[0]}+{a.clock[1]} rated brazilian, every {a.every}s", flush=True)
    recent: dict[str, float] = {}
    while True:
        try:
            playing = s.get(f"{BASE}/api/account/playing", timeout=30).json().get("nowPlaying", [])
            if playing:
                print(f"in game vs {playing[0].get('opponent', {}).get('username')}", flush=True)
                time.sleep(a.every); continue
            html = s.get(f"{BASE}/player/bots", timeout=30).text
            bots = [b for b in dict.fromkeys(re.findall(r'href="/@/([^"]+)"', html)) if b.lower() != me.lower()]
            status = s.get(f"{BASE}/api/users/status", params={"ids": ",".join(bots)}, timeout=30).json()
            online = [u["name"] for u in status if u.get("online")]
            cands = []
            for b in online:
                try:
                    perf = s.get(f"{BASE}/api/user/{b}", timeout=30).json().get("perfs", {}).get("brazilian")
                except Exception:
                    perf = None
                if perf and perf.get("games", 0) > 0 and time.time() - recent.get(b, 0) > 900:
                    cands.append((b, perf.get("rating")))
            if not cands:
                print(f"no eligible online bot ({len(online)} online of {len(bots)}); waiting", flush=True)
                time.sleep(a.every); continue
            opp, rating = random.choice(cands)
            r = s.post(f"{BASE}/api/challenge/{opp}", timeout=30,
                       data={"rated": "true" if a.rated else "false", "clock.limit": a.clock[0],
                             "clock.increment": a.clock[1], "variant": "brazilian", "color": "random"})
            recent[opp] = time.time()
            print(f"challenged {opp} ({rating}): {r.status_code} {r.text[:120]}", flush=True)
        except Exception as e:
            print(f"error: {e}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
