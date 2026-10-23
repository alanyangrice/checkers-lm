"""Keep the lidraughts BOT busy: when it is idle, challenge an online bot to a rated Brazilian game.

    python bot/matchmaker.py --token-file ../lidraughts-bot/config.yml [--every 45]

lidraughts-bot (the client) accepts challenges and plays the games; this fork has no
matchmaking of its own, so this loop issues outgoing challenges.  Candidate opponents are the
accounts listed on https://lidraughts.org/player/bots that have a Brazilian rating.

Behaviour:
  * one outstanding challenge at a time; cancelled if not accepted within --accept-timeout s
  * time control rotates through --clocks (bots often accept only some speeds)
  * opponents that accepted before are preferred and re-challenged after a short cooldown;
    opponents that ignored/declined get a long cooldown
Humans can still challenge the bot directly at any time.
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
    ap.add_argument("--clocks", nargs="+", default=["300+3", "180+2", "600+0", "300+0", "120+1"],
                    help="time controls to rotate through, as base+increment seconds")
    ap.add_argument("--every", type=int, default=45, help="seconds between checks")
    ap.add_argument("--accept-timeout", type=int, default=75, help="cancel a challenge not accepted within this many seconds")
    ap.add_argument("--cooldown-good", type=int, default=20, help="re-challenge an opponent who accepted before after this")
    ap.add_argument("--cooldown-bad", type=int, default=600, help="skip an opponent who ignored a challenge for this long")
    a = ap.parse_args()
    clocks = [tuple(int(x) for x in c.split("+")) for c in a.clocks]
    tok = token_from_config(a.token_file)
    s = requests.Session(); s.headers["Authorization"] = f"Bearer {tok}"
    me = s.get(f"{BASE}/api/account", timeout=30).json()["username"]
    print(f"matchmaker for {me}: rated brazilian, clocks {a.clocks}, every {a.every}s", flush=True)

    next_ok: dict[str, float] = {}         # opponent -> earliest time to challenge again
    accepted: dict[str, int] = {}          # opponent -> games accepted
    pending: tuple[str, str, float] | None = None   # (challenge id, opponent, created at)
    ci = 0
    games_seen = 0
    while True:
        try:
            playing = s.get(f"{BASE}/api/account/playing", timeout=30).json().get("nowPlaying", [])
            if playing:
                opp = playing[0].get("opponent", {}).get("username", "?").replace("BOT ", "")
                if pending and pending[1].lower() == opp.lower():
                    accepted[opp] = accepted.get(opp, 0) + 1; pending = None
                print(f"in game vs {opp}", flush=True)
                time.sleep(a.every); continue
            if pending:
                cid, opp, t0 = pending
                if time.time() - t0 > a.accept_timeout:
                    s.post(f"{BASE}/api/challenge/{cid}/cancel", timeout=30)
                    next_ok[opp] = time.time() + a.cooldown_bad
                    print(f"cancelled unanswered challenge to {opp}", flush=True)
                    pending = None
                else:
                    time.sleep(a.every); continue
            acct = s.get(f"{BASE}/api/account", timeout=30).json()
            n = acct.get("count", {}).get("all", 0)
            if n != games_seen:
                p = acct.get("perfs", {}).get("brazilian", {})
                print(f"games {n}: W{acct['count'].get('win')} D{acct['count'].get('draw')} L{acct['count'].get('loss')} "
                      f"brazilian {p.get('rating')}{'?' if p.get('prov') else ''}", flush=True)
                games_seen = n
            html = s.get(f"{BASE}/player/bots", timeout=30).text
            bots = [b for b in dict.fromkeys(re.findall(r'href="/@/([^"]+)"', html)) if b.lower() != me.lower()]
            status = s.get(f"{BASE}/api/users/status", params={"ids": ",".join(bots)}, timeout=30).json()
            online = [u["name"] for u in status if u.get("online")]
            cands = []
            for b in online:
                if time.time() < next_ok.get(b, 0):
                    continue
                try:
                    perf = s.get(f"{BASE}/api/user/{b}", timeout=30).json().get("perfs", {}).get("brazilian")
                except Exception:
                    perf = None
                if perf and perf.get("games", 0) > 0:
                    cands.append((b, perf.get("rating")))
            if not cands:
                print(f"no eligible bot right now ({len(online)} online, all cooling down); waiting", flush=True)
                time.sleep(a.every); continue
            # prefer opponents that have accepted before
            cands.sort(key=lambda x: -accepted.get(x[0], 0))
            opp, rating = cands[0] if accepted.get(cands[0][0]) else random.choice(cands)
            base, inc = clocks[ci % len(clocks)]; ci += 1
            r = s.post(f"{BASE}/api/challenge/{opp}", timeout=30,
                       data={"rated": "true", "clock.limit": base, "clock.increment": inc,
                             "variant": "brazilian", "color": "random"})
            if r.status_code == 200 and "challenge" in r.json():
                cid = r.json()["challenge"]["id"]
                pending = (cid, opp, time.time())
                next_ok[opp] = time.time() + a.cooldown_good
                print(f"challenged {opp} ({rating}) {base}+{inc}: {cid}", flush=True)
            else:
                next_ok[opp] = time.time() + a.cooldown_bad
                print(f"challenge to {opp} failed: {r.status_code} {r.text[:100]}", flush=True)
        except Exception as e:
            print(f"error: {e}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
