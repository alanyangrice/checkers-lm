"""Speed of native legal-move generation vs pydraughts on random midgame positions."""
import random
import time

from dq.ref_pydraughts import ref_legal_moves
from dq.rules import legal_moves, random_position

rng = random.Random(0)
poss = [random_position(rng, rng.randint(4, 12), rng.randint(4, 12), king_prob=0.2) for _ in range(300)]
for name, fn in (("native", legal_moves), ("pydraughts", ref_legal_moves)):
    t = time.perf_counter()
    for p in poss:
        fn(p)
    dt = time.perf_counter() - t
    print(f"{name:10s} {dt / len(poss) * 1e6:8.0f} us/position")
