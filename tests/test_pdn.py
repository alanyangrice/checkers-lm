from pathlib import Path

import pytest

from dq.pdn import consistent_result, read_pdn, replay
from dq.rules import BLACK, WHITE, Position, legal_moves

DATA = Path(__file__).resolve().parents[1] / "data" / "lidraughts_brazilian.pdn"

SAMPLE = """[Event "Rated Brazilian game"]
[Site "https://lidraughts.org/hBqxo8uO"]
[Result "2-0"]
[Termination "Normal"]
[FEN "W:Wa3,c3,e3,g3,b2,d2,f2,h2,a1,c1,e1,g1:Bb8,d8,f8,h8,a7,c7,e7,g7,b6,d6,f6,h6"]

1. c3-b4 f6-e5 2. b4-a5 g7-f6 3. e3-f4 b6-c5 4. d2-e3 c7-b6 5. a5xc7 d8xb6 6. c1-d2 b6-a5 7. g3-h4 e5xg3 8. h2xf4 f6-e5 9. f4-g5 h6xf4 10. e3xg5 a7-b6 11. g5-h6 b8-a7 12. g1-h2 h8-g7 13. d2-c3 g7-f6 14. h4-g5 f6xh4 15. c3-b4 a5xc3 16. b2xd8 b6-a5 17. d8-f6 a7-b6 18. f6-c3 a5-b4 19. c3xe5 2-0
"""


def test_parse_and_replay_sample() -> None:
    games = read_pdn(SAMPLE)
    assert len(games) == 1
    g = games[0]
    assert g.result == WHITE and len(g.moves) == 37
    assert Position.from_fen(g.headers["FEN"]) == Position.start()
    rep = replay(g, keep=True)
    assert rep.ok and rep.plies == 37 and len(rep.positions) == 38
    assert rep.final.fen() == "B:W12,K15,21,27,28,29,31:B3,14,20"


def test_ambiguous_short_form_resolved_by_continuation() -> None:
    # 15. d4xd4 is a four-piece circular capture that can be taken clockwise
    # or anticlockwise (18x9x2x11x18 / 18x11x2x9x18); lidraughts writes only
    # start and end.  Real game, non-standard (balloted) start position.
    pdn = """[Site "https://lidraughts.org/R7v78zbE"]
[Result "2-0"]
[Termination "Normal"]
[FEN "W:Wd4,h4,a3,e3,b2,d2,f2,h2,a1,c1,e1,g1:Bb8,d8,f8,h8,a7,c7,e7,g7,b6,f6,c5,g5"]

1. f2-g3 c7-d6 2. b2-c3 b6-a5 3. d4xb6 a5xc7 4. a3-b4 a7-b6 5. b4-a5 g7-h6 6. a1-b2 d6-e5 7. b2-a3 e7-d6 8. g3-f4 e5xg3 9. h2xf4 f8-g7 10. c1-b2 b6-c5 11. c3-d4 b8-a7 12. d4xb6 a7xc5 13. d2-c3 d8-e7 14. c3-d4 f6-e5 15. d4xd4 d6-c5 16. d4xb6 2-0
"""
    g = read_pdn(pdn)[0]
    rep = replay(g)
    assert rep.ok
    assert len(rep.played[28].captured) == 4 and rep.played[28].start == rep.played[28].end == 18


@pytest.mark.skipif(not DATA.exists(), reason="lidraughts export not present")
def test_lidraughts_games_replay() -> None:
    games = read_pdn(DATA.read_text())
    assert len(games) >= 500
    bad = []
    checked = 0
    for g in games:
        rep = replay(g)
        if not rep.ok:
            bad.append((g.site, rep.error))
            continue
        cr = consistent_result(g, rep)
        if cr is False:
            bad.append((g.site, "result mismatch"))
        elif cr is True:
            checked += 1
    assert not bad, bad[:5]
    assert checked >= 100
