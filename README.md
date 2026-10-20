# Draughts-QUEEN

An LM that plays Brazilian draughts and explains its moves. See `PLAN.md`.

## Setup

```
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/crosscheck.py --games 300 --positions 6000   # native rules vs pydraughts
```

## Layout

| Path | What |
|---|---|
| `src/dq/rules.py` | Native Brazilian rules: `Position`, `Move`, `legal_moves`, `capture_sequences`, `max_capture_filter`, `apply_move`, `zobrist`, FEN I/O, POV rotation. ~14 µs per position. |
| `src/dq/ref_pydraughts.py` | pydraughts wrapped to the same types. Reference oracle only (~1.8 ms per position). |
| `src/dq/verify.py` | Verification harness: QA question kinds with gold answers, parsers and checkers; explanation-structure checks. |
| `tests/` | Unit tests, pydraughts cross-check, harness tests. |
| `scripts/` | `crosscheck.py` (long cross-check), `bench.py`. |

## Conventions

- **Squares** 1–32, lidraughts numbering: 1 = b8, 4 = h8, 5 = a7, 29 = a1, 32 = g1. Black men start on 1–12, White on 21–32. White moves first, toward square 1. `square_to_alg` / `alg_to_square` convert to a1–h8.
- **FEN** `W:W21,22,K23:B1,2` (lidraughts style). Ranges (`21-32`) and algebraic squares are accepted on input.
- **Moves** are the full landing path: `21-17`, `22x13x6`. Short PDN forms (`22x6`) and algebraic (`d6xd2`) are accepted by `parse_move` when unambiguous. With flying kings, different landing squares are different moves.
- **Colours** `WHITE = 0`, `BLACK = 1`. `Position.pov()` rotates the board 180° and swaps colours so the side to move is always White.
- **Terminal**: no legal move = loss. Draw adjudication is not in `rules.py`.

## Verification harness

Every question kind in `dq.verify.KINDS` has `gold`, `render`, `parse`, and `check`. Any kind takes an optional `moves` prefix, which turns a `*-CURRENT` question into its `*-FUTURE` form.

| Stage | Kinds |
|---|---|
| STATIC-CURRENT | `piece_on_square`, `locate_kings`, `count`, `tempo`, `board` |
| DYNAMIC-CURRENT | `legal_moves_for_piece`, `legal_moves`, `capture_sequences` (all vs maximal), `threatened`, `safe_king_moves` |

`check_explanation(pos, text)` verifies `BEST_MOVE`, `CRITICAL_LINE`, `PROMISING_MOVES`, `EVALUATION` for legality and well-formedness.
