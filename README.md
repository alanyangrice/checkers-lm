# Draughts-QUEEN

An LM that plays Brazilian draughts and explains its moves, following the QUEEN recipe
(Bhaskar, Cheng, Chen — [arXiv 2610.03695](https://arxiv.org/abs/2610.03695)): a frozen expert
encoder trained by self-play, a frozen instruction-tuned LM, a cross-attention bridge trained on
a verifiable QA curriculum, and iterative search distillation. See `PLAN.md` for the full plan,
status, and the 5090 runbook.

## Setup

```
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q                                          # ~2 min
.venv/bin/python scripts/crosscheck.py --games 300 --positions 6000     # native rules vs pydraughts
.venv/bin/python scripts/crosscheck_edges.py --per-kind 2000            # constructed edge cases
```

Tablebases (`data/tb/`, gitignored, 11 MB) are needed for self-play gates and exact endgame play:

```
.venv/bin/python -c "from dq.tablebase import Tablebase; t=Tablebase(); t.build(4, verbose=True); t.save('data/tb')"
```

Self-play (see `scripts/selfplay.py` for the full flag list):

```
.venv/bin/python scripts/selfplay.py runs/gen1 --rounds 2000 --workers 8 --games-per-worker 32 \
    --sims 200 --round-seconds 120 --train-steps 500 --tb data/tb
tail -f runs/gen1/log.jsonl
```

## Layout

| Path | What |
|---|---|
| `src/dq/rules.py` | Native Brazilian rules: `Position`, `Move`, `legal_moves`, `capture_sequences`, `max_capture_filter`, `apply_move`, `zobrist`, FEN I/O, POV rotation. ~14 µs per position. |
| `src/dq/ref_pydraughts.py` | pydraughts wrapped to the same types. Reference oracle only (~1.8 ms per position). |
| `src/dq/tokens.py` | Board tokens (5-symbol, POV) for the encoder; `<SQ_n>` / `<PIECE_xx>` text tokens, move↔text, POV move conversion. |
| `src/dq/pdn.py` | lidraughts PDN reader + replayer (resolves ambiguous short-form captures by continuation). |
| `src/dq/tablebase.py` | Retrograde WDL/DTM tablebases, POV-normalised, units by (pieces, men). Pure Python; ≤3 pieces in seconds, 4 in minutes. |
| `src/dq/game.py` | Game loop with lidraughts draw adjudication (rules 7.2.3–7.2.8). |
| `src/dq/mcts.py` | PUCT MCTS; `search_batch` runs many trees in lockstep with virtual loss and one batched network call per round; tablebase leaves; tree reuse. |
| `src/dq/model.py` | Encoder: 32-token transformer, policy over (from,to) pairs, WDL + max-capture heads; `NetEvaluator` (batched). |
| `src/dq/selfplay.py` | Self-play: GPU workers with G concurrent games each (no per-leaf IPC), trainer with replay buffer, gating, weight publishing by file. |
| `src/dq/verify.py` | Verification harness: QA question kinds with gold answers, parsers and checkers; explanation-structure checks. |
| `bot/` | lidraughts-bot homemade engine (`dq_engine.py`) + config template; see the module docstring for setup. |
| `scripts/` | `crosscheck.py`, `crosscheck_edges.py`, `bench.py`, `replay_pdn.py` (replay a lidraughts export), `selfplay.py` (training CLI). |
| `tests/` | 194 tests: rules, pydraughts cross-checks, PDN replay, tablebases, game loop, MCTS, model, self-play, bot. |
| `data/lidraughts_brazilian.pdn` | 1,030 rated Brazilian games from top lidraughts players (Oct 2026); all replay through `rules.py`. |

## Conventions

- **Squares** 1–32, lidraughts numbering: 1 = b8, 4 = h8, 5 = a7, 29 = a1, 32 = g1. Black men start on 1–12, White on 21–32. White moves first, toward square 1. `square_to_alg` / `alg_to_square` convert to a1–h8.
- **FEN** `W:W21,22,K23:B1,2` (lidraughts style). Ranges (`21-32`) and algebraic squares are accepted on input.
- **Moves** are the full landing path: `21-17`, `22x13x6`. Short PDN forms (`22x6`) and algebraic (`d6xd2`) are accepted by `parse_move` when unambiguous. With flying kings, different landing squares are different moves.
- **Colours** `WHITE = 0`, `BLACK = 1`. `Position.pov()` rotates the board 180° and swaps colours so the side to move is always White; the encoder always sees the POV board.
- **Terminal**: no legal move = loss. Draw adjudication lives in `game.py`, not `rules.py`.
- **Results** in `game.py` are from White's view: `1.0` / `0.5` / `0.0`.

## Verification harness

Every question kind in `dq.verify.KINDS` has `gold`, `render`, `parse`, and `check`. Any kind takes an optional `moves` prefix, which turns a `*-CURRENT` question into its `*-FUTURE` form.

| Stage | Kinds |
|---|---|
| STATIC-CURRENT | `piece_on_square`, `locate_kings`, `count`, `tempo`, `board` |
| DYNAMIC-CURRENT | `legal_moves_for_piece`, `legal_moves`, `capture_sequences` (all vs maximal), `threatened`, `safe_king_moves` |

`check_explanation(pos, text)` verifies `BEST_MOVE`, `CRITICAL_LINE`, `PROMISING_MOVES`, `EVALUATION` for legality and well-formedness.
