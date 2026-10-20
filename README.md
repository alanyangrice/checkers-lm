 `src/dq/verify.py` # Draughts-QUEEN
 `src/dq/verify.py` 
 `src/dq/verify.py` An LM that plays Brazilian draughts and explains its moves. See `PLAN.md`.
 `src/dq/verify.py` 
 `src/dq/verify.py` ## Setup
 `src/dq/verify.py` 
 `src/dq/verify.py` ```
 `src/dq/verify.py` uv venv --python 3.12 && uv pip install -e ".[dev]"
 `src/dq/verify.py` .venv/bin/python -m pytest -q
 `src/dq/verify.py` .venv/bin/python scripts/crosscheck.py --games 300 --positions 6000   # native rules vs pydraughts
 `src/dq/verify.py` ```
 `src/dq/verify.py` 
 `src/dq/verify.py` ## Layout
 `src/dq/verify.py` 
 `src/dq/verify.py` | Path | What |
 `src/dq/verify.py` |---|---|
 `src/dq/verify.py` | `src/dq/rules.py` | Native Brazilian rules: `Position`, `Move`, `legal_moves`, `capture_sequences`, `max_capture_filter`, `apply_move`, `zobrist`, FEN I/O, POV rotation. ~14 µs per position. |
 `src/dq/verify.py` | `src/dq/ref_pydraughts.py` | pydraughts wrapped to the same types. Reference oracle only (~1.8 ms per position). |
 `src/dq/verify.py` | `src/dq/tokens.py` | Board tokens (5-symbol, POV) for the encoder; `<SQ_n>` / `<PIECE_xx>` text tokens, move<->text, POV move conversion. |
 `src/dq/verify.py` | `src/dq/pdn.py` | lidraughts PDN reader + replayer (resolves ambiguous short-form captures by continuation). |
 `src/dq/verify.py` | `src/dq/tablebase.py` | Retrograde WDL/DTM tablebases, POV-normalised, units by (pieces, men). Pure Python; <=3 pieces in seconds, 4 in minutes. |
 `src/dq/verify.py` | `src/dq/game.py` | Game loop with lidraughts draw adjudication (7.2.3–7.2.8). |
| `src/dq/mcts.py` | PUCT MCTS; `search_batch` runs many trees in lockstep with virtual loss and one batched network call per round; tablebase leaves. |
| `src/dq/model.py` | Encoder: 32-token transformer, policy over (from,to) pairs, WDL + max-capture heads; `NetEvaluator` (batched). |
| `src/dq/selfplay.py` | Self-play: GPU workers with G concurrent games each, trainer with replay buffer, gating and weight publishing by file. |
| `src/dq/verify.py` | Verification harness: QA question kinds with gold answers, parsers and checkers; explanation-structure checks. |
 `src/dq/verify.py` | `tests/` | Unit tests, pydraughts cross-check, harness tests. |
 `src/dq/verify.py` | `scripts/` | `crosscheck.py` (random cross-check), `crosscheck_edges.py` (targeted edge cases: max-capture ties, flying-king landings, back-rank pass-through, obstacle rings, dense king, mixed man/king), `bench.py`, `replay_pdn.py` (replay a lidraughts export), `selfplay.py` (training CLI). |
 `src/dq/verify.py` | `data/lidraughts_brazilian.pdn` | 1,030 rated Brazilian games from top lidraughts players (Oct 2026); all replay through `rules.py`. |
 `src/dq/verify.py` 
 `src/dq/verify.py` ## Conventions
 `src/dq/verify.py` 
 `src/dq/verify.py` - **Squares** 1–32, lidraughts numbering: 1 = b8, 4 = h8, 5 = a7, 29 = a1, 32 = g1. Black men start on 1–12, White on 21–32. White moves first, toward square 1. `square_to_alg` / `alg_to_square` convert to a1–h8.
 `src/dq/verify.py` - **FEN** `W:W21,22,K23:B1,2` (lidraughts style). Ranges (`21-32`) and algebraic squares are accepted on input.
 `src/dq/verify.py` - **Moves** are the full landing path: `21-17`, `22x13x6`. Short PDN forms (`22x6`) and algebraic (`d6xd2`) are accepted by `parse_move` when unambiguous. With flying kings, different landing squares are different moves.
 `src/dq/verify.py` - **Colours** `WHITE = 0`, `BLACK = 1`. `Position.pov()` rotates the board 180° and swaps colours so the side to move is always White.
 `src/dq/verify.py` - **Terminal**: no legal move = loss. Draw adjudication is not in `rules.py`.
 `src/dq/verify.py` 
 `src/dq/verify.py` ## Verification harness
 `src/dq/verify.py` 
 `src/dq/verify.py` Every question kind in `dq.verify.KINDS` has `gold`, `render`, `parse`, and `check`. Any kind takes an optional `moves` prefix, which turns a `*-CURRENT` question into its `*-FUTURE` form.
 `src/dq/verify.py` 
 `src/dq/verify.py` | Stage | Kinds |
 `src/dq/verify.py` |---|---|
 `src/dq/verify.py` | STATIC-CURRENT | `piece_on_square`, `locate_kings`, `count`, `tempo`, `board` |
 `src/dq/verify.py` | DYNAMIC-CURRENT | `legal_moves_for_piece`, `legal_moves`, `capture_sequences` (all vs maximal), `threatened`, `safe_king_moves` |
 `src/dq/verify.py` 
 `src/dq/verify.py` `check_explanation(pos, text)` verifies `BEST_MOVE`, `CRITICAL_LINE`, `PROMISING_MOVES`, `EVALUATION` for legality and well-formedness.
