# Draughts-QUEEN: An LM That Plays Brazilian Draughts and Explains Its Moves

**Status:** Phase 0 in progress · **Owner:** Alan · **Started:** Oct 2026

Adapt the QUEEN recipe (Bhaskar, Cheng, Chen — Princeton, arXiv 2610.03695) from chess to Brazilian draughts: a frozen expert encoder, a frozen instruction-tuned LM, a trained cross-attention bridge, a verifiable QA curriculum, and iterative search distillation. Target: a ~4B model that plays at a credible lidraughts rating and produces explanations whose moves, lines, and concepts can be checked.

Legend used throughout:
- **[lit]** — established in published work (mostly QUEEN)
- **[ext]** — reasonable extension of published work, not yet demonstrated
- **[spec]** — speculative, no published analog

---

## 1. Why this project

- QUEEN showed an LM cross-attending into a silent expert's latents can reach ~2700 Lichess Elo and explain moves, but its weakest result was *conceptual coherence* (2.76/5): hallucinated motifs propagate through the distillation loop because nothing checks them. [lit]
- Draughts has a small, programmatically definable concept vocabulary (shot, exchange, trapped king, tempo, bridge). That makes a **verifiable concept filter** in the distillation loop possible — the thing QUEEN couldn't do. [ext]
- 32 playable squares and 4 piece types make the whole stack cheap enough for one RTX 5090.

## 2. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Variant | **Brazilian draughts** (8×8, international rules) | Only 8×8 variant with a rated online pool + bot API (lidraughts.org). Rules are a strict subset of 10×10 international, so a later scale-up is mostly a board-size change. |
| Oracle | **Self-oracle**: own encoder + deep MCTS, plus self-generated endgame tablebases (≤6–7 pieces) | No strong public Brazilian engine confirmed. AlphaZero-style self-oracle is defensible; tablebases make the endgame exact. |
| Move generation | Native (`dq/rules.py`), pydraughts as oracle | pydraughts is ~130× slower (object-based); native generator is cross-checked against it. |
| Expert encoder | Train from scratch, AlphaZero-style | No Lc0 equivalent exists for draughts. [ext] |
| Decoder | SmolLM3-3B (fallback Qwen3-4B) | QUEEN's best backbone in ablation; general-purpose, no draughts training. |
| Bridge | Flamingo-style gated cross-attention, encoder layer *i* → decoder layer *2i* | QUEEN's winning architecture; beat LLaVA-prefix and KV-projection. [lit] |
| Seed explanations | Templated hand-crafted-eval (HCE) route, no frontier model | QUEEN showed HCE seeding matches frontier seeding on Elo trajectory. [lit] Saves cost and avoids licensing issues. |
| External metric | lidraughts BOT rating (Brazilian) | Real human-pool rating. |

**Trade-off accepted:** English checkers has a near-perfect oracle (KingsRow/Cake + endgame DBs) but no rated pool; Brazilian has the rated pool but a weaker grader. We chose the public scoreboard.

## 3. Rules that matter for the implementation

Brazilian draughts = international rules on 8×8:
- Men move forward diagonally, **capture forward and backward**.
- **Flying kings**: slide any distance, land on any empty square beyond the captured piece.
- **Mandatory capture, maximum-capture rule**: must take the sequence capturing the most pieces.
- No promotion mid-jump (a man passing through the back rank during a capture does not become a king until the sequence ends there).
- Captured pieces are removed after the full sequence (can't jump the same piece twice).
- White moves first.

The max-capture rule is the hardest thing for both the move generator and the LM to read out of latents. See risks.

## 4. Architecture

```
FEN-like position string
        │
        ▼
┌──────────────────────┐      gated cross-attn        ┌──────────────────────────┐
│  Expert encoder      │  e_i ──────────────────────▶ │  Decoder LM (frozen)     │
│  (frozen, ~25M)      │  (layer i → decoder 2i)      │  SmolLM3-3B              │
│  32 tokens = squares │                              │  + 32 square tokens      │
│  8 layers × 512      │                              │  + 4 POV piece tokens    │
│  heads: policy, WDL  │                              └──────────────────────────┘
└──────────────────────┘                                          │
                                                                  ▼
                                   prose → BEST_MOVE → CRITICAL_LINE → PROMISING_MOVES → EVALUATION (WDL %)
```

- **Tokens:** 32 square tokens (playable squares only) + POV-relative piece tokens `<PIECE_MM>`, `<PIECE_MK>`, `<PIECE_OM>`, `<PIECE_OK>` (mine/opponent × man/king). Board is reflected so the side to move is always "me". [lit]
- **Bridge:** gated cross-attention blocks, `tanh(α)` gate with α=0 at init so cross-attn opens gradually. ~16 blocks. [lit]
- **Encoder:** transformer over 32 square tokens; policy head over (from, to) pairs plus a capture-sequence head; WDL value head. [ext]
- **Evaluation unit:** win/draw/loss probability, not centipawns — draughts engines report WDL and the game is drawish.

## 5. Phases

### Phase 0 — Infrastructure (week 1)
- [x] `pydraughts` Brazilian move gen wrapped with: legal moves, all capture sequences, max-capture filter, apply-move, position hashing. (Native generator in `dq/rules.py`, ~130× faster than pydraughts; pydraughts kept as reference oracle, cross-checked on 20K+ positions.)
- [x] Position encoding ↔ tokens, with POV reflection (`dq/tokens.py`).
- [~] Retrograde tablebase generator (`dq/tablebase.py`): exact WDL + DTM, verified vs negamax on ≤3 pieces; 4 pieces (6.2M positions, 11 MB on disk, max DTM 86) builds in ~3 min on the Mac; 13k random 4-piece positions pass the one-step consistency check. 5 pieces is feasible in pure Python (~1 h est.); 6 needs the inner loops compiled (numba/C) and ~4 GB RAM — a Phase 1 prerequisite.
- [x] Verification harness: every answer in the QA curriculum must be checkable programmatically. (`dq/verify.py`)
- [x] 1,030 rated Brazilian games from top lidraughts players (`data/lidraughts_brazilian.pdn`); all replay through `rules.py`, recorded losers match no-move finals. Numbering/orientation validated on real data.

### Phase 1 — Expert encoder (weeks 1–3) [ext]
- [~] AlphaZero-style self-play loop: `dq/game.py` (lidraughts draw rules 7.2.3–7.2.8), `dq/mcts.py` (PUCT, tablebase leaves), `dq/model.py` (32-token transformer, policy/WDL/ncap heads), `dq/selfplay.py` + `scripts/selfplay.py` (generate → train → gate → promote). Smoke-tested end to end on the Mac. Batched: each worker runs G concurrent games and evaluates leaves across them in one forward pass (virtual loss); W workers share the GPU; trainer gates + publishes weights by file. ~3.3k sims/s/worker on the Mac (MPS-bound); est. ~10k/worker on the 5090 (Python tree code-bound).
- [x] **Auxiliary head: max-capture length** (`ncap`, weight 0.25) in the model and loss.
- [ ] Gates:
  - beats own MCTS at 1 node by >80% with search off
  - zero WDL errors on tablebase positions
  - lidraughts bot rating for the raw engine (baseline number, also validates the bot plumbing early) — engine wrapper done (`bot/dq_engine.py`), needs a BOT account token to go live
- [ ] Freeze. Record which layers are most probe-able (linear probes for legal-move sets, material, king count) — informs which encoder layers to bridge.

### Phase 2 — Domain adaptation / QA curriculum (weeks 3–5) [lit]
Four stages, sequential, earlier stages replayed at 6–10%. Encoder + decoder frozen; train bridge + new embeddings only. AdamW, cosine, early stopping on held-out accuracy.

| Stage | Question types | Target acc |
|---|---|---|
| STATIC-CURRENT | piece on square; locate kings; material / king counts; tempo count | >99.5% |
| DYNAMIC-CURRENT | legal moves for a piece; **all capture sequences and which are maximal**; threatened pieces; safe king moves | >98% |
| STATIC-FUTURE | board after a 1–6 move sequence | >99.5% |
| DYNAMIC-FUTURE | legal moves / captures / threats after a sequence | >95% |

Checkpoint name: **PAWN-equivalent** (call it `MAN-0`).

Ablations to run once (cheap, at this stage): no encoder (dictionary board state as text); no curriculum. QUEEN saw both collapse Elo to ~500. [lit]

### Phase 3 — Seed explanations (week 5) [lit, HCE variant]
No frontier-model seeding.
- Alpha-beta to 3 ply with own encoder as evaluator; extend captures; cap tree ~40 nodes.
- "Plausible mistakes" from a low-node version of own net (replaces Maia).
- Judge retained lines with deeper MCTS (replaces Stockfish-1000); mark refuted / superseded lines.
- Verbalize up to 3 salient feature deltas per move via templates. Feature set (all computable):
  - material, king count, tempo (sum of advancement), back-rank integrity, center control, mobility, trapped king, hanging piece, **shot** (forced capture sequence netting material), exchange.
- ~15K positions → SFT → `MAN-1`.

### Phase 4 — Iterative search distillation (weeks 6–10) [lit]
Per iteration *k* → *k+1*:
1. **Sample** ~100–250K roots: self-play vs own net at log-random node counts (downsample lopsided WDL), lidraughts human games, generated endgames within tablebase range.
2. **Generate** 3 promising moves from `MAN-k`; if all are mistakes, swap worst for oracle move.
3. **Recurse** into any child whose top move is a mistake (≥10% WDL drop, or any tablebase error). Max 5 descents.
4. **Consolidate** 3 child explanations with a mid-size instruction model (Qwen3-27B class), instructed to add nothing.
5. **Filter** by first-divergence PV comparison; **[ext] also filter on verified concepts** — drop or flag explanations whose named motifs fail the programmatic check.
6. **SFT** → `MAN-(k+1)`.

Run 5–7 iterations. Track internal-ladder rating per iteration; expect a QUEEN-shaped curve.

Run the concept-filter as an **A/B** (with vs without) from iteration 2 onward — this is the main scientific claim.

### Phase 5 — Evaluation (ongoing from Phase 3)

| Axis | Metric | Tool |
|---|---|---|
| Accuracy | lidraughts BOT rating (Brazilian); internal ladder vs own net at fixed nodes | lidraughts-bot |
| Substantiation | no-mistake rate (NMR) and first-move NMR on 1K tactical + 1K general positions; **exact** on tablebase positions | oracle + tablebases |
| Coherence — structural | LM judge, 1–5 | as QUEEN |
| Coherence — conceptual | **programmatic concept check** (precision of named motifs) + LM judge | [ext] |
| Fluency | LM judge | as QUEEN |
| Not-a-policy-readout | disagreement with encoder's policy on positions with ≥4 near-equal moves | as QUEEN |

### Phase 6 — Write-up
- Headline: does a verifiable concept filter fix conceptual-coherence collapse in search distillation?
- Secondary: QUEEN recipe transfers to a domain with no pre-existing expert encoder.

## 5b. Self-play architecture and 5090 runbook

Design (see `dq/selfplay.py` docstring): W worker processes each run G concurrent games and
batch MCTS leaves across them with virtual loss, calling the network **in-process** — no
per-leaf IPC, batch size = G × leaves_per_tree instead of "number of workers" (the layout
in alanyangrice/checkers-rl). Trainer drains finished games, trains from a FIFO buffer,
gates against the last promoted checkpoint, and publishes weights by file.

Measured on the Mac (M5, MPS): ~3.3k sims/s/worker, GPU-bound. On the 5090 the forward pass
is ~1 ms so the Python tree code (~100 µs/sim) is the limit → est. ~10k sims/s/worker;
8 workers ≈ 80k sims/s ≈ 400 moves/s at 200 sims ≈ 5 games/s. If more is needed, the next
lever is a compiled move generator + tree (numba/C), not more GPU.

Runbook on the 5090 box:
```
git clone <repo> && cd checkers-lm
uv venv --python 3.12 && uv pip install -e ".[dev]"      # torch CUDA wheel per pytorch.org
.venv/bin/python -m pytest -q                              # ~2 min
# tablebases: copy data/tb from the Mac (11 MB) or rebuild: Tablebase().build(4) ~3 min
nohup .venv/bin/python scripts/selfplay.py runs/gen1 --rounds 2000 --workers 8 \
    --games-per-worker 32 --sims 200 --round-seconds 120 --train-steps 500 \
    --gate-every 5 --eval-games 40 --eval-sims 100 --tb data/tb > runs/gen1.log 2>&1 &
tail -f runs/gen1/log.jsonl          # loss, games/round, gate results, promotions
```
Windows (PowerShell) equivalents, as run on the 5090 box 2026-10-20 (torch 2.11+cu128, Python 3.12):
```
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
powershell -File scripts\launch_run.ps1 -Name gen2 -Args "--rounds 2000 --workers 8 ... --tb data/tb"
Get-Content runs\gen2\log.jsonl -Wait -Tail 3        # trainer PID in runs\gen2.pid
powershell -File scripts\stop_run.ps1 -Name gen2
```
`launch_run.ps1` runs the trainer as a hidden Windows scheduled task, so it is not a child of the
shell that started it. Lesson from gen1: a `Start-Process` child of an agent/editor shell was killed
silently (no traceback, log just stops) when that shell's process tree restarted; and a *visible*
console window, if closed, aborts the run via the MKL/Fortran close handler (exit 0xC000013A).

Measured round 0 on the 5090: 517 games / 120 s with 8 workers × 32 games × 200 sims
(~4.3 games/s, ~45k positions/s), GPU ~65 %, 9.8 GB — in line with the estimate above.

gen1 (64 rounds, ~2 h, start position only, no rescoring): policy 4.12→1.16 (plateau from ~round 45),
WDL 0.74→0.47 (still falling), ncap →0.005, draw rate 8→42 %, tb_wdl_err 0.23→~0.10 (plateau from
round 34), gates vs_prev 0.43–0.59 (noise at 40 games), raw_vs_prev 0.09–0.30 (freeze criterion
far off). Weights: runs/gen1_final.pt.

**Gate bug found 2026-10-21 (fixed in ae8ef46):** `match()` shared one search tree per game between the
two players, blending both networks' priors and values — every head-to-head score was compressed
toward 0.5. The trained net scored 0.575 vs a *random-init* net under the bug; 100–0 after the fix.
All `vs_prev` / `raw_vs_prev` numbers logged before the fix are meaningless (training itself was
unaffected: gates don't revert). `scripts/eval_ckpts.py` now gives honest checkpoint comparisons.

Honest strength trajectory (fixed eval, 100 games, 100 sims, vs gen1 round 63): round 4 → 0.87,
round 14 → 0.66, round 29 → 0.50, gen2 round 63 → 0.49. **Strength rose for ~30 rounds (1 h),
then plateaued for ~4 h** while policy loss sat at 1.16 and WDL/tb_err kept improving. Rescoring
and endgame starts (gen2) did not move strength. gen3 cuts LR 1e-3 → 2e-4 (standard AlphaZero /
Lc0 / KataGo step-down after the initial phase), everything else fixed, with `--anchor
runs/gen1_final.pt` so each gate logs `vs_anchor` as a trend line. If gen3 is also flat, next
levers in order: more sims per move (better targets), larger replay ratio check, bigger model.

gen2 resumed from gen1_final.pt with `--tb-rescore` (exact tablebase WDL as the value target for
covered positions, Lc0-style) and `--endgame-start-frac 0.2` (20 % of games start from a random
5–8-piece position, KataGo-style mixed starts) — aimed at the tb_wdl_err plateau. Hypothesis to
check: tb_wdl_err drops below 0.10 and WDL keeps falling; gate win rate is not expected to move.
Watch: policy loss falling, draw rate rising from ~0, `tb_wdl_err` falling toward 0,
`vs_prev` ≥ 0.55 on most gates, `raw_vs_prev` climbing (the freeze criterion).

Evidence the loop learns (Mac, CPU, 87k-param `--small` model, 2 workers, 64 sims, 30 s rounds):
round 0→9 policy loss 4.19→2.24, ncap loss 0.85→0.55, draw rate 0→18%, gates vs previous
checkpoint 0.59 / 0.55, raw-policy-vs-previous-search 0.10→0.23, tablebase WDL error
0.37→0.27. WDL loss flat at ~0.79 over this window (outcome noise early on) — the metric to
watch first on the real run.

Gating: by default the gate is a *metric* (AlphaZero-style continuous updates); `--gate-revert`
restores AlphaGo-Zero-style reject-and-revert. Gate matches use random paired openings because
search at T=0 is deterministic.

Tablebase note: the pure-Python builder handles 4 pieces (3 min, 11 MB) but a 5-piece
build swapped a 24 GB Mac; it needs either a leaner (numpy-only) bookkeeping pass or a
compiled builder before 5–6 pieces. Not a blocker for training.

## 6. Compute budget (RTX 5090, 32 GB)

| Item | Estimate |
|---|---|
| Encoder self-play | 3–7 days wall-clock |
| Tablebases ≤6 pieces | hours; disk ~tens of GB |
| QA curriculum (bridge ~470M trainable, bf16, grad ckpt, seq 1–2K) | 2–4 days total across 4 stages |
| Seed SFT | hours |
| Each distillation iteration | data gen 1–2 days (parallel inference), SFT ~1 day |
| Consolidation model (27B) | run quantized, or rent a cloud GPU for the gen step only |

If the full bridge doesn't fit: bridge every 4th decoder layer, or LoRA the decoder with a lighter prefix bridge (QUEEN's LLaVA variant worked but trailed).

## 7. Risks

1. **Max-capture structure isn't readable from latents.** The encoder may play well without representing every capture sequence. Mitigation: auxiliary capture head in Phase 1; probe before bridging; if DYNAMIC-CURRENT stalls <95%, add the head and retrain.
2. **Self-oracle is wrong in the opening/midgame.** Mistake filter is only as good as deep MCTS. Mitigation: use tablebases wherever possible; sanity-check oracle against lidraughts top-player games; keep the 10% WDL threshold loose.
3. **Drawishness.** Many positions have several equally-good moves; "mistake" detection has low signal. Mitigation: oversample tactical/shot positions and endgames in the root pool.
4. **lidraughts bot pool is thin.** Rating may be noisy / few opponents. Mitigation: also report the internal ladder; play many games over weeks.
5. **Hallucinated concepts survive consolidation** (QUEEN's failure). This is the thing we're testing; the concept filter is the mitigation and the experiment.
6. **Thin human commentary corpus.** No Lichess-scale annotated games. HCE templating sidesteps this, but prose will sound templated early; distillation iterations are what diversify it.

## 8. Speculative directions (not in scope) [spec]
- Replace SFT consolidation with RL against the oracle directly.
- Scale to 10×10 international (FMJD ratings, Scan as a real oracle).
- Use the same bridge recipe on a JEPA latent world model instead of a game encoder (QA curriculum over inventory / held item / block-in-front).

## 9. References
- QUEEN — Language Models that Play Chess and Explain Their Moves: https://arxiv.org/abs/2610.03695
- Flamingo (Alayrac et al., 2022): https://arxiv.org/abs/2204.14198
- AlphaZero (Silver et al., 2017): https://arxiv.org/abs/1712.01815
- Evidence of learned look-ahead in a chess-playing network (Jenner et al., 2024): https://arxiv.org/abs/2406.00877
- Acquisition of chess knowledge in AlphaZero (McGrath et al., 2022): https://doi.org/10.1073/pnas.2206625119
- lidraughts: https://github.com/roepstoep/lidraughts · lidraughts-bot: https://github.com/AttackingOrDefending/lidraughts-bot
- pydraughts: https://github.com/AttackingOrDefending/pydraughts