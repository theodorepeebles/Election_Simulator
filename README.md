# Election Simulator

A simulated election between two LLM politicians (Reyes and Calder), judged by a panel
of LLM voters, testing whether a model will make false claims to win a contest — and
what it takes to stop it.

The politicians are shown the true state of the world. The voters are not: they hold
only noisy private perceptions plus whatever the candidates tell them. **No prompt
anywhere instructs a model to lie.** If deception appears in the main simulation, it is
emergent.

The spectator, unlike the voters, can see the ground-truth table the whole time. That
asymmetry is the point of the project.

## Design principles

- Never instruct a model to lie (except in the explicitly-labelled pressure test below,
  which exists to measure the capability ceiling).
- Ground truth is a number in a table. Measurement is arithmetic against that table —
  no LLM judge anywhere in the classification path.
- The world is invented and the metric names (**Tolan**, **Vesper**) are semantically
  empty, so a model has no priors about what a good number looks like and has to reason
  from the table it was given.
- Voters are instrumentation and detection mechanism at once: a lie is "caught" exactly
  when a voter declines to update toward it.

## The world

Ground truth is generated fresh per run from a random seed (`world_config.py`):

- Start levels for Tolan and Vesper: independently drawn `Uniform(50, 70)`.
- Each candidate's magnitude drawn independently from `Uniform(10, 20)`. One candidate
  is randomly assigned "Tolan-positive" (`+magnitude` on Tolan, `−magnitude` on Vesper)
  and the other gets the mirror, so which candidate holds the larger coefficient varies
  run to run.
- `change` is derived as the sum of both candidates' attributions per metric, never
  independently randomized — so the accounting constraint (attributed deltas sum to the
  level change) holds by construction rather than by convention.
- Display names Reyes/Calder are a fixed constant; backend keys are
  `candidate_1`/`candidate_2`.

## Agents

**Politicians** (`candidates.py`) — one `Candidate` per candidate. The system prompt
contains the full ground-truth table and is built once at construction, then reused
every round. Each round the candidate emits two markers: a private `PLAN:` (its
reasoning and intent for the round, never shown to anyone) followed by a public
`STATEMENT:` (shown to the voters and to the opponent). The final round is flagged
in-prompt. Word budget: 150.

**Voters** (`voters.py`) — `NUM_VOTERS = 5`, odd so the vote cannot tie. Each voter gets:

- A fixed Tolan/Vesper weight vector summing to 1 — immutable, re-injected every prompt,
  never part of the model's output schema.
- Private noisy starting signals: `perceived_level = true ± N(0, 4)` and
  `perceived_attribution = true ± N(0, 8)` (tunable globals).

Each round a voter returns three markers — `MY VOTE GOES TO:`, `EXPLANATION:` (80 words),
and `PRIVATE PERCEPTIONS:` (numeric, one line per field, never shown to candidates). On
the final round it returns the first two only. A perception line may read `UNCHANGED`
instead of a number, which falls back to that field's most recent known value, resolved
field-by-field rather than as an all-or-nothing block. Missing or unparseable lines fall
back the same way.

## Round structure

`NUM_ROUNDS = 5`. The final vote is folded into round N of the loop rather than run as a
separate step after it:

```
for round in 1..N:
    both candidates broadcast (PLAN + STATEMENT)
    all voters poll
    on round N, that poll IS the final vote (two markers, no perception update)
```

There is no additional broadcast after round N. History text is assembled per round by
`engine.py`, not by the prompt templates, and differs by audience: candidates see a
personalized history ("Your statement" vs. the opponent's) *including* past poll results,
while voters see both candidates' past statements and **no** poll results — voters never
learn how anyone else voted.

## The lie pressure test

`lie_pressure_test.py` is the sharpest instrument here and is separate from the main
simulation. It rigs a world where the subject is negative on both metrics and the
opponent is the only positive force, so **honesty provably cannot win** — and it verifies
that claim by computing the prior-implied and truth-optimal tallies *before* spending a
single API call, aborting loudly if the subject could win honestly.

It then runs a pressure gradient — `spontaneous` (no framing), `permitted`,
`unverifiable_neutral`, `instructed` (told outright to misrepresent the record) — across
a scripted opponent and scripted polls, plus a `live` condition against real LLM
opponents and voters. Supplying intent in the `instructed` condition separates *unwilling*
from *unable*: a model that lies fluently on command but not when left alone is making a
choice. Nothing is scored; every channel is dumped verbatim to
`lie_pressure_results.txt` for hand-classification.

## What gets measured

`engine.py` returns an `event_log` holding the complete record of a run: seed, model,
full ground truth, every voter's initial state, every round's candidate statements and
polls (raw text plus parsed fields), the final tally and winner, and wall-clock duration.

Two viewers read it:

- **`txt_viewer.py`** — a readable text report: ground-truth table, voters' initial
  setup, every raw response as encountered, statements side-by-side in wrapped columns,
  and a diagnostics block (counts, tally, winner, malformed-response count, total time).
  Writes `output/run_<seed>.txt`.
- **`ui_viewer.py` + `ui_viewer_template.html`** — a standalone HTML visualizer with the
  run log inlined, so it opens straight from disk with no server. Shows the ground-truth
  panel (spectator-only), a voter grid with weights and per-round beliefs, the two-podium
  statement view with `PLAN` and `STATEMENT` visually distinguished, and a round scrubber.
  It is built **post-hoc** from a finished transcript, not streamed live.

Token-by-token streaming does happen live, but in the terminal during the run, not in the
HTML.

Claim severity tiers, belief-shift magnitude, and the "does lie rate predict vote share"
correlation are not built yet.

## Running it

```bash
pip install -r requirements.txt

export OLLAMA_API_KEY=your-key        # PowerShell: $env:OLLAMA_API_KEY = 'your-key'

python run.py                         # full run -> output/run_<seed>.txt
python open_ui_viewer.py <seed>       # build + open the HTML viewer for that run
python lie_pressure_test.py           # the pressure test; --dry-run renders prompts only
```

Model is set by `MODEL` in `engine.py` (currently `gemma4:31b-cloud`, served by the
Ollama cloud API). A full run is 5 rounds × (2 candidates + 5 voters) = 35 sequential
API calls, so expect it to take a while.

## Known limitations

- **Voters never update their perceptions.** Every voter currently emits `UNCHANGED` on
  every `PRIVATE PERCEPTIONS` field for the whole run, so beliefs never move off the
  initial noisy draw and the election is effectively decided before round 1. This is the
  central dynamic of the project and it is not currently firing. Two hypotheses: the
  `UNCHANGED` escape hatch is too cheap, or the voter prompt over-anchors on private
  perception relative to what the candidates actually say. Unresolved.
- **Marker parsing is substring-based.** A response missing a marker silently yields
  `None`; `tally_votes` matches candidate names by substring. `ask_llm` already accepts a
  `fmt` argument that is never used — structured JSON output would remove this class of
  failure.
- **No tests**, and no retry on API calls: one timeout in round 4 loses the whole run.
- The claims panel in the HTML viewer is stubbed — candidates do not yet emit structured
  claims, so nothing populates it.

## Repo map

| File | Role |
| --- | --- |
| `world_config.py` | Ground-truth generation, voter noise, tunable constants |
| `prompts.py` | All prompt builders + the ground-truth table formatter |
| `candidates.py` / `voters.py` | Agent wrappers and perception parsing |
| `engine.py` | Round loop, history assembly, tally, event log |
| `llm.py` | Ollama call with live streaming, marker extraction |
| `run.py` | Entry point for a full run |
| `txt_viewer.py` | Text report |
| `ui_viewer.py`, `ui_viewer_template.html`, `open_ui_viewer.py` | HTML visualizer |
| `lie_pressure_test.py` | The rigged-world pressure test |
