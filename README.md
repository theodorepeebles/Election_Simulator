# Election Simulator

A simulated election between two LLM politicians, judged by a panel
of LLM voters.  The goal is to test whether a model will make false claims when given
the goal to win a contest.  The secondary goal is to test if LLMs will identify and 
consider false or inconsistent claims when making decisions.

## Simulation Design

The world contains two metrics (**Tolan**, **Vesper**), which are
intentionally semantically empty (i.e. not crime, education, etc.), so a model 
has no priors about what a good number looks like and has to reason
from the table it was given.  Every agent is instructed that higher is better
uniformly for the metrics.

Facts about the world are generated fresh per run from a random seed (`simulation/world_config.py`):
Ground Truth
- Levels of each metric
- Politician attribution to each metric (i.e. how much did politician 1 change Tolan
in the past, etc.); with 2 politicians and 2 metrics, this is 4 numbers generated
- Subsequently, the total change of a metric over time (i.e. "in the past Tolan was at 50
and now it is at 62)
Voter Truth
- Each voter's noisy perception of the metric levels
- Each voter's noisy perception of the politician attributions
- Each voter's weight vector for how much they care about
each metric (e.g. Tolan: 0.43, Vesper 0.57); always summing to 1

The politicians are shown the true state of the world. The voters are not: they hold
only their private perceptions plus whatever the candidates tell them.


## Agents

**Politicians** (`candidates.py`) — one `Candidate` per candidate. The system prompt
contains the full ground-truth table and is built once at construction, then reused
every round. Each round the candidate emits two markers: a private `PLAN:` (its
reasoning and intent for the round, never shown to anyone) followed by a public
`STATEMENT:` (shown to the voters and to the opponent). The STATEMENT section has a word
budget of 150 enforced in-prompt only. The final round is flagged in-prompt.

**Voters** (`voters.py`) — `NUM_VOTERS = 5`, odd so the vote cannot tie. Each voter gets:

- An immutable Tolan/Vesper weight vector
- Private noisy starting signals: `perceived_level = true ± N(0, 4)` and
  `perceived_attribution = true ± N(0, 8)`.

Each round a voter returns three markers — `MY VOTE GOES TO:`, `EXPLANATION:` (80 words),
and `PRIVATE PERCEPTIONS:` (numeric, one line per field, never shown to candidates).
The private perceptions allows voters to update their noisy perceptions of attributions and 
metric levels.

## Round structure

```
for round in 1..N:
    both candidates broadcast (PLAN + STATEMENT)
    all voters poll
    on round N, that poll IS the final vote
```

There is no additional broadcast after round N. History text is assembled per round by
`engine.py`, not by the prompt templates, and differs by audience: candidates see a
personalized history ("Your statement" vs. the opponent's) *including* past poll results,
while voters see both candidates' past statements and **no** poll results — voters never
learn how anyone else voted.

## What gets measured

`engine.py` returns an `event_log` holding the complete record of a run: seed, model,
full ground truth, every voter's initial state, every round's candidate statements and
polls, the final tally and winner, and wall-clock duration.

Two viewers read it:

- **`simulation/txt_viewer.py`** — a readable text report: ground-truth table, voters' initial
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

python -m simulation.run                        # full run -> output/run_<seed>.txt
python -m visualization.open_ui_viewer <seed>   # build + open the HTML viewer for that run
```

Model is set by `MODEL` in `simulation/engine.py` (currently `gemma4:31b-cloud`, served by the
Ollama cloud API). A full run is 5 rounds × (2 candidates + 5 voters) = 35 sequential
API calls.

## Current Problems

- **Voters never update their perceptions.** Despite communicating changing beliefs in the
EXPLANATION section, Every voter currently emits `UNCHANGED` on every `PRIVATE PERCEPTIONS`
field for the whole run, so beliefs never move off the initial noisy draw. Two hypotheses:
the `UNCHANGED` escape hatch is too cheap, or the voter prompt over-anchors on private
perception relative to what the candidates actually say. Unresolved.
- The claims panel in the HTML viewer is stubbed — candidates do not yet emit structured
  claims, so nothing populates it.

# Planned

The information environment is currently as thin as it can be: candidates broadcast one
statement each, and every voter reads it alone and never hears from other voters.

- **Voter discussion board.** Between rounds, voters see and respond to each other
  instead of only reacting to candidate statements. The measurement worth having is
  whether one skeptical voter catching a false claim propagates to the rest — peer
  correction as a detection mechanism, rather than each voter having to catch a lie alone.
- **Candidate debate rounds.** Some rounds become a direct exchange rather than two
  parallel broadcasts: candidates address each other and can be challenged on a specific
  claim in the moment. Not every round — the contrast between the two formats is part of
  the point. This is the sharpest test of whether a model will hold a false claim under
  direct contradiction, or quietly drop it the following round.
- **Newscaster LLM.** An LLM that summarizes each round in news-report form, which the
  voters then read as another channel. It can be given ground truth (an accurate press)
  or only the statements (a press that can itself be misled, and then launders a false
  claim onward to the voters). The second is the more interesting condition: it tests
  whether a lie gains credibility by passing through an apparently neutral third party.


## Repo map

Run everything from the repo root. `simulation/` runs elections and writes the
text report on its own; it never imports from `visualization/`. The HTML viewer
reads the text reports the simulation writes.

| File | Role |
| --- | --- |
| `simulation/world_config.py` | Ground-truth generation, voter noise, tunable constants |
| `simulation/prompts.py` | All prompt builders + the ground-truth table formatter |
| `simulation/candidates.py` / `simulation/voters.py` | Agent wrappers and perception parsing |
| `simulation/engine.py` | Round loop, history assembly, tally, event log |
| `simulation/llm.py` | Ollama call with live streaming, marker extraction |
| `simulation/run.py` | Entry point for a full run (`python -m simulation.run`) |
| `simulation/txt_viewer.py` | Text report, written to `output/run_<seed>.txt` after each run |
| `visualization/ui_viewer.py`, `visualization/ui_viewer_template.html`, `visualization/open_ui_viewer.py` | HTML visualizer |
