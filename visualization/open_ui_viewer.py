"""
Builds and opens the HTML UI for a finished run.

    python -m visualization.open_ui_viewer run_1984977510.txt
    python -m visualization.open_ui_viewer 1984977510          # seed alone works too

Reads the transcript txt_viewer.py wrote, reconstructs the event log from it,
writes run_<seed>.json and run_<seed>.html beside it, and opens the HTML.

This is the only entry point that opens the UI viewer. simulation/run.py writes the
transcript and stops; nothing else builds or launches the interface.

Everything is read out of the transcript rather than regenerated from the seed,
so a run recorded under different world_config constants (a different
NUM_VOTERS, say) still renders as it actually happened. The cost is that levels
and attributions come back at the one decimal place the transcript prints, which
is all the interface displays anyway.
"""
import argparse
import ast
import os
import pathlib
import re
import sys
import webbrowser

from simulation.llm import extract_marked_fields
from simulation.candidates import CANDIDATE_MARKERS
from simulation.voters import VOTE_MARKERS, FINAL_VOTE_MARKERS
from simulation.engine import tally_votes
from visualization.ui_viewer import write_visualizer

OUTPUT_DIR = "output"

SEED_RE = re.compile(r"=== SIMULATION RUN \(seed=(\d+)\) ===")
MODEL_RE = re.compile(r"^MODEL: (.+)$", re.M)
GT_HEADER_RE = re.compile(r"^\s*start\s+now\s+change\s+(.+?)\s*$")
ROUND_RE = re.compile(r"^--- ROUND (\d+) ---$")
CANDIDATE_RE = re.compile(r"^\[Candidate: (.+)\] raw response:$")
VOTER_RE = re.compile(r"^\[Voter: (.+)\] raw response:$")
COMPARISON_RULE_RE = re.compile(r"^-{5,}\+-{5,}$")
TALLY_RE = re.compile(r"^Vote tally: (\{.*\})$", re.M)
WINNER_RE = re.compile(r"^Winner: (.+)$", re.M)
DURATION_RE = re.compile(r"^Total time: ([\d.]+)s$", re.M)


class TranscriptError(Exception):
    """The transcript is not in the shape txt_viewer.py writes."""


def _number(token):
    """Transcripts print an absent value as 'unknown' (see prompts._fmt)."""
    try:
        return float(token)
    except ValueError:
        return None


def _parse_ground_truth(lines, seed):
    """
    Reads the === GROUND TRUTH === table, whose header names the candidates in
    column order and whose rows name the metrics:

              start   now   change   Reyes   Calder
    Tolan     67.8  71.9    +4.1   -12.0    16.1
    """
    try:
        start = lines.index("=== GROUND TRUTH ===")
    except ValueError:
        raise TranscriptError(
            "no '=== GROUND TRUTH ===' section — this transcript predates it, and the "
            "world state it ran under cannot be recovered from the seed alone"
        )

    header = GT_HEADER_RE.match(lines[start + 1])
    if not header:
        raise TranscriptError(f"unreadable ground truth header: {lines[start + 1]!r}")

    names = header.group(1).split()
    candidate_keys = [f"candidate_{i}" for i in range(1, len(names) + 1)]
    candidate_names = dict(zip(candidate_keys, names))

    metrics, gt_start, gt_now, gt_change = [], {}, {}, {}
    attribution = {key: {} for key in candidate_keys}

    for line in lines[start + 2:]:
        if not line.strip():
            break
        parts = line.split()
        if len(parts) != 4 + len(candidate_keys):
            raise TranscriptError(f"unreadable ground truth row: {line!r}")
        metric = parts[0].lower()
        metrics.append(metric)
        gt_start[metric] = _number(parts[1])
        gt_now[metric] = _number(parts[2])
        gt_change[metric] = _number(parts[3])
        for key, token in zip(candidate_keys, parts[4:]):
            attribution[key][metric] = _number(token)

    if not metrics:
        raise TranscriptError("ground truth table has no metric rows")

    return {
        "seed": seed,
        "metrics": metrics,
        "candidates": candidate_keys,
        "candidate_names": candidate_names,
        "start": gt_start,
        "now": gt_now,
        "change": gt_change,
        "attribution": attribution,
    }


def _parse_voters(lines, ground_truth):
    """
    Reads the === VOTERS' INITIAL SETUP === block, one voter per line:

    Voter 1: weight_tolan=0.03 weight_vesper=0.97 | level_tolan=77.2 ... |
        Reyes_attr(tolan=-19.3, vesper=8.7) Calder_attr(tolan=11.9, vesper=-10.8)
    """
    header = "=== VOTERS' INITIAL SETUP ==="
    if header not in lines:
        raise TranscriptError(f"no {header!r} section — this transcript predates it")

    name_to_key = {v: k for k, v in ground_truth["candidate_names"].items()}
    metrics = ground_truth["metrics"]

    voters = []
    for line in lines[lines.index(header) + 1:]:
        if not line.strip():
            break
        name, _, body = line.partition(": ")

        weights = {m: _number(v) for m, v in re.findall(r"weight_(\w+)=(\S+)", body)}
        levels = {m: _number(v) for m, v in re.findall(r"level_(\w+)=(\S+)", body)}

        attribution = {key: {} for key in ground_truth["candidates"]}
        for who, inner in re.findall(r"(\S+)_attr\(([^)]*)\)", body):
            key = name_to_key.get(who)
            if key is None:
                raise TranscriptError(f"attribution for unknown candidate {who!r}")
            for metric, value in re.findall(r"(\w+)=([^,\s]+)", inner):
                attribution[key][metric] = _number(value)

        missing = [m for m in metrics if m not in weights or m not in levels]
        if missing:
            raise TranscriptError(f"{name}: no weight/level for {', '.join(missing)}")

        voters.append({
            "voter_id": f"voter_{len(voters) + 1}",
            "name": name.strip(),
            "weights": weights,
            "perceived_level": levels,
            "perceived_attribution": attribution,
        })

    if not voters:
        raise TranscriptError("voter setup block is empty")
    return voters


def _parse_rounds(lines):
    """
    Splits the per-round transcript into raw responses. Each speaker's block runs
    from its '[Candidate: X] raw response:' header to the next header, minus the
    side-by-side comparison table txt_viewer.py prints after the statements.
    """
    rounds = []
    current = None
    speaker = None
    buffer = []

    def flush():
        nonlocal speaker, buffer
        if speaker is None:
            return
        body = buffer
        for i, line in enumerate(body):
            if COMPARISON_RULE_RE.match(line.strip()):
                body = body[:max(0, i - 1)]  # also drop the table's header row
                break
        current[speaker[0]].append((speaker[1], "\n".join(body).strip()))
        speaker, buffer = None, []

    for line in lines:
        stripped = line.strip()

        match = ROUND_RE.match(stripped)
        if match:
            flush()
            current = {"round": int(match.group(1)), "candidates": [], "voters": []}
            rounds.append(current)
            continue

        if stripped == "=== DIAGNOSTICS ===":
            flush()
            break

        if current is None:
            continue

        match = CANDIDATE_RE.match(stripped)
        if match:
            flush()
            speaker = ("candidates", match.group(1))
            continue

        match = VOTER_RE.match(stripped)
        if match:
            flush()
            speaker = ("voters", match.group(1))
            continue

        if speaker:
            buffer.append(line)
    else:
        flush()

    if not rounds:
        raise TranscriptError("no '--- ROUND n ---' sections")
    return rounds


def _parse_result(text, ground_truth, final_polls):
    """
    Prefers the tally the run itself recorded. Falls back to recounting the final
    polls when the transcript predates the diagnostics block.
    """
    tally_match = TALLY_RE.search(text)
    winner_match = WINNER_RE.search(text)

    if tally_match and winner_match:
        return {
            "votes": ast.literal_eval(tally_match.group(1)),
            "winner": winner_match.group(1).strip(),
        }

    return tally_votes(final_polls, ground_truth["candidate_names"])


def parse_transcript(path):
    """Reconstructs an event_log from a run_<seed>.txt written by txt_viewer.py."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()

    seed_match = SEED_RE.search(text)
    if not seed_match:
        raise TranscriptError("no '=== SIMULATION RUN (seed=...) ===' header")
    seed = int(seed_match.group(1))

    model_match = MODEL_RE.search(text)
    ground_truth = _parse_ground_truth(lines, seed)
    voter_dicts = _parse_voters(lines, ground_truth)
    parsed_rounds = _parse_rounds(lines)

    name_to_key = {v: k for k, v in ground_truth["candidate_names"].items()}
    name_to_voter_id = {v["name"]: v["voter_id"] for v in voter_dicts}
    last_round = parsed_rounds[-1]["round"]

    rounds = []
    for entry in parsed_rounds:
        is_final = entry["round"] == last_round

        statements = {}
        for name, raw in entry["candidates"]:
            key = name_to_key.get(name)
            if key is None:
                raise TranscriptError(f"round {entry['round']}: unknown candidate {name!r}")
            result = extract_marked_fields(raw, CANDIDATE_MARKERS)
            result["candidate_key"] = key
            result["name"] = name
            statements[key] = result

        polls = {}
        for name, raw in entry["voters"]:
            voter_id = name_to_voter_id.get(name)
            if voter_id is None:
                raise TranscriptError(f"round {entry['round']}: unknown voter {name!r}")
            markers = FINAL_VOTE_MARKERS if is_final else VOTE_MARKERS
            result = extract_marked_fields(raw, markers)
            result["voter_id"] = voter_id
            result["name"] = name
            polls[voter_id] = result

        rounds.append({
            "round": entry["round"],
            "candidate_statements": statements,
            "polls": polls,
        })

    duration_match = DURATION_RE.search(text)

    return {
        "seed": seed,
        "model": model_match.group(1).strip() if model_match else "unrecorded",
        "ground_truth": ground_truth,
        "voter_dicts": voter_dicts,
        "rounds": rounds,
        "result": _parse_result(text, ground_truth, list(rounds[-1]["polls"].values())),
        "duration_seconds": float(duration_match.group(1)) if duration_match else 0.0,
    }


def available_runs():
    if not os.path.isdir(OUTPUT_DIR):
        return []
    return sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith("run_") and f.endswith(".txt"))


def resolve_transcript(arg):
    """Accepts a path, a bare filename, or just the seed."""
    candidates = [
        arg,
        os.path.join(OUTPUT_DIR, arg),
        os.path.join(OUTPUT_DIR, f"{arg}.txt"),
        os.path.join(OUTPUT_DIR, f"run_{arg}.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build and open the HTML UI for a finished run.",
        epilog="example: python -m visualization.open_ui_viewer run_1984977510.txt",
    )
    parser.add_argument("run", nargs="?", help="transcript filename in output/, or just the seed")
    parser.add_argument("--no-open", action="store_true", help="write the files but do not open a browser")
    args = parser.parse_args()

    runs = available_runs()
    if not args.run:
        print("Which run? Pass a transcript filename, e.g.:\n")
        for name in runs:
            print(f"    python -m visualization.open_ui_viewer {name}")
        if not runs:
            print(f"    (no run_*.txt found in {OUTPUT_DIR}/ — python -m simulation.run writes them)")
        return 1

    path = resolve_transcript(args.run)
    if path is None:
        print(f"No transcript matching {args.run!r}.")
        if runs:
            print("Available:", ", ".join(runs))
        return 1

    try:
        event_log = parse_transcript(path)
    except TranscriptError as e:
        print(f"Could not read {path}: {e}")
        return 1

    html_path = write_visualizer(event_log, output_dir=os.path.dirname(path) or ".")
    print(f"Built: {html_path}")

    if not args.no_open:
        # as_uri() percent-encodes, which a hand-built file:/// URL would not do
        # for the spaces this project's path contains.
        webbrowser.open(pathlib.Path(html_path).resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
