import os
import textwrap
from itertools import zip_longest

META_KEYS = {"raw", "voter_id", "name", "candidate_key"}
COLUMN_WIDTH = 45


def _is_malformed(result):
    return any(v is None for k, v in result.items() if k not in META_KEYS)


def _side_by_side(label_a, text_a, label_b, text_b, width=COLUMN_WIDTH):
    lines_a = textwrap.wrap(text_a or "", width) or [""]
    lines_b = textwrap.wrap(text_b or "", width) or [""]
    rows = [f"{label_a:<{width}} | {label_b}"]
    rows.append(f"{'-' * width}-+-{'-' * width}")
    for a, b in zip_longest(lines_a, lines_b, fillvalue=""):
        rows.append(f"{a:<{width}} | {b}")
    return "\n".join(rows)


def render_event_log(event_log):
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    seed = event_log["seed"]
    rounds = event_log["rounds"]
    num_candidates = len(rounds[0]["candidate_statements"]) if rounds else 0
    num_voters = len(rounds[0]["polls"]) if rounds else 0
    malformed = 0

    emit(f"=== SIMULATION RUN (seed={seed}) ===")

    for entry in rounds:
        emit()
        emit(f"--- ROUND {entry['round']} ---")

        statements = entry["candidate_statements"]
        for result in statements.values():
            if _is_malformed(result):
                malformed += 1
            emit()
            emit(f"[Candidate: {result['name']}] raw response:")
            emit(result["raw"])

        a, b = statements.values()
        emit()
        emit(_side_by_side(a["name"], a["statement"], b["name"], b["statement"]))

        for result in entry["polls"].values():
            if _is_malformed(result):
                malformed += 1
            emit()
            emit(f"[Voter: {result['name']}] raw response:")
            emit(result["raw"])

    result = event_log["result"]
    emit()
    emit("=== DIAGNOSTICS ===")
    emit(f"Candidates: {num_candidates}")
    emit(f"Voters: {num_voters}")
    emit(f"Rounds: {len(rounds)}")
    emit(f"Vote tally: {result['votes']}")
    emit(f"Winner: {result['winner']}")
    emit(f"Malformed responses: {malformed}")
    emit(f"Total time: {event_log.get('duration_seconds', 0):.1f}s")

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"run_{seed}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved to: {output_path}")

    return output_path
