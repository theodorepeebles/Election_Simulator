import json
import os

from simulation.voters import parse_private_perceptions

TEMPLATE_NAME = "ui_viewer_template.html"
LOG_PLACEHOLDER = "/*__RUN_LOG__*/null"


def _words(text):
    return len((text or "").split())


def _flatten_beliefs(beliefs, metrics, candidate_keys):
    """
    Turns the nested {perceived_level, perceived_attribution} shape used by the
    simulation into the flat "<metric>_level" / "<metric>_attr_<candidate_key>"
    keys the visualizer indexes by. Fields the voter never gave a number for
    come through as None and render as an em dash.
    """
    if not beliefs:
        return None

    level = beliefs.get("perceived_level") or {}
    attribution = beliefs.get("perceived_attribution") or {}

    flat = {}
    for metric in metrics:
        flat[f"{metric}_level"] = level.get(metric)
        for key in candidate_keys:
            flat[f"{metric}_attr_{key}"] = (attribution.get(key) or {}).get(metric)
    return flat


def _vote_key(vote_text, candidate_names):
    """
    Maps a voter's free-text MY VOTE GOES TO line onto a candidate key, using
    the same substring match engine.tally_votes counts with, so the visualizer's
    running tally never disagrees with the recorded result.
    """
    text = (vote_text or "").strip().lower()
    for key, name in candidate_names.items():
        if name.lower() in text:
            return key
    return None


def _statement_entry(key, result):
    plan = result.get("plan") or ""
    statement = result.get("statement") or ""
    malformed = not statement.strip()
    if malformed:
        # Nothing parsed out of the response; show the raw text rather than a
        # blank card, so a bad generation is still inspectable.
        plan, statement = "", (result.get("raw") or "").strip()

    return {
        "candidate": key,
        "name": result.get("name"),
        "plan": plan,
        "statement": statement,
        # TODO: candidates do not declare structured claims yet. Populate with
        # {"cue": <substring of statement>, "metric": ..., "value": ...} and the
        # CLAIMS ON THE RECORD panel fills itself in.
        "claims": [],
        "words": _words(plan) + _words(statement),
        "malformed": malformed,
    }


def _poll_entry(voter_id, poll, candidate_names, perceptions):
    explanation = poll.get("explanation") or ""
    vote = _vote_key(poll.get("my_vote_goes_to"), candidate_names)
    malformed = not explanation.strip() or vote is None
    if not explanation.strip():
        explanation = (poll.get("raw") or "").strip()

    return {
        "voter": voter_id,
        "name": poll.get("name"),
        "vote": vote,
        "explanation": explanation,
        "perceptions": perceptions,
        "words": _words(explanation),
        "malformed": malformed,
    }


def build_run_log(event_log):
    """
    Reshapes the simulation's event_log into the flat, JSON-serialisable log the
    HTML visualizer reads. Nothing here mutates event_log.
    """
    ground_truth = event_log["ground_truth"]
    voter_dicts = event_log.get("voter_dicts") or []
    metrics = ground_truth["metrics"]
    candidate_keys = ground_truth["candidates"]
    candidate_names = ground_truth["candidate_names"]

    truth = {
        metric: {
            "start": ground_truth["start"][metric],
            "now": ground_truth["now"][metric],
            "change": ground_truth["change"][metric],
            "attribution": {key: ground_truth["attribution"][key][metric] for key in candidate_keys},
        }
        for metric in metrics
    }

    voters = []
    beliefs = {}
    for voter in voter_dicts:
        beliefs[voter["voter_id"]] = {
            "perceived_level": voter["perceived_level"],
            "perceived_attribution": voter["perceived_attribution"],
        }
        voters.append({
            "id": voter["name"],
            "voter_id": voter["voter_id"],
            "weights": {metric: voter["weights"][metric] for metric in metrics},
            "initial": _flatten_beliefs(beliefs[voter["voter_id"]], metrics, candidate_keys),
        })

    rounds = []
    malformed_total = 0
    round_entries = event_log["rounds"]
    for entry in round_entries:
        statements = [_statement_entry(key, result) for key, result in entry["candidate_statements"].items()]

        polls = []
        for voter_id, poll in entry["polls"].items():
            # The final round asks for a vote only, so there is no perception
            # update to fold in; the visualizer carries the last one forward.
            text = poll.get("private_perceptions")
            if text:
                beliefs[voter_id] = parse_private_perceptions(text, beliefs.get(voter_id))
                perceptions = _flatten_beliefs(beliefs.get(voter_id), metrics, candidate_keys)
            else:
                perceptions = None
            polls.append(_poll_entry(voter_id, poll, candidate_names, perceptions))

        malformed_total += sum(1 for s in statements if s["malformed"])
        malformed_total += sum(1 for p in polls if p["malformed"])

        rounds.append({
            "round": entry["round"],
            "is_final": entry["round"] == round_entries[-1]["round"],
            "statements": statements,
            "polls": polls,
        })

    result = event_log["result"]
    winner_key = next(
        (key for key, name in candidate_names.items() if name == result["winner"]),
        candidate_keys[0],
    )

    return {
        "seed": event_log["seed"],
        "config": {"model": event_log.get("model") or "—"},
        "metrics": metrics,
        "candidates": candidate_keys,
        "names": candidate_names,
        "ground_truth": truth,
        "voters": voters,
        "rounds": rounds,
        "result": {
            "winner": winner_key,
            "tally": result["votes"],
            "duration_s": round(event_log.get("duration_seconds", 0), 1),
        },
        "diagnostics": {"malformed": malformed_total},
    }


def write_visualizer(event_log, output_dir="output"):
    """
    Writes two files for the run: a run_<seed>.json log, and a standalone
    run_<seed>.html with that log inlined so it opens straight from disk with no
    server. Returns the path to the HTML.
    """
    run_log = build_run_log(event_log)
    seed = event_log["seed"]

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"run_{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEMPLATE_NAME)
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Escaping "<" keeps a "</script>" inside any statement from closing the
    # inlined block early; it stays a valid escape inside a JSON string.
    payload = json.dumps(run_log, ensure_ascii=False).replace("<", "\\u003c")
    html_path = os.path.join(output_dir, f"run_{seed}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(template.replace(LOG_PLACEHOLDER, payload))

    return html_path
