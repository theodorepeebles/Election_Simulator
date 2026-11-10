import json
import os
import random
import time

from world_config import generate_world_state, CONDITIONS, NUM_VOTERS, NUM_CANDIDATES
from candidates import Candidate
from voters import Voter, parse_private_perceptions
from llm import set_quiet
from txt_viewer import run_path

MODEL = "deepseek-v4.1-flash"
# A thinking level: "low", "medium" or "high" (or True/False)
THINK = "low"
NUM_ROUNDS = 5
CANDIDATE_WORD_BUDGET = 150
VOTER_EXPLANATION_WORD_BUDGET = 80


def build_candidate_history_text(candidate_key, ground_truth, rounds_so_far):
    if not rounds_so_far:
        return ""

    opponent_key = "candidate_2" if candidate_key == "candidate_1" else "candidate_1"
    opponent_name = ground_truth["candidate_names"][opponent_key]

    blocks = []
    for entry in rounds_so_far:
        r = entry["round"]
        own_statement = entry["candidate_statements"][candidate_key]["statement"]
        opp_statement = entry["candidate_statements"][opponent_key]["statement"]
        poll_lines = "\n".join(
            f"{poll['name']} -> {poll['my_vote_goes_to']}: {poll['explanation']}"
            for poll in entry["polls"].values()
        )
        blocks.append(
            f"--- ROUND {r} ---\n"
            f"Your statement:\n{own_statement}\n\n"
            f"{opponent_name}'s statement:\n{opp_statement}\n\n"
            f"Poll results after round {r}:\n{poll_lines}"
        )
    return "\n\n".join(blocks)


def build_voter_history_text(ground_truth, rounds_so_far):
    if not rounds_so_far:
        return ""

    c1 = ground_truth["candidate_names"]["candidate_1"]
    c2 = ground_truth["candidate_names"]["candidate_2"]

    blocks = []
    for entry in rounds_so_far:
        r = entry["round"]
        stmt_1 = entry["candidate_statements"]["candidate_1"]["statement"]
        stmt_2 = entry["candidate_statements"]["candidate_2"]["statement"]
        blocks.append(
            f"--- ROUND {r} ---\n"
            f"{c1}'s statement:\n{stmt_1}\n\n"
            f"{c2}'s statement:\n{stmt_2}"
        )
    return "\n\n".join(blocks)


def print_progress(progress, who, start_time):
    """Quiet mode's one line per LLM call. progress is None when not quiet."""
    if progress:
        print(f"{progress}  {who}  {time.time() - start_time:.1f}s", flush=True)


def get_candidate_statements(candidates, ground_truth, round_number, rounds_so_far, statements, save,
                             progress=None):
    """Fills `statements` with each candidate's response, skipping any already in it
    (from a resumed run), and calls save() after each new one."""
    for key, candidate in candidates.items():
        if key in statements:
            continue
        start_time = time.time()
        history_text = build_candidate_history_text(key, ground_truth, rounds_so_far)
        result = candidate.give_statement(round_number, history_text)
        result["candidate_key"] = key
        result["name"] = candidate.name
        statements[key] = result
        save()
        print_progress(progress, candidate.name, start_time)
    return statements


def get_polls(voters, ground_truth, round_number, rounds_so_far, candidate_a_statement, candidate_b_statement,
              last_beliefs_by_voter, is_final_round, polls, save, progress=None):
    """Fills `polls` with each voter's response, skipping any already in it (from a
    resumed run), and calls save() after each new one."""
    history_text = build_voter_history_text(ground_truth, rounds_so_far)
    for voter in voters:
        if voter.voter_id in polls:
            continue
        start_time = time.time()
        result = voter.run_poll(
            round_number=round_number,
            candidate_a_statement=candidate_a_statement,
            candidate_b_statement=candidate_b_statement,
            history_text=history_text,
            last_beliefs=last_beliefs_by_voter[voter.voter_id],
            is_final_round=is_final_round,
        )
        result["voter_id"] = voter.voter_id
        result["name"] = voter.name
        polls[voter.voter_id] = result
        save()
        print_progress(progress, voter.name, start_time)
    return polls


def tally_votes(final_polls, candidate_names):
    votes = {key: 0 for key in candidate_names}
    for poll in final_polls:
        vote_text = (poll["my_vote_goes_to"] or "").strip().lower()
        for key, name in candidate_names.items():
            if name.lower() in vote_text:
                votes[key] += 1
                break

    winner_key = max(votes, key=votes.get)
    return {"votes": votes, "winner": candidate_names[winner_key]}


def save_event_log(event_log, path):
    """
    Writes the full event log to path. Writes a .tmp file first and then swaps
    it in, so a run killed mid-write never leaves a half-written JSON behind.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(event_log, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def build_agents(event_log):
    """Candidates and voters for a run, using the settings saved in its event log."""
    ground_truth = event_log["ground_truth"]
    num_rounds = event_log["num_rounds"]
    model = event_log["model"]
    think = event_log["think"]

    candidate_keys = [f"candidate_{i}" for i in range(1, NUM_CANDIDATES + 1)]
    candidates = {
        key: Candidate(key, ground_truth, num_rounds, event_log["candidate_word_budget"], model, think)
        for key in candidate_keys
    }
    voters = [
        Voter(v, num_rounds, event_log["voter_explanation_word_budget"], model, think)
        for v in event_log["voter_dicts"]
    ]
    return candidates, voters


def run_rounds(event_log, candidates, voters, last_beliefs_by_voter, quiet, json_path):
    """
    Plays every round not yet in event_log["rounds"], saving the JSON to
    json_path after every LLM response. Responses for the round in progress are
    kept in event_log["partial_round"]; on resume, those are reused and only the
    missing ones are asked. A round moves into event_log["rounds"] once it's complete.
    """
    set_quiet(quiet)
    ground_truth = event_log["ground_truth"]
    num_rounds = event_log["num_rounds"]
    voters_by_id = {v.voter_id: v for v in voters}

    # Counting from earlier sessions too, if this run was resumed.
    start_time = time.time() - event_log["duration_seconds"]

    def save():
        event_log["duration_seconds"] = time.time() - start_time
        save_event_log(event_log, json_path)

    save()

    for round_number in range(len(event_log["rounds"]) + 1, num_rounds + 1):
        is_final_round = round_number == num_rounds
        progress = None
        if quiet:
            progress = f"[{event_log['condition']} seed={event_log['seed']}] round {round_number}/{num_rounds}"

        partial = event_log.get("partial_round")
        if not partial or partial["round"] != round_number:
            partial = {"round": round_number, "candidate_statements": {}, "polls": {}}
            event_log["partial_round"] = partial

        statements = get_candidate_statements(
            candidates, ground_truth, round_number, event_log["rounds"],
            partial["candidate_statements"], save, progress,
        )
        candidate_a_statement = statements["candidate_1"]["statement"]
        candidate_b_statement = statements["candidate_2"]["statement"]

        polls = get_polls(
            voters, ground_truth, round_number, event_log["rounds"], candidate_a_statement, candidate_b_statement,
            last_beliefs_by_voter, is_final_round, partial["polls"], save, progress,
        )

        for voter_id, poll in polls.items():
            previous = last_beliefs_by_voter[voter_id]
            if previous is None:
                v = voters_by_id[voter_id]
                previous = {
                    "perceived_level": v.voter["perceived_level"],
                    "perceived_attribution": v.voter["perceived_attribution"],
                }
            last_beliefs_by_voter[voter_id] = parse_private_perceptions(poll.get("private_perceptions"), previous)

        event_log["rounds"].append({
            "round": round_number,
            "candidate_statements": statements,
            "polls": polls,
            # Each voter's beliefs after this round; resume reads the last round's.
            "beliefs": dict(last_beliefs_by_voter),
        })
        del event_log["partial_round"]
        save()

    event_log["result"] = tally_votes(list(event_log["rounds"][-1]["polls"].values()), ground_truth["candidate_names"])
    save()

    return event_log


def run_simulation(seed=None, condition="neutral", model=MODEL, quiet=False):
    """
    Plays a full run and returns its event log. Saves
    output/<model>/run_<condition>_<seed>.json after every LLM response (overwriting
    any existing file of that name). A random seed is picked if none is given.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    config = CONDITIONS[condition]
    ground_truth, voter_dicts = generate_world_state(seed, config)

    event_log = {
        "seed": seed,
        "condition": condition,
        "condition_config": config,
        "model": model,
        "think": THINK,
        "num_rounds": NUM_ROUNDS,
        "candidate_word_budget": CANDIDATE_WORD_BUDGET,
        "voter_explanation_word_budget": VOTER_EXPLANATION_WORD_BUDGET,
        "ground_truth": ground_truth,
        "voter_dicts": voter_dicts,
        "rounds": [],
        "duration_seconds": 0.0,
    }
    candidates, voters = build_agents(event_log)
    last_beliefs_by_voter = {v.voter_id: None for v in voters}

    json_path = run_path(condition, seed, "json", model)
    return run_rounds(event_log, candidates, voters, last_beliefs_by_voter, quiet, json_path)


def resume_simulation(path, quiet=False):
    """
    Continues a run from its saved JSON at the first unfinished round, using the
    world, voters and settings stored in it, and keeps saving to that same path.
    Returns the finished event log. A run that already finished is returned as is.
    """
    with open(path, "r", encoding="utf-8") as f:
        event_log = json.load(f)

    if "result" in event_log:
        print(f"{path} already finished; nothing to resume.")
        return event_log

    candidates, voters = build_agents(event_log)
    last_beliefs_by_voter = {v.voter_id: None for v in voters}
    if event_log["rounds"]:
        last_beliefs_by_voter.update(event_log["rounds"][-1]["beliefs"])

    print(f"Resuming {path} at round {len(event_log['rounds']) + 1}/{event_log['num_rounds']}.")
    return run_rounds(event_log, candidates, voters, last_beliefs_by_voter, quiet, path)
