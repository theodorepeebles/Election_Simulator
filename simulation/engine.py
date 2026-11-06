import random
import time

from simulation.world_config import generate_world_state, NUM_VOTERS, NUM_CANDIDATES
from simulation.candidates import Candidate
from simulation.voters import Voter, parse_private_perceptions

MODEL = "gemma4:31b-cloud"
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


def get_candidate_statements(candidates, ground_truth, round_number, rounds_so_far):
    statements = {}
    for key, candidate in candidates.items():
        history_text = build_candidate_history_text(key, ground_truth, rounds_so_far)
        result = candidate.give_statement(round_number, history_text)
        result["candidate_key"] = key
        result["name"] = candidate.name
        statements[key] = result
    return statements


def get_polls(voters, ground_truth, round_number, rounds_so_far, candidate_a_statement, candidate_b_statement,
              last_beliefs_by_voter, is_final_round):
    history_text = build_voter_history_text(ground_truth, rounds_so_far)
    polls = {}
    for voter in voters:
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


def run_simulation():
    start_time = time.time()

    seed = random.randint(0, 2**32 - 1)
    ground_truth, voter_dicts = generate_world_state(seed)

    candidate_keys = [f"candidate_{i}" for i in range(1, NUM_CANDIDATES + 1)]
    candidates = {
        key: Candidate(key, ground_truth, NUM_ROUNDS, CANDIDATE_WORD_BUDGET, MODEL)
        for key in candidate_keys
    }
    voters = [Voter(v, NUM_ROUNDS, VOTER_EXPLANATION_WORD_BUDGET, MODEL) for v in voter_dicts]
    voters_by_id = {v.voter_id: v for v in voters}
    last_beliefs_by_voter = {v.voter_id: None for v in voters}

    event_log = {"seed": seed, "model": MODEL, "ground_truth": ground_truth, "voter_dicts": voter_dicts, "rounds": []}

    for round_number in range(1, NUM_ROUNDS + 1):
        is_final_round = round_number == NUM_ROUNDS

        statements = get_candidate_statements(candidates, ground_truth, round_number, event_log["rounds"])
        candidate_a_statement = statements["candidate_1"]["statement"]
        candidate_b_statement = statements["candidate_2"]["statement"]

        polls = get_polls(
            voters, ground_truth, round_number, event_log["rounds"], candidate_a_statement, candidate_b_statement,
            last_beliefs_by_voter, is_final_round,
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
        })

    event_log["result"] = tally_votes(list(event_log["rounds"][-1]["polls"].values()), ground_truth["candidate_names"])
    event_log["duration_seconds"] = time.time() - start_time

    return event_log
