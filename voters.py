from prompts import build_voter_system_prompt, build_voter_prompt
from llm import ask_llm, extract_marked_fields
from world_config import CANDIDATE_NAMES

VOTE_MARKERS = ["MY VOTE GOES TO:", "EXPLANATION:", "PRIVATE PERCEPTIONS:"]
FINAL_VOTE_MARKERS = ["MY VOTE GOES TO:", "EXPLANATION:"]


class Voter:
    def __init__(self, voter, num_rounds, explanation_word_budget):
        self.voter_id = voter["voter_id"]
        self.name = voter["name"]
        self.voter = voter
        self.num_rounds = num_rounds
        self.explanation_word_budget = explanation_word_budget
        self.system_prompt = build_voter_system_prompt(voter, num_rounds, explanation_word_budget)

    def run_poll(
        self,
        round_number,
        candidate_a_statement,
        candidate_b_statement,
        history_text,
        last_beliefs=None,
        is_final_round=False,
    ):
        prompt = build_voter_prompt(
            self.voter,
            round_number,
            self.num_rounds,
            candidate_a_statement,
            candidate_b_statement,
            history_text,
            self.explanation_word_budget,
            last_beliefs,
            is_final_round,
        )
        raw = ask_llm(prompt, self.system_prompt)
        markers = FINAL_VOTE_MARKERS if is_final_round else VOTE_MARKERS
        return extract_marked_fields(raw, markers)


def parse_private_perceptions(text):
    """
    Turns a voter's raw PRIVATE PERCEPTIONS text block (one "key: value" line
    per field, per the format requested in the voter system prompt) into the
    structured dict shape build_voter_prompt expects for last_beliefs.
    Missing/unparseable fields come back as None.
    """
    if not text:
        return None

    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            continue

    c1 = CANDIDATE_NAMES["candidate_1"]
    c2 = CANDIDATE_NAMES["candidate_2"]

    return {
        "perceived_level": {
            "tolan": values.get("level_tolan"),
            "vesper": values.get("level_vesper"),
        },
        "perceived_attribution": {
            "candidate_1": {
                "tolan": values.get(f"{c1}_attribution_tolan"),
                "vesper": values.get(f"{c1}_attribution_vesper"),
            },
            "candidate_2": {
                "tolan": values.get(f"{c2}_attribution_tolan"),
                "vesper": values.get(f"{c2}_attribution_vesper"),
            },
        },
    }
