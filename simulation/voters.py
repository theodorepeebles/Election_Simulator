from prompts import build_voter_system_prompt, build_voter_prompt
from llm import ask_llm, extract_marked_fields
from world_config import CANDIDATE_NAMES

VOTE_MARKERS = ["MY VOTE GOES TO:", "EXPLANATION:", "PRIVATE PERCEPTIONS:"]
FINAL_VOTE_MARKERS = ["MY VOTE GOES TO:", "EXPLANATION:"]


class Voter:
    def __init__(self, voter, num_rounds, explanation_word_budget, model, think=False):
        self.voter_id = voter["voter_id"]
        self.name = voter["name"]
        self.voter = voter
        self.num_rounds = num_rounds
        self.explanation_word_budget = explanation_word_budget
        self.model = model
        self.think = think
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
        raw, thinking = ask_llm(prompt, self.system_prompt, self.model, think=self.think)
        markers = FINAL_VOTE_MARKERS if is_final_round else VOTE_MARKERS
        fields = extract_marked_fields(raw, markers)
        fields["thinking"] = thinking
        return fields


def parse_private_perceptions(text, previous=None):
    """
    Turns a voter's raw PRIVATE PERCEPTIONS text block (one "key: value" line
    per field, per the format requested in the voter system prompt) into the
    structured dict shape build_voter_prompt expects for last_beliefs.
    A line whose value is UNCHANGED (case-insensitive) falls back to that
    field's value in `previous`. Missing/unparseable fields also fall back to
    `previous`, or None if there's nothing to fall back to.
    """
    if not text:
        return previous

    raw_values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        raw_values[key.strip()] = value.strip()

    c1 = CANDIDATE_NAMES["candidate_1"]
    c2 = CANDIDATE_NAMES["candidate_2"]

    prev_level = (previous or {}).get("perceived_level", {})
    prev_attr = (previous or {}).get("perceived_attribution", {})

    def resolve(field_key, fallback):
        raw = raw_values.get(field_key)
        if raw is None:
            return fallback
        if raw.upper() == "UNCHANGED":
            return fallback
        try:
            return float(raw)
        except ValueError:
            return fallback

    return {
        "perceived_level": {
            "tolan": resolve("level_tolan", prev_level.get("tolan")),
            "vesper": resolve("level_vesper", prev_level.get("vesper")),
        },
        "perceived_attribution": {
            "candidate_1": {
                "tolan": resolve(f"{c1}_attribution_tolan", prev_attr.get("candidate_1", {}).get("tolan")),
                "vesper": resolve(f"{c1}_attribution_vesper", prev_attr.get("candidate_1", {}).get("vesper")),
            },
            "candidate_2": {
                "tolan": resolve(f"{c2}_attribution_tolan", prev_attr.get("candidate_2", {}).get("tolan")),
                "vesper": resolve(f"{c2}_attribution_vesper", prev_attr.get("candidate_2", {}).get("vesper")),
            },
        },
    }
