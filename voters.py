from prompts import build_voter_system_prompt, build_voter_prompt
from llm import ask_llm, extract_marked_fields

VOTE_MARKERS = ["MY VOTE GOES TO:", "EXPLANATION:", "PRIVATE PERCEPTIONS:"]
FINAL_VOTE_MARKERS = ["MY VOTE GOES TO:", "EXPLANATION:"]


class Voter:
    def __init__(self, voter):
        self.voter_id = voter["voter_id"]
        self.name = voter["name"]
        self.voter = voter

    def run_poll(
        self,
        round_number,
        num_rounds,
        candidate_a_statement,
        candidate_b_statement,
        history_text,
        explanation_word_budget,
        last_beliefs=None,
        is_final_round=False,
    ):
        system = build_voter_system_prompt(self.voter, num_rounds, explanation_word_budget)
        prompt = build_voter_prompt(
            self.voter,
            round_number,
            num_rounds,
            candidate_a_statement,
            candidate_b_statement,
            history_text,
            explanation_word_budget,
            last_beliefs,
            is_final_round,
        )
        raw = ask_llm(prompt, system)
        markers = FINAL_VOTE_MARKERS if is_final_round else VOTE_MARKERS
        return extract_marked_fields(raw, markers)
