from prompts import build_politician_system_prompt, build_politician_round_prompt
from llm import ask_llm, extract_marked_fields

STATEMENT_MARKERS = ["STATEMENT:"]


class Candidate:
    def __init__(self, candidate_key, ground_truth):
        self.candidate_key = candidate_key
        self.name = ground_truth["candidate_names"][candidate_key]
        self.ground_truth = ground_truth

    def give_statement(self, round_number, num_rounds, word_budget, history_text=""):
        system = build_politician_system_prompt(self.candidate_key, num_rounds, word_budget, self.ground_truth)
        prompt = build_politician_round_prompt(round_number, num_rounds, history_text)
        raw = ask_llm(prompt, system)
        return extract_marked_fields(raw, STATEMENT_MARKERS)
