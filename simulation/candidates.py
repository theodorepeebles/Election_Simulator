from prompts import build_politician_system_prompt, build_politician_round_prompt
from llm import ask_llm, extract_marked_fields

CANDIDATE_MARKERS = ["PLAN:", "STATEMENT:"]


class Candidate:
    def __init__(self, candidate_key, ground_truth, num_rounds, word_budget, model, think=False):
        self.candidate_key = candidate_key
        self.name = ground_truth["candidate_names"][candidate_key]
        self.ground_truth = ground_truth
        self.num_rounds = num_rounds
        self.word_budget = word_budget
        self.model = model
        self.think = think
        self.system_prompt = build_politician_system_prompt(candidate_key, num_rounds, word_budget, ground_truth)

    def give_statement(self, round_number, history_text=""):
        prompt = build_politician_round_prompt(round_number, self.num_rounds, history_text)
        raw, thinking = ask_llm(prompt, self.system_prompt, self.model, think=self.think)
        fields = extract_marked_fields(raw, CANDIDATE_MARKERS)
        fields["thinking"] = thinking
        return fields
