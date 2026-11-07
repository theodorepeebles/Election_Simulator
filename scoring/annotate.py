"""
Human labels for checking the LLM judge.

    python annotate.py ../output/gpt-oss-120b/scores/nemotron-3-ultra/score_neutral_9001.json

Goes through the claims one at a time. For each, it rewrites
annotations/annotation_<condition>_<seed>.md (next to the score file) with only what's
relevant to that claim: the claim, then exactly what the judge saw for its statement. The
judge's labels are hidden, so your labels are independent. Keep its preview open beside the
terminal. The terminal shows only the claim's ID and quote and asks for your labels. When
you stop or finish (and when you come back), it reports how often you agreed with the judge. Answers are saved after each one to annotation_<condition>_<seed>.json;
re-running skips claims you've already labeled. Score and run files are only read.
"""
import json
import os
import sys

from judge import format_ground_truth_table
from score import save_json
from checks import parse_number

# Fixed order, so the number keys never change.
STEP1_ORDER = ["TRUE", "DIRECTION", "NUMBER", "ATTRIBUTION", "OUTSIDE_CAUSE", "SKIP"]
STEP2_ORDER = ["STATED", "UNSTATED", "ERROR", "UNCLEAR"]
NOT_FALSE = {"TRUE", "SKIP"}


def menu(labels):
    return ", ".join(f"{i} {label}" for i, label in enumerate(labels))


def claim_id(statement_id, number):
    return f"{statement_id} #{number}"


def quote_block(text):
    return "\n".join(f"> {line}" for line in (text or "(none)").splitlines())


def write_claim_md(score, event_log, statement_id, number, c, path):
    """Rewrites the md with only what's relevant to this one claim: the claim, then the
    same context the judge saw for its statement (table, statement, PLAN and thinking for
    this round and earlier). The judge's labels are left out so yours are independent."""
    names = event_log["ground_truth"]["candidate_names"]
    round_part, candidate_key = statement_id.split("_", 1)
    round_number = int(round_part[1:])
    name = names[candidate_key]
    rounds = event_log["rounds"]

    lines = [
        f"# [{claim_id(statement_id, number)}] {name}, round {round_number}",
        f"*{score['condition']}, seed {score['seed']}*",
        "",
        quote_block(c["claim"]),
        "",
    ]
    # One sentence can hold several claims ("from 50.7 to 50.8" is start, current and change).
    # Show which number this one is about; anything more would hint at the judge's labels.
    if parse_number(c.get("claimed_value")) is not None:
        lines += [f"**This claim is about the number:** {c['claimed_value']}", ""]
    lines += [
        f"**Step 1:** {menu(STEP1_ORDER)}  ",
        f"**Step 2** (false claims only): {menu(STEP2_ORDER)}",
        "",
        "## True record",
        "```",
        format_ground_truth_table(event_log["ground_truth"]),
        "```",
        "",
        f"## {name}'s statement, round {round_number}",
        quote_block(rounds[round_number - 1]["candidate_statements"][candidate_key]["statement"]),
        "",
        f"## {name}'s private reasoning, rounds 1–{round_number}",
    ]
    for entry in rounds[:round_number]:
        result = entry["candidate_statements"][candidate_key]
        lines += [
            "", f"### Round {entry['round']} PLAN", "", result.get("plan") or "(none)",
            "", f"### Round {entry['round']} thinking", "", result.get("thinking") or "(none)",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def report_agreement(score, claims, answers, md_path):
    """Prints how often your labels match the judge's final labels, over the claims
    labeled so far, and writes the same report to md_path."""
    labeled = [(claim_id(sid, n), c, answers[claim_id(sid, n)]) for sid, n, c in claims
               if claim_id(sid, n) in answers]
    if not labeled:
        return

    def status(step1):
        return step1 if step1 in NOT_FALSE else "FALSE"

    def pct(agree, total):
        return f"{agree}/{total} ({agree / total:.0%})"

    step1_agree = sum(a["step1"] == c["final_step1"] for _, c, a in labeled)
    status_agree = sum(status(a["step1"]) == status(c["final_step1"]) for _, c, a in labeled)
    both_false = [(cid, c, a) for cid, c, a in labeled
                  if a["step1"] not in NOT_FALSE and c["final_step1"] not in NOT_FALSE]
    step2_agree = sum(a["step2"] == c["final_step2"] for _, c, a in both_false)
    disagreements = [(cid, c, a) for cid, c, a in labeled
                     if a["step1"] != c["final_step1"] or a["step2"] != c["final_step2"]]

    n = len(labeled)
    lines = [
        f"# Agreement with the judge: {score['condition']} seed {score['seed']}",
        "",
        f"Judge: {score['judge_model']}. Claims labeled: {n} of {len(claims)}.",
        "",
        f"- true / false / skip: {pct(status_agree, n)}",
        f"- exact step 1: {pct(step1_agree, n)}",
    ]
    if both_false:
        lines.append(f"- step 2, where you both said false: {pct(step2_agree, len(both_false))}")
    if disagreements:
        lines += ["", "## Disagreements (you vs. judge)", ""]
        for cid, c, a in disagreements:
            lines.append(f"- [{cid}] {a['step1']}/{a['step2'] or '-'} vs. "
                         f"{c['final_step1']}/{c['final_step2'] or '-'}: \"{c['claim']}\"")

    print("\n" + "\n".join(lines))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n(saved to {md_path})")


def ask(prompt, labels):
    """Returns the chosen label, or None to quit. Asks again on anything invalid."""
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return None
        if answer == "q":
            return None
        if answer.isdigit() and int(answer) < len(labels):
            return labels[int(answer)]
        print("  type one of the numbers shown, or q to quit")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) != 2:
        sys.exit("usage: python annotate.py <score_file.json>")
    score_file = sys.argv[1]

    with open(score_file, "r", encoding="utf-8") as f:
        score = json.load(f)
    with open(score["run_path"], "r", encoding="utf-8") as f:
        event_log = json.load(f)

    folder = os.path.join(os.path.dirname(os.path.abspath(score_file)), "annotations")
    os.makedirs(folder, exist_ok=True)
    base = f"annotation_{score['condition']}_{score['seed']}"
    claim_md_path = os.path.join(folder, base + ".md")
    answers_path = os.path.join(folder, base + ".json")
    agreement_path = os.path.join(folder, f"agreement_{score['condition']}_{score['seed']}.md")

    print(f"Open this next to the terminal (it changes with each claim): {claim_md_path}")

    answers = {}
    if os.path.exists(answers_path):
        with open(answers_path, "r", encoding="utf-8") as f:
            answers = json.load(f)

    # Statements the judge failed on have no claims, so there's nothing to label there.
    claims = [
        (statement_id, number, c)
        for statement_id, result in score["statements"].items()
        for number, c in enumerate(result.get("claims", []), start=1)
    ]
    todo = [item for item in claims if claim_id(item[0], item[1]) not in answers]
    print(f"{len(claims) - len(todo)} of {len(claims)} claims already labeled.")
    if len(todo) < len(claims):
        report_agreement(score, claims, answers, agreement_path)

    for statement_id, number, c in todo:
        cid = claim_id(statement_id, number)
        write_claim_md(score, event_log, statement_id, number, c, claim_md_path)
        print(f"\n[{cid}] \"{c['claim']}\"")
        step1 = ask(f"step 1 ({menu(STEP1_ORDER)}, q quit): ", STEP1_ORDER)
        if step1 is None:
            break
        step2 = None
        if step1 not in NOT_FALSE:
            step2 = ask(f"step 2 ({menu(STEP2_ORDER)}, q quit): ", STEP2_ORDER)
            if step2 is None:
                break
        answers[cid] = {"step1": step1, "step2": step2}
        save_json(answers, answers_path)

    print(f"\n{len(answers)} of {len(claims)} claims labeled. Saved to {answers_path}")
    report_agreement(score, claims, answers, agreement_path)
