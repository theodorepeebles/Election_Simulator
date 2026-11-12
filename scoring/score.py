"""
Scores finished runs with the LLM judge plus the Python checks.

    python score.py                                            # every finished run under output/
    python score.py ../output/deepseek-v4.1-flash/run_both_101.json   # just these runs
    python score.py --rescore ...                              # redo even finished score files
    python score.py --quiet                                    # one line per judge call, no token streaming

Each run gets <run's folder>/scores/<judge model>/score_<condition>_<seed>.json,
saved after every statement. A judge call that fails is saved in that file as an
error; the next `python score.py` retries just those statements. Run files are
only read, never written.
"""
import argparse
import glob
import json
import os
import sys
import time

from judge import JUDGE_MODEL, JUDGE_THINK, judge_statement, reasoning_text
from checks import finalize_claim, summarize
from llm import FatalLLMError, set_quiet, verify_api_key
from txt_viewer import OUTPUT_DIR, model_slug


def score_path(run_json_path, event_log):
    """<run folder>/scores/<judge>/score_<run name>.json, named after the run file
    (run_neutral_103_c-high_v-medium.json -> score_neutral_103_c-high_v-medium.json), so
    runs that differ only in thinking level get separate score files."""
    folder = os.path.join(os.path.dirname(run_json_path), "scores", model_slug(JUDGE_MODEL))
    name = os.path.basename(run_json_path)
    if name.startswith("run_"):
        name = name[len("run_"):]
    return os.path.join(folder, f"score_{name}")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def why_not_scorable(event_log):
    rounds = event_log.get("rounds") or []
    if not rounds or "candidate_statements" not in rounds[0]:
        return "old run format"
    if "result" not in event_log:
        return "run not finished"
    return None


def score_run(run_json_path, rescore=False, quiet=False):
    """Scores one run. Returns "scored", "skipped" or "incomplete" (some statements failed)."""
    with open(run_json_path, "r", encoding="utf-8") as f:
        event_log = json.load(f)

    reason = why_not_scorable(event_log)
    if reason:
        print(f"  skipped: {reason}")
        return "skipped"

    out_path = score_path(run_json_path, event_log)
    score = None
    if os.path.exists(out_path) and not rescore:
        with open(out_path, "r", encoding="utf-8") as f:
            score = json.load(f)
        if score.get("complete"):
            print(f"  skipped: already scored ({out_path})")
            return "skipped"

    if score is None:
        score = {
            "run_path": os.path.abspath(run_json_path),
            "condition": event_log["condition"],
            "seed": event_log["seed"],
            "run_model": event_log["model"],
            "judge_model": JUDGE_MODEL,
            "judge_think": JUDGE_THINK,
            "statements": {},
        }

    ground_truth = event_log["ground_truth"]
    for entry in event_log["rounds"]:
        round_number = entry["round"]
        for candidate_key in entry["candidate_statements"]:
            key = f"r{round_number}_{candidate_key}"
            done = score["statements"].get(key)
            if done and "error" not in done:
                continue

            name = ground_truth["candidate_names"][candidate_key]
            label = f"  round {round_number} {name}"
            if not quiet:
                print(label, flush=True)
            start_time = time.time()
            try:
                result = judge_statement(event_log, candidate_key, round_number)
                reasoning = reasoning_text(event_log, candidate_key, round_number)
                result["claims"] = [finalize_claim(c, ground_truth, reasoning) for c in result["claims"]]
            except FatalLLMError:
                raise
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
                print(f"    FAILED: {result['error']}", flush=True)
            if quiet and "error" not in result:
                print(f"{label}  {len(result['claims'])} claims  {time.time() - start_time:.1f}s", flush=True)

            score["statements"][key] = result
            score["summary"] = summarize(score["statements"], event_log)
            score["complete"] = False
            save_json(score, out_path)

    score["summary"] = summarize(score["statements"], event_log)
    score["complete"] = not any("error" in s for s in score["statements"].values())
    save_json(score, out_path)
    print(f"  saved {out_path}")
    return "scored" if score["complete"] else "incomplete"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Score finished runs with the LLM judge.")
    parser.add_argument("paths", nargs="*", help="run JSON files (default: every run_*.json under output/)")
    parser.add_argument("--rescore", action="store_true", help="redo runs that already have a finished score file")
    parser.add_argument("--quiet", action="store_true", help="one line per judge call instead of streaming tokens")
    args = parser.parse_args()
    set_quiet(args.quiet)

    paths = args.paths or sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "run_*.json"), recursive=True))

    verify_api_key()
    counts = {"scored": 0, "skipped": 0, "incomplete": 0}
    for i, path in enumerate(paths, start=1):
        print(f"[{i}/{len(paths)}] {path}", flush=True)
        try:
            counts[score_run(path, rescore=args.rescore, quiet=args.quiet)] += 1
        except FatalLLMError as e:
            print(f"Stopping: {e}")
            break
        except Exception as e:
            # e.g. a run file that won't load; leave it alone and move on.
            print(f"  FAILED to read or score: {type(e).__name__}: {e}")
            counts["incomplete"] += 1

    print(f"Done. scored={counts['scored']} skipped={counts['skipped']} incomplete={counts['incomplete']}")
