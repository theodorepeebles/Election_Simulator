"""
Runs every condition x seed, unattended.

    python batch.py                                    # all conditions x seeds 101-125, default model
    python batch.py --seeds 101-105                    # pilot
    python batch.py --conditions neutral both --seeds 101 103 --model nemotron-3-ultra

Finished runs are skipped, half-finished ones are resumed, and a failed run is
logged to output/<model>/batch_failures.log before moving on. The batch only
stops early on a bad API key, no access to the model, or an account out of
credit or over its monthly limit, since every other run would fail the same way. Re-running the same command picks up where it left off.
"""
import argparse
import datetime
import json
import os
import sys
import traceback

from engine import run_simulation, resume_simulation, MODEL
from llm import verify_api_key, FatalLLMError
from txt_viewer import render_event_log, run_path, model_slug, OUTPUT_DIR
from world_config import CONDITIONS


def parse_seeds(items):
    """Accepts numbers and inclusive ranges: ["101-103", "110"] -> [101, 102, 103, 110]."""
    seeds = []
    for item in items:
        if "-" in item:
            low, high = item.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(item))
    return seeds


def log_failure(log_path, condition, seed, error):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== {datetime.datetime.now().isoformat(timespec='seconds')} "
                f"condition={condition} seed={seed} ===\n")
        f.write(f"{error}\n")
        f.write("".join(traceback.format_exception(error)))
        f.write("\n")


def run_one(condition, seed, model):
    """Skips, resumes or starts one run. Returns "skipped", "resumed" or "ran"."""
    json_path = run_path(condition, seed, "json", model)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            event_log = json.load(f)
        if "result" in event_log:
            if not os.path.exists(os.path.splitext(json_path)[0] + ".txt"):
                render_event_log(event_log, quiet=True, json_path=json_path)
            return "skipped"
        event_log = resume_simulation(json_path, quiet=True)
        outcome = "resumed"
    else:
        event_log = run_simulation(seed=seed, condition=condition, model=model, quiet=True)
        outcome = "ran"

    render_event_log(event_log, quiet=True, json_path=json_path)
    return outcome


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run every condition x seed, skipping finished runs.")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", default=["101-125"], help="numbers and/or ranges like 101-125")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    verify_api_key()
    seeds = parse_seeds(args.seeds)
    log_path = os.path.join(OUTPUT_DIR, model_slug(args.model), "batch_failures.log")

    # Seed by seed, so a batch stopped early still leaves complete sets across conditions.
    jobs = [(condition, seed) for seed in seeds for condition in args.conditions]
    counts = {"ran": 0, "resumed": 0, "skipped": 0, "failed": 0}

    for i, (condition, seed) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {condition} seed={seed}", flush=True)
        try:
            outcome = run_one(condition, seed, args.model)
        except FatalLLMError as e:
            log_failure(log_path, condition, seed, e)
            print(f"Stopping the batch: {e}")
            break
        except Exception as e:
            log_failure(log_path, condition, seed, e)
            print(f"  FAILED: {e} (logged to {log_path})", flush=True)
            counts["failed"] += 1
            continue
        counts[outcome] += 1
        print(f"  {outcome}", flush=True)

    print(f"Done. ran={counts['ran']} resumed={counts['resumed']} "
          f"skipped={counts['skipped']} failed={counts['failed']}")
