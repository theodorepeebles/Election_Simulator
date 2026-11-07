"""
    python run.py                                     # new run, random seed, condition "neutral"
    python run.py --seed 123 --condition bad_record   # new run with a set seed and condition
    python run.py --quiet                             # one progress line per LLM call
    python run.py ../output/gpt-oss-120b/run_neutral_123.json   # resume a stopped run from its JSON
"""
import argparse
import os
import sys

from engine import run_simulation, resume_simulation, MODEL
from llm import verify_api_key
from txt_viewer import render_event_log, run_path
from world_config import CONDITIONS

if __name__ == "__main__":
    # Model output can contain characters like U+2011 that crash the default
    # Windows encoding when output is redirected to a file.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run one election, or resume one from its JSON.")
    parser.add_argument("resume_path", nargs="?", help="a run's JSON file to resume")
    parser.add_argument("--seed", type=int, help="world seed (random if not given)")
    parser.add_argument("--condition", default="neutral", choices=list(CONDITIONS),
                        help="world condition (see EXPERIMENT.md), saved in the run log and file name")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--quiet", action="store_true", help="no token-by-token printing")
    args = parser.parse_args()

    verify_api_key()

    if args.resume_path:
        json_path = args.resume_path
        event_log = resume_simulation(json_path, quiet=args.quiet)
    else:
        json_path = None
        if args.seed is not None:
            json_path = run_path(args.condition, args.seed, "json", args.model)
            if os.path.exists(json_path):
                print(f"{json_path} already exists. To resume it instead, run: python run.py \"{json_path}\"")
                try:
                    answer = input("Overwrite it with a fresh run? (y/n) ")
                except EOFError:
                    answer = ""
                if answer.strip().lower() != "y":
                    sys.exit("Not overwritten.")
        event_log = run_simulation(seed=args.seed, condition=args.condition, model=args.model, quiet=args.quiet)

    render_event_log(event_log, quiet=args.quiet, json_path=json_path)
