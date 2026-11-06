from simulation.engine import run_simulation
from simulation.llm import verify_api_key
from simulation.txt_viewer import render_event_log

if __name__ == "__main__":
    verify_api_key()

    event_log = run_simulation()
    render_event_log(event_log)
