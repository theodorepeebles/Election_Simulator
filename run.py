from engine import run_simulation
from viewer import render_event_log

if __name__ == "__main__":
    event_log = run_simulation()
    render_event_log(event_log)
