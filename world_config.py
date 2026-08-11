import random

NUM_VOTERS = 3

# infrastructure is wired to 2 candidates for now
NUM_CANDIDATES = 2

# range for the metrics ground truth
START_LEVEL_MIN = 50
START_LEVEL_MAX = 70

# range for the candidates attribution ground truth
CANDIDATE_MAGNITUDE_MIN = 10
CANDIDATE_MAGNITUDE_MAX = 20

# range for each voters' perception noise
LEVEL_NOISE_STD = 4
ATTRIBUTION_NOISE_STD = 8

CANDIDATE_NAMES = {"candidate_1": "Reyes", "candidate_2": "Calder"}


def generate_world_state(seed=None):
    rng = random.Random(seed)

    start = {
        "tolan": rng.uniform(START_LEVEL_MIN, START_LEVEL_MAX),
        "vesper": rng.uniform(START_LEVEL_MIN, START_LEVEL_MAX),
    }

    mag_1 = rng.uniform(CANDIDATE_MAGNITUDE_MIN, CANDIDATE_MAGNITUDE_MAX)
    mag_2 = rng.uniform(CANDIDATE_MAGNITUDE_MIN, CANDIDATE_MAGNITUDE_MAX)

    # candidates help one metric as much as they hurt the other 
    if rng.random() < 0.5:
        attribution = {
            "candidate_1": {"tolan": mag_1, "vesper": -mag_1},
            "candidate_2": {"tolan": -mag_2, "vesper": mag_2},
        }
    else:
        attribution = {
            "candidate_1": {"tolan": -mag_1, "vesper": mag_1},
            "candidate_2": {"tolan": mag_2, "vesper": -mag_2},
        }

    # only relevant for constructing the world, and for candidate claims i.e. "tolan started here and I brought it up to here"
    change = {
        "tolan": attribution["candidate_1"]["tolan"] + attribution["candidate_2"]["tolan"],
        "vesper": attribution["candidate_1"]["vesper"] + attribution["candidate_2"]["vesper"],
    }

    now = {
        "tolan": start["tolan"] + change["tolan"],
        "vesper": start["vesper"] + change["vesper"],
    }

    ground_truth = {
        "seed": seed,
        "metrics": ["tolan", "vesper"],
        "candidates": ["candidate_1", "candidate_2"],
        "candidate_names": dict(CANDIDATE_NAMES),
        "start": start,
        "now": now,
        "change": change,
        "attribution": attribution,
    }

    # create the voters' starting stats
    voter_dicts = []
    for i in range(1, NUM_VOTERS + 1):
        w_tolan = rng.uniform(0, 1)
        w_vesper = 1 - w_tolan

        perceived_level = {
            "tolan": now["tolan"] + rng.gauss(0, LEVEL_NOISE_STD),
            "vesper": now["vesper"] + rng.gauss(0, LEVEL_NOISE_STD),
        }

        perceived_attribution = {
            "candidate_1": {
                "tolan": attribution["candidate_1"]["tolan"] + rng.gauss(0, ATTRIBUTION_NOISE_STD),
                "vesper": attribution["candidate_1"]["vesper"] + rng.gauss(0, ATTRIBUTION_NOISE_STD),
            },
            "candidate_2": {
                "tolan": attribution["candidate_2"]["tolan"] + rng.gauss(0, ATTRIBUTION_NOISE_STD),
                "vesper": attribution["candidate_2"]["vesper"] + rng.gauss(0, ATTRIBUTION_NOISE_STD),
            },
        }

        voter_dicts.append({
            "voter_id": f"voter_{i}",
            "name": f"Voter {i}",
            "weights": {"tolan": w_tolan, "vesper": w_vesper},
            "perceived_level": perceived_level,
            "perceived_attribution": perceived_attribution,
        })

    return ground_truth, voter_dicts
