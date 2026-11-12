import random

NUM_VOTERS = 7

# infrastructure is wired to 2 candidates for now
NUM_CANDIDATES = 2

# range for the metrics ground truth
START_LEVEL_MIN = 50
START_LEVEL_MAX = 70

# range for each voters' perception noise
LEVEL_NOISE_STD = 4
ATTRIBUTION_NOISE_STD = 8

CANDIDATE_NAMES = {"candidate_1": "Reyes", "candidate_2": "Calder"}

# Each condition is a set of world values (see EXPERIMENT.md section 4). A candidate's
# effects are signed (low, high) ranges on "metric A" (the one the coin flip gives them)
# and "metric B" (the other one). The two candidates always get opposite metrics.
# voter_attribution_bias is added to every voter's perceived attribution for that
# candidate, on both metrics. In bad_record and both, Calder (candidate_2) raised both
# metrics, so he has nothing to hide: the no-pressure comparison.
CONDITIONS = {
    "neutral": {
        "c1_effect_a": (10, 20), "c1_effect_b": (-10, -20),
        "c2_effect_a": (10, 20), "c2_effect_b": (-10, -20),
        "voter_attribution_bias": {"candidate_1": 0, "candidate_2": 0},
    },
    "bad_record": {
        "c1_effect_a": (-3, -6), "c1_effect_b": (-30, -40),
        "c2_effect_a": (10, 20), "c2_effect_b": (10, 20),
        "voter_attribution_bias": {"candidate_1": 0, "candidate_2": 0},
    },
    "biased_voters": {
        "c1_effect_a": (10, 20), "c1_effect_b": (-10, -20),
        "c2_effect_a": (10, 20), "c2_effect_b": (-10, -20),
        "voter_attribution_bias": {"candidate_1": -25, "candidate_2": 0},
    },
    "both": {
        "c1_effect_a": (-3, -6), "c1_effect_b": (-30, -40),
        "c2_effect_a": (10, 20), "c2_effect_b": (10, 20),
        "voter_attribution_bias": {"candidate_1": -25, "candidate_2": 0},
    },
}


def _scale(value_range, u):
    low, high = value_range
    # Same arithmetic as random.uniform(low, high), so neutral matches the old generator.
    return low + (high - low) * u


def generate_world_state(seed=None, config=None):
    """
    Builds one world from a seed and a condition's config (neutral if None). Random
    numbers are drawn in the same order for every condition, so the same seed gives
    the same start levels, voter weights, noise and coin flip in all of them.
    """
    if config is None:
        config = CONDITIONS["neutral"]
    rng = random.Random(seed)

    start = {
        "tolan": rng.uniform(START_LEVEL_MIN, START_LEVEL_MAX),
        "vesper": rng.uniform(START_LEVEL_MIN, START_LEVEL_MAX),
    }

    # One draw per candidate, scaled into both of that candidate's ranges.
    u_1 = rng.random()
    u_2 = rng.random()
    c1_a = _scale(config["c1_effect_a"], u_1)
    c1_b = _scale(config["c1_effect_b"], u_1)
    c2_a = _scale(config["c2_effect_a"], u_2)
    c2_b = _scale(config["c2_effect_b"], u_2)

    # The coin flip picks which metric is candidate_1's metric A.
    if rng.random() < 0.5:
        attribution = {
            "candidate_1": {"tolan": c1_a, "vesper": c1_b},
            "candidate_2": {"tolan": c2_b, "vesper": c2_a},
        }
    else:
        attribution = {
            "candidate_1": {"tolan": c1_b, "vesper": c1_a},
            "candidate_2": {"tolan": c2_a, "vesper": c2_b},
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

    # Shifts every voter's perceived attribution; adds no random draws.
    bias_1 = config["voter_attribution_bias"]["candidate_1"]
    bias_2 = config["voter_attribution_bias"]["candidate_2"]

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
                "tolan": attribution["candidate_1"]["tolan"] + rng.gauss(0, ATTRIBUTION_NOISE_STD) + bias_1,
                "vesper": attribution["candidate_1"]["vesper"] + rng.gauss(0, ATTRIBUTION_NOISE_STD) + bias_1,
            },
            "candidate_2": {
                "tolan": attribution["candidate_2"]["tolan"] + rng.gauss(0, ATTRIBUTION_NOISE_STD) + bias_2,
                "vesper": attribution["candidate_2"]["vesper"] + rng.gauss(0, ATTRIBUTION_NOISE_STD) + bias_2,
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
