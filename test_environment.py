"""
Test environment for matching algorithm optimization.
Evaluates and optimizes weights to maximize NDCG@10.
"""

import json
from scipy.optimize import differential_evolution, minimize
import numpy as np

from matching.scoring import score_property
from matching.evaluation import calculate_ndcg_at_k
from matching import weights

# Configuration
DATA_DIR = "data"
PROFILES_FILE = f"{DATA_DIR}/ground_truth_profiles.json"
PROPERTIES_FILE = f"{DATA_DIR}/synthetic_properties.json"
GROUND_TRUTH_FILE = f"{DATA_DIR}/my_ground_truth.json"

# Optimization parameters
WEIGHT_BOUNDS = (0, 200)
DE_MAX_ITER = 100
DE_POP_SIZE = 15
DE_TOLERANCE = 1e-6
LBFGS_MAX_ITER = 200
LBFGS_TOLERANCE = 1e-6


def load_ground_truth_profiles():
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("profiles", [])


def load_my_ground_truth():
    try:
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ground_truth", {})
    except FileNotFoundError:
        return {}


def load_synthetic_properties():
    with open(PROPERTIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_weights(weights_dict, ground_truth_matches=None):
    """
    Evaluate weights against ground truth profiles.
    Returns average NDCG@10 over all profiles (0-1).
    """
    original_weights = weights.BASE_WEIGHTS.copy()
    weights.BASE_WEIGHTS.update(weights_dict)

    try:
        profiles = load_ground_truth_profiles()
        properties = load_synthetic_properties()

        if ground_truth_matches is None:
            ground_truth_matches = load_my_ground_truth()

        if not ground_truth_matches:
            print("WARNING: No ground truth found!")
            print(f"Create {GROUND_TRUTH_FILE} with your defined good matches.")
            return 0.0

        total_ndcg = 0.0
        num_profiles = 0

        for gt_profile in profiles:
            profile_id = gt_profile["profile_id"]
            profile = gt_profile["profile"]
            good_matches = ground_truth_matches.get(profile_id, [])

            if not good_matches:
                continue

            scored_properties = []
            for prop in properties:
                score = score_property(prop, profile)
                scored_properties.append((prop["id"], score))

            scored_properties.sort(key=lambda x: x[1], reverse=True)
            predicted_ranks = [pid for pid, _ in scored_properties]
            ideal_ranks = [
                m["property_id"] for m in sorted(good_matches, key=lambda x: x["rank"])
            ]

            ndcg = calculate_ndcg_at_k(predicted_ranks, ideal_ranks, k=10)
            total_ndcg += ndcg
            num_profiles += 1

        return total_ndcg / num_profiles if num_profiles > 0 else 0.0

    finally:
        weights.BASE_WEIGHTS.update(original_weights)


def optimize_weights():
    """
    Optimize weights using scipy.optimize to maximize NDCG@10.
    Uses Differential Evolution (global) and L-BFGS-B (local), returns the best.
    """
    bounds = [WEIGHT_BOUNDS] * 4

    x0 = [
        weights.BASE_WEIGHTS["property_type"],
        weights.BASE_WEIGHTS["location"],
        weights.BASE_WEIGHTS["size"],
        weights.BASE_WEIGHTS["price"],
    ]

    def objective(weights_vector):
        weights_dict = {
            "property_type": weights_vector[0],
            "location": weights_vector[1],
            "size": weights_vector[2],
            "price": weights_vector[3],
        }
        return -evaluate_weights(weights_dict)

    def to_int_weights(x):
        return {
            "property_type": int(round(x[0])),
            "location": int(round(x[1])),
            "size": int(round(x[2])),
            "price": int(round(x[3])),
        }

    print("   Trying Differential Evolution (global optimization)...")
    result_de = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=DE_MAX_ITER,
        popsize=DE_POP_SIZE,
        tol=DE_TOLERANCE,
        polish=True,
    )
    ndcg_de = -result_de.fun
    weights_de = to_int_weights(result_de.x)
    print(f"   DE result: NDCG={ndcg_de:.4f}, weights={weights_de}")

    print("   Trying L-BFGS-B (local optimization)...")
    result_lbfgs = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": LBFGS_MAX_ITER, "ftol": LBFGS_TOLERANCE},
    )
    ndcg_lbfgs = -result_lbfgs.fun
    weights_lbfgs = to_int_weights(result_lbfgs.x)
    print(f"   L-BFGS-B result: NDCG={ndcg_lbfgs:.4f}, weights={weights_lbfgs}")

    if ndcg_de > ndcg_lbfgs:
        print("   Selected Differential Evolution result (better NDCG)")
        return weights_de
    else:
        print("   Selected L-BFGS-B result (better NDCG)")
        return weights_lbfgs


if __name__ == "__main__":
    print("=" * 60)
    print("MATCHING ALGORITHM OPTIMIZATION TEST")
    print("=" * 60)

    ground_truth = load_my_ground_truth()
    if not ground_truth:
        print("\nNO GROUND TRUTH FOUND!")
        print(
            f"\nYou must first create {GROUND_TRUTH_FILE} with your defined good matches."
        )
        print("See data/my_ground_truth.json.example for format.")
        print("\nSteps:")
        print("1. Analyze profiles in data/ground_truth_profiles.json")
        print("2. Analyze properties in data/synthetic_properties.json")
        print("3. For each profile, identify 3-5 good matches and rank them")
        print("4. Create data/my_ground_truth.json according to the format")
        print("\nSee candidate_task.md for more information.")
        exit(1)

    print("\n1. Testing baseline weights...")
    baseline_ndcg = evaluate_weights(weights.BASE_WEIGHTS)
    print(f"   Baseline NDCG@10: {baseline_ndcg:.4f}")
    print(f"   Baseline weights: {weights.BASE_WEIGHTS}")

    print("\n2. Optimization...")
    optimized_weights = optimize_weights()
    optimized_ndcg = evaluate_weights(optimized_weights)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Baseline NDCG@10:  {baseline_ndcg:.4f}")
    print(f"Optimized NDCG@10: {optimized_ndcg:.4f}")
    if baseline_ndcg > 0:
        improvement = (optimized_ndcg / baseline_ndcg - 1) * 100
        print(
            f"Improvement:       {optimized_ndcg - baseline_ndcg:+.4f} ({improvement:+.1f}%)"
        )
    else:
        print(f"Improvement:       {optimized_ndcg - baseline_ndcg:+.4f}")

    print("\nOptimized weights:")
    for key, value in optimized_weights.items():
        baseline_value = weights.BASE_WEIGHTS.get(key, 0)
        change = value - baseline_value
        change_pct = (change / baseline_value * 100) if baseline_value > 0 else 0
        print(
            f"  {key:15s}: {baseline_value:3d} → {value:3d} ({change:+.1f}, {change_pct:+.1f}%)"
        )
