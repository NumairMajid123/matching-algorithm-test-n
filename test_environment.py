"""
Test environment for candidates.

This file contains:
- evaluate_weights(): Function that evaluates weights against ground truth
- optimize_weights(): Function that the candidate should implement

The candidate should implement optimize_weights() to maximize NDCG@10.
"""

import json
from matching.scoring import score_property
from matching.evaluation import calculate_ndcg_at_k
from matching.weights import BASE_WEIGHTS


def load_ground_truth_profiles():
    """
    Load profiles from JSON file.
    
    Returns:
        list: List of profile dictionaries with 'profile'
    """
    with open('data/ground_truth_profiles.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('profiles', [])


def load_my_ground_truth():
    """
    Load candidate's defined ground truth matches.
    
    This file should be created by the candidate and contain good_matches for each profile.
    See documentation for format.
    
    Returns:
        dict: Mapping from profile_id to list of good_matches
            e.g. {'profile_1': [{'property_id': 45, 'rank': 1}, ...], ...}
    """
    try:
        with open('data/my_ground_truth.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('ground_truth', {})
    except FileNotFoundError:
        return {}


def load_synthetic_properties():
    """
    Load synthetic properties from JSON file.
    
    Returns:
        list: List of property dictionaries
    """
    with open('data/synthetic_properties.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_weights(weights_dict, ground_truth_matches=None):
    """
    Evaluate weights against ground truth profiles.
    
    This function:
    1. Updates BASE_WEIGHTS with new values
    2. Loops through all profiles
    3. For each profile: score all properties and rank them
    4. Compares predicted ranking with ideal ranking (ground truth)
    5. Calculates NDCG@10 for each profile
    6. Returns average NDCG@10
    
    Args:
        weights_dict: Dict with weights to test, e.g.:
            {
                'property_type': 60,
                'location': 35,
                'size': 25,
                'price': 12
            }
            All 4 weights must be present.
        ground_truth_matches: Optional dict with good_matches per profile_id.
            If None, tries to load from data/my_ground_truth.json.
            Format: {'profile_1': [{'property_id': 45, 'rank': 1}, ...], ...}
    
    Returns:
        float: Average NDCG@10 over all profiles (0-1)
    
    Example:
        # Use candidate's defined ground truth
        baseline = evaluate_weights(BASE_WEIGHTS)
        print(f"Baseline NDCG: {baseline:.4f}")
        
        # Or pass your own matches
        my_matches = {'profile_1': [{'property_id': 45, 'rank': 1}, ...]}
        test_ndcg = evaluate_weights(BASE_WEIGHTS, ground_truth_matches=my_matches)
    """
    # Update BASE_WEIGHTS temporarily
    from matching import weights
    original_weights = weights.BASE_WEIGHTS.copy()
    weights.BASE_WEIGHTS.update(weights_dict)
    
    try:
        profiles = load_ground_truth_profiles()
        properties = load_synthetic_properties()
        
        # Load ground truth matches
        if ground_truth_matches is None:
            ground_truth_matches = load_my_ground_truth()
        
        if not ground_truth_matches:
            print("WARNING: No ground truth found!")
            print("Create data/my_ground_truth.json with your defined good matches.")
            print("See documentation for format.")
            return 0.0
        
        total_ndcg = 0.0
        num_profiles = 0
        
        for gt_profile in profiles:
            profile_id = gt_profile['profile_id']
            profile = gt_profile['profile']
            good_matches = ground_truth_matches.get(profile_id, [])
            
            if not good_matches:
                continue  # Skip profiles without ground truth
            
            # Score all properties
            scored_properties = []
            for prop in properties:
                score = score_property(prop, profile)
                scored_properties.append((prop['id'], score))
            
            # Sort by score (highest first)
            scored_properties.sort(key=lambda x: x[1], reverse=True)
            predicted_ranks = [pid for pid, _ in scored_properties]
            
            # Ideal ranks from ground truth (rank 1 = first in list)
            ideal_ranks = [m['property_id'] for m in sorted(good_matches, key=lambda x: x['rank'])]
            
            # Calculate NDCG@10
            ndcg = calculate_ndcg_at_k(predicted_ranks, ideal_ranks, k=10)
            total_ndcg += ndcg
            num_profiles += 1
        
        return total_ndcg / num_profiles if num_profiles > 0 else 0.0
    
    finally:
        # Restore original weights
        weights.BASE_WEIGHTS = original_weights


def optimize_weights():
    """
    Optimize weights using scipy.optimize to maximize NDCG@10.
    
    This function uses multiple optimization methods and selects the best result:
    1. Differential Evolution - Global optimization, good for avoiding local minima
    2. L-BFGS-B - Local optimization with bounds, faster convergence
    
    Strategy:
    - Weights are bounded between 0-200 to prevent extreme values
    - Objective function is negative NDCG (since optimizers minimize)
    - Starting point is current BASE_WEIGHTS
    - Both methods are tried and the best result is returned
    
    Returns:
        dict: Optimized weights, e.g.:
            {
                'property_type': 60,
                'location': 35,
                'size': 25,
                'price': 12
            }
    """
    from scipy.optimize import differential_evolution, minimize
    import numpy as np
    
    # Define bounds for each weight (0-200)
    bounds = [(0, 200), (0, 200), (0, 200), (0, 200)]
    
    # Starting point from current BASE_WEIGHTS
    x0 = [
        BASE_WEIGHTS['property_type'],
        BASE_WEIGHTS['location'],
        BASE_WEIGHTS['size'],
        BASE_WEIGHTS['price']
    ]
    
    def objective(weights_vector):
        """
        Objective function: negative NDCG@10 (since optimizers minimize).
        
        Args:
            weights_vector: [property_type, location, size, price]
        
        Returns:
            float: Negative NDCG@10 (to maximize NDCG, we minimize -NDCG)
        """
        weights_dict = {
            'property_type': max(0, weights_vector[0]),  # Ensure non-negative
            'location': max(0, weights_vector[1]),
            'size': max(0, weights_vector[2]),
            'price': max(0, weights_vector[3])
        }
        ndcg = evaluate_weights(weights_dict)
        return -ndcg  # Negative because we minimize
    
    print("   Trying Differential Evolution (global optimization)...")
    # Method 1: Differential Evolution (global optimization)
    result_de = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        polish=True  # Polish result with local optimization
    )
    ndcg_de = -result_de.fun
    weights_de = {
        'property_type': max(0, int(round(result_de.x[0]))),
        'location': max(0, int(round(result_de.x[1]))),
        'size': max(0, int(round(result_de.x[2]))),
        'price': max(0, int(round(result_de.x[3])))
    }
    print(f"   DE result: NDCG={ndcg_de:.4f}, weights={weights_de}")
    
    print("   Trying L-BFGS-B (local optimization)...")
    # Method 2: L-BFGS-B (local optimization with bounds)
    result_lbfgs = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6}
    )
    ndcg_lbfgs = -result_lbfgs.fun
    weights_lbfgs = {
        'property_type': max(0, int(round(result_lbfgs.x[0]))),
        'location': max(0, int(round(result_lbfgs.x[1]))),
        'size': max(0, int(round(result_lbfgs.x[2]))),
        'price': max(0, int(round(result_lbfgs.x[3])))
    }
    print(f"   L-BFGS-B result: NDCG={ndcg_lbfgs:.4f}, weights={weights_lbfgs}")
    
    # Select the best result
    if ndcg_de > ndcg_lbfgs:
        print(f"   Selected Differential Evolution result (better NDCG)")
        return weights_de
    else:
        print(f"   Selected L-BFGS-B result (better NDCG)")
        return weights_lbfgs


if __name__ == '__main__':
    print("=" * 60)
    print("MATCHING ALGORITHM OPTIMIZATION TEST")
    print("=" * 60)
    
    # Check if ground truth exists
    ground_truth = load_my_ground_truth()
    if not ground_truth:
        print("\nNO GROUND TRUTH FOUND!")
        print("\nYou must first create data/my_ground_truth.json with your defined good matches.")
        print("See data/my_ground_truth.json.example for format.")
        print("\nSteps:")
        print("1. Analyze profiles in data/ground_truth_profiles.json")
        print("2. Analyze properties in data/synthetic_properties.json")
        print("3. For each profile, identify 3-5 good matches and rank them")
        print("4. Create data/my_ground_truth.json according to the format in the example file")
        print("\nSee candidate_task.md for more information.")
        exit(1)
    
    # Test baseline
    print("\n1. Testing baseline weights...")
    baseline_ndcg = evaluate_weights(BASE_WEIGHTS)
    print(f"   Baseline NDCG@10: {baseline_ndcg:.4f}")
    print(f"   Baseline weights: {BASE_WEIGHTS}")
    
    # Optimize
    print("\n2. Optimization...")
    optimized_weights = optimize_weights()
    optimized_ndcg = evaluate_weights(optimized_weights)
    
    # Show results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Baseline NDCG@10:  {baseline_ndcg:.4f}")
    print(f"Optimized NDCG@10: {optimized_ndcg:.4f}")
    if baseline_ndcg > 0:
        print(f"Improvement:       {optimized_ndcg - baseline_ndcg:+.4f} ({(optimized_ndcg/baseline_ndcg - 1)*100:+.1f}%)")
    else:
        print(f"Improvement:       {optimized_ndcg - baseline_ndcg:+.4f}")
    print(f"\nOptimized weights:")
    for key, value in optimized_weights.items():
        baseline_value = BASE_WEIGHTS.get(key, 0)
        change = value - baseline_value
        change_pct = (change / baseline_value * 100) if baseline_value > 0 else 0
        print(f"  {key:15s}: {baseline_value:3d} → {value:3d} ({change:+.1f}, {change_pct:+.1f}%)")
