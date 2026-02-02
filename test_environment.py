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
    OPTIMIZE THIS FUNCTION!
    
    Your task is to implement ML optimization to find
    the best weights that maximize NDCG@10.
    
    Requirements:
    - Use systematic optimization (not manual adjustment)
    - Test different methods if possible
    - Document your method
    
    Tips:
    - Use scipy.optimize.differential_evolution or minimize
    - Test different bounds for weights (e.g. 0-200)
    - Starting values can be current BASE_WEIGHTS
    
    Returns:
        dict: Optimized weights, e.g.:
            {
                'property_type': 60,
                'location': 35,
                'size': 25,
                'price': 12
            }
    """
    # TODO: Implement optimization here
    # 
    # Example of how you can start:
    # from scipy.optimize import differential_evolution
    # 
    # def objective(weights_vector):
    #     weights_dict = {
    #         'property_type': weights_vector[0],
    #         'location': weights_vector[1],
    #         'size': weights_vector[2],
    #         'price': weights_vector[3]
    #     }
    #     return -evaluate_weights(weights_dict)  # Negative because we minimize
    
    # For now: return baseline weights
    return BASE_WEIGHTS.copy()


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
