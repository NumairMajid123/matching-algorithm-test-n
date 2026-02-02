"""
Script to generate ground truth matches for each profile.

This script analyzes profiles and properties to identify good matches
based on property type, city, size, and price criteria.
"""

import json
from matching.scoring import score_property
from matching.weights import BASE_WEIGHTS


def load_profiles():
    """Load profiles from JSON file."""
    with open('data/ground_truth_profiles.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('profiles', [])


def load_properties():
    """Load properties from JSON file."""
    with open('data/synthetic_properties.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def find_good_matches(profile, properties, num_matches=5):
    """
    Find good matches for a profile.
    
    Criteria (in order of importance):
    1. Property type must match
    2. City must match
    3. Price must be within budget (or slightly over)
    4. Size should be close to desired size
    
    Returns list of property IDs ranked by match quality.
    """
    profile_data = profile['profile']
    matches = []
    
    for prop in properties:
        # Must match property type
        if prop.get('property_type', '').lower() != profile_data.get('property_type', '').lower():
            continue
        
        # Must match city
        prop_city = prop.get('city', '').lower()
        profile_city = profile_data.get('city', '').lower()
        if profile_city not in prop_city and prop_city not in profile_city:
            continue
        
        # Check price
        try:
            price_str = str(prop.get('price_per_month', '0')).replace(' ', '').replace(',', '')
            price = int(price_str)
            max_price = profile_data.get('max_price', 0)
            
            # Allow up to 10% over budget (still consider it, but rank lower)
            if price > max_price * 1.10:
                continue
        except (ValueError, TypeError):
            continue
        
        # Check size
        prop_size = prop.get('square_meters', 0)
        desired_size = profile_data.get('square_meters', 0)
        
        if desired_size > 0:
            size_diff_ratio = abs(prop_size - desired_size) / desired_size
            # Prefer properties within 30% of desired size
            if size_diff_ratio > 0.50:  # Too far off
                continue
        
        # Calculate a match score for ranking
        # Use the scoring function to get a baseline score
        score = score_property(prop, profile_data)
        
        # Add bonus for being within budget
        if price <= max_price:
            score += 10
        
        # Add bonus for size match (closer = better)
        if desired_size > 0:
            size_diff = abs(prop_size - desired_size)
            if size_diff <= desired_size * 0.10:  # Within 10%
                score += 20
            elif size_diff <= desired_size * 0.20:  # Within 20%
                score += 10
        
        matches.append({
            'property_id': prop['id'],
            'score': score,
            'price': price,
            'size': prop_size,
            'size_diff': abs(prop_size - desired_size) if desired_size > 0 else float('inf')
        })
    
    # Sort by score (highest first), then by size difference, then by price
    matches.sort(key=lambda x: (-x['score'], x['size_diff'], x['price']))
    
    # Return top N matches with ranks
    ranked_matches = []
    for rank, match in enumerate(matches[:num_matches], start=1):
        ranked_matches.append({
            'property_id': match['property_id'],
            'rank': rank
        })
    
    return ranked_matches


def main():
    """Generate ground truth matches for all profiles."""
    print("Generating ground truth matches...")
    
    profiles = load_profiles()
    properties = load_properties()
    
    ground_truth = {}
    
    for profile in profiles:
        profile_id = profile['profile_id']
        print(f"\nProcessing {profile_id}...")
        print(f"  Looking for: {profile['profile']}")
        
        matches = find_good_matches(profile, properties, num_matches=5)
        
        if matches:
            ground_truth[profile_id] = matches
            print(f"  Found {len(matches)} good matches:")
            for match in matches:
                prop_id = match['property_id']
                prop = next((p for p in properties if p['id'] == prop_id), None)
                if prop:
                    print(f"    Rank {match['rank']}: Property {prop_id} - {prop['property_type']}, {prop['city']}, {prop['square_meters']}m², {prop['price_per_month']}kr")
        else:
            print(f"  WARNING: No matches found for {profile_id}")
    
    # Save to file
    output = {
        'ground_truth': ground_truth
    }
    
    with open('data/my_ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Ground truth saved to data/my_ground_truth.json")
    print(f"  Total profiles with matches: {len(ground_truth)}")


if __name__ == '__main__':
    main()


