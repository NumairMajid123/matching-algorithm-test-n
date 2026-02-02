"""
Simplified scoring algorithm for matching properties against profiles.

This algorithm is a simplified version of MyPortal's matching algorithm.
It uses 4 weights to calculate a total score for each property.
"""

from matching.weights import BASE_WEIGHTS


def score_property(property_obj, profile):
    """
    Score a property against a profile.
    
    Args:
        property_obj: Dict with property data, must contain:
            - 'id': int
            - 'property_type': str (e.g. 'kontor', 'butik', 'lager')
            - 'city': str (e.g. 'Stockholm', 'Göteborg', 'Malmö')
            - 'square_meters': int
            - 'price_per_month': str or int (e.g. '50000' or 50000)
        
        profile: Dict with profile data, can contain:
            - 'property_type': str (desired property type)
            - 'city': str (desired city)
            - 'square_meters': int (desired size)
            - 'max_price': int (max budget)
    
    Returns:
        float: Total score (higher = better match)
    """
    score = 0.0
    
    # 1. Property type match (binary: matches or doesn't match)
    if profile.get('property_type') and property_obj.get('property_type'):
        if profile['property_type'].lower() == property_obj['property_type'].lower():
            score += BASE_WEIGHTS['property_type']
    
    # 2. Location match (binary: matches city or doesn't match)
    if profile.get('city') and property_obj.get('city'):
        if profile['city'].lower() in property_obj['city'].lower():
            score += BASE_WEIGHTS['location']
    
    # 3. Size match (continuous: closer to desired size = higher score)
    if profile.get('square_meters') and property_obj.get('square_meters'):
        target = profile['square_meters']
        actual = property_obj['square_meters']
        
        if target > 0:
            diff_ratio = abs(actual - target) / target
            
            if diff_ratio <= 0.15:  # Within 15% = full score
                score += BASE_WEIGHTS['size'] * (1 - diff_ratio / 0.15)
            elif diff_ratio <= 0.30:  # Within 30% = half score
                score += BASE_WEIGHTS['size'] * 0.5 * (1 - (diff_ratio - 0.15) / 0.15)
            # Outside 30% = 0 points
    
    # 4. Price match (continuous: under budget = positive, over = negative)
    if profile.get('max_price') and property_obj.get('price_per_month'):
        try:
            # Convert price_per_month to int (handle str with spaces/commas)
            price_str = str(property_obj['price_per_month']).replace(' ', '').replace(',', '')
            price = int(price_str)
            max_price = profile['max_price']
            
            if price <= max_price:
                # Within budget: full score
                score += BASE_WEIGHTS['price']
            else:
                # Over budget: penalty
                over_ratio = (price - max_price) / max_price
                if over_ratio <= 0.05:  # Slightly over (5%)
                    score += BASE_WEIGHTS['price'] * 0.5
                else:
                    # Far over: negative penalty (can make total score negative)
                    score -= BASE_WEIGHTS['price'] * (1 + over_ratio)
        except (ValueError, TypeError):
            # If price cannot be converted, skip
            pass
    
    return score
