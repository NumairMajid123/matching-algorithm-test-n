"""
Generator for synthetic property data.

Creates 500 properties with varied attributes for testing.
"""

import json
import random

# Seed for reproducibility
random.seed(42)

PROPERTY_TYPES = ['kontor', 'butik', 'lager']
CITIES = ['Stockholm', 'Göteborg', 'Malmö']


def generate_synthetic_properties(count=500):
    """
    Generate synthetic properties.
    
    Args:
        count: Number of properties to generate
    
    Returns:
        list: List of property dictionaries
    """

    properties = []
    
    for i in range(1, count + 1):
        property_type = random.choice(PROPERTY_TYPES)
        city = random.choice(CITIES)
        
        # Generate realistic values based on property type
        if property_type == 'kontor':
            square_meters = random.randint(50, 500)
            base_price = random.randint(20000, 150000)
        elif property_type == 'butik':
            square_meters = random.randint(30, 300)
            base_price = random.randint(15000, 120000)
        else:  # lager
            square_meters = random.randint(100, 2000)
            base_price = random.randint(10000, 100000)
        
        # Add some variation to the price
        price = base_price + random.randint(-5000, 5000)
        price = max(10000, price)  # Minimum 10000
        
        prop = {
            'id': i,
            'property_type': property_type,
            'city': city,
            'square_meters': square_meters,
            'price_per_month': str(price),  # As string for realism
        }
        
        properties.append(prop)
    
    return properties


if __name__ == '__main__':
    properties = generate_synthetic_properties(500)
    
    with open('synthetic_properties.json', 'w', encoding='utf-8') as f:
        json.dump(properties, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(properties)} synthetic properties")
    print(f"Saved to synthetic_properties.json")
    
    # Show statistics
    by_type = {}
    by_city = {}
    for p in properties:
        by_type[p['property_type']] = by_type.get(p['property_type'], 0) + 1
        by_city[p['city']] = by_city.get(p['city'], 0) + 1
    
    print(f"\nDistribution by property type:")
    for pt, count in by_type.items():
        print(f"  {pt}: {count}")
    
    print(f"\nDistribution by city:")
    for city, count in by_city.items():
        print(f"  {city}: {count}")
