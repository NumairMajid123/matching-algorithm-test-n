"""
Base weights for the matching algorithm.

These are the 4 weights that should be optimized:
- property_type: Weight for property type matching
- location: Weight for location (city) matching
- size: Weight for size matching
- price: Weight for price matching
"""

BASE_WEIGHTS = {
    'property_type': 50,
    'location': 30,
    'size': 20,
    'price': 15,
}
