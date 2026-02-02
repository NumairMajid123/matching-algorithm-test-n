"""
Generate ground truth matches for each profile based on independent criteria.
"""

import json

DATA_DIR = "data"
PROFILES_FILE = f"{DATA_DIR}/ground_truth_profiles.json"
PROPERTIES_FILE = f"{DATA_DIR}/synthetic_properties.json"
OUTPUT_FILE = f"{DATA_DIR}/my_ground_truth.json"

MAX_PRICE_TOLERANCE = 1.10  # Allow up to 10% over budget
MAX_SIZE_DEVIATION = 0.50  # Filter out properties >50% off desired size
SIZE_BONUS_TIER1 = 0.10  # Within 10% = best size match
SIZE_BONUS_TIER2 = 0.20  # Within 20% = good size match
NUM_MATCHES_PER_PROFILE = 5


def load_profiles():
  with open(PROFILES_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
    return data.get("profiles", [])


def load_properties():
  with open(PROPERTIES_FILE, "r", encoding="utf-8") as f:
    return json.load(f)


def find_good_matches(profile, properties, num_matches=NUM_MATCHES_PER_PROFILE):
  """
  Criteria:
  1. Property type must match exactly
  2. City must match exactly
  3. Price must be within budget (or slightly over)
  4. Size should be close to desired size
  """
  profile_data = profile["profile"]
  matches = []

  desired_type = profile_data.get("property_type", "").lower()
  desired_city = profile_data.get("city", "").lower()
  desired_size = profile_data.get("square_meters", 0)
  max_price = profile_data.get("max_price", 0)

  for prop in properties:
      if prop.get("property_type", "").lower() != desired_type:
          continue

      if prop.get("city", "").lower() != desired_city:
          continue

      try:
          price_str = (
              str(prop.get("price_per_month", "0")).replace(" ", "").replace(",", "")
          )
          price = int(price_str)
          if max_price > 0 and price > max_price * MAX_PRICE_TOLERANCE:
              continue
      except (ValueError, TypeError):
          continue

      prop_size = prop.get("square_meters", 0)
      size_diff = abs(prop_size - desired_size) if desired_size > 0 else 0
      size_diff_ratio = size_diff / desired_size if desired_size > 0 else 0

      if desired_size > 0 and size_diff_ratio > MAX_SIZE_DEVIATION:
          continue

      score = 0

      if price <= max_price:
          score += 100 + (max_price - price) / max(max_price, 1) * 50
      else:
          score += 50 - (price - max_price) / max(max_price, 1) * 50

      if desired_size > 0:
          if size_diff_ratio <= SIZE_BONUS_TIER1:
              score += 100
          elif size_diff_ratio <= SIZE_BONUS_TIER2:
              score += 70
          else:
              score += 40 * (1 - size_diff_ratio)

      matches.append(
          {
              "property_id": prop["id"],
              "score": score,
              "price": price,
              "size": prop_size,
              "size_diff": size_diff,
          }
      )

  matches.sort(key=lambda x: (-x["score"], x["size_diff"], x["price"]))

  return [
    {"property_id": m["property_id"], "rank": rank}
    for rank, m in enumerate(matches[:num_matches], start=1)
  ]


def main():
    print("Generating ground truth matches...")

    profiles = load_profiles()
    properties = load_properties()
    props_by_id = {p["id"]: p for p in properties}

    ground_truth = {}

    for profile in profiles:
        profile_id = profile["profile_id"]
        print(f"\nProcessing {profile_id}...")
        print(f"  Looking for: {profile['profile']}")

        matches = find_good_matches(profile, properties)

        if matches:
            ground_truth[profile_id] = matches
            print(f"  Found {len(matches)} good matches:")
            for match in matches:
                prop = props_by_id.get(match["property_id"])
                if prop:
                    print(
                        f"    Rank {match['rank']}: Property {prop['id']} - "
                        f"{prop['property_type']}, {prop['city']}, "
                        f"{prop['square_meters']}m², {prop['price_per_month']}kr"
                    )
        else:
            print(f"  WARNING: No matches found for {profile_id}")

    output = {"ground_truth": ground_truth}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Ground truth saved to {OUTPUT_FILE}")
    print(f"  Total profiles with matches: {len(ground_truth)}")


if __name__ == "__main__":
    main()
