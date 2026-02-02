# Task: Optimize Matching Algorithm

## Background
You should optimize the weights in a matching algorithm that matches properties against search profiles. The algorithm uses 4 weights to calculate a total score for each property.

## Data
- **500 synthetic properties** with varied attributes (property type, city, size, price)
- **10 search profiles** that define what the user is looking for

## Task
1. **Define good matches**: For each profile, analyze the properties and define which ones are "good matches" and in what order (rank 1 = best, rank 2 = second best, etc.). This is an important part of the task - to intuitively understand what constitutes a good match.

2. **Optimize weights**: Implement the `optimize_weights()` function in `test_environment.py` so that **NDCG@10 is maximized** against your defined good matches.

## Requirements

### 1. Technical
- Use **ML optimization** (not manual adjustment or trial-and-error)
- Test different optimization methods if possible
- Code should be clean and well-commented

### 2. Documentation
- Document your method and why you chose it
- Explain the result and which weights are most important
- Analyze why the algorithm works well/poorly

### 3. Analysis
- Identify which profiles are hardest to match
- Suggest possible algorithm improvements (beyond weights)

## Time Limit
4 hours

## Evaluation

### Pass
- NDCG@10 > 0.4
- Systematic optimization implemented
- Basic documentation

### Excellent
- NDCG@10 > 0.5
- Multiple optimization methods tested and compared
- In-depth analysis of results
- Concrete suggestions for algorithm improvements

## Tips

### Getting Started

1. **Analyze profiles and properties**
   - Read through `data/ground_truth_profiles.json` to see which profiles exist
   - Read through `data/synthetic_properties.json` to see which properties exist
   - For each profile, identify which properties are good matches based on:
     - Property type matching
     - City matching
     - Size (close to desired size)
     - Price (within budget)

2. **Create your ground truth**
   - Create the file `data/my_ground_truth.json` with the format:
   ```json
   {
     "ground_truth": {
       "profile_1": [
         {"property_id": 45, "rank": 1},
         {"property_id": 123, "rank": 2},
         {"property_id": 67, "rank": 3}
       ],
       "profile_2": [...],
       ...
     }
   }
   ```
   - Rank 1 = best match, rank 2 = second best, etc.
   - Choose 3-5 good matches per profile

3. **Test baseline**
   - Run `python3 test_environment.py` to see baseline NDCG
   - Test `evaluate_weights()` with different weights manually first

4. **Implement optimization**
   - Implement `optimize_weights()` step by step

### Optimization
- Use `scipy.optimize.differential_evolution` or `minimize`
- Test different bounds (e.g. 0-200 for each weight)
- Starting values can be current BASE_WEIGHTS

### Debugging
- If NDCG doesn't improve, check which properties are ranked high
- Analyze why ground truth properties don't end up at the top
- Test extreme values to see if weights have any effect

## Deliverables
- `data/my_ground_truth.json` with your defined good matches
- Implemented `optimize_weights()` function
- Short report (1-2 pages) explaining:
  - How you defined "good matches" (what criteria did you use?)
  - Which optimization method you used
  - Why you chose it
  - The result (NDCG before/after)
  - Analysis of which weights are most important
  - Suggestions for improvements

## Questions?
Ask questions if anything is unclear.
