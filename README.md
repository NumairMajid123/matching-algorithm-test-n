# Matching Algorithm Test Environment

Test environment for evaluating candidates' ability to optimize matching algorithms with machine learning.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
cd data
python3 generate_synthetic_data.py
cd ..
```

This creates `data/synthetic_properties.json` with 500 properties.

### 3. Run baseline test
```bash
python3 test_environment.py
```

This shows baseline NDCG@10 with current weights.

## Structure

```
matching-algorithm-test/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── candidate_task.md            # Instructions for the candidate
├── matching/                    # Algorithm code
│   ├── weights.py              # BASE_WEIGHTS (4 weights)
│   ├── scoring.py              # Scoring algorithm
│   └── evaluation.py           # NDCG calculation
├── data/                        # Data
│   ├── generate_synthetic_data.py
│   ├── synthetic_properties.json
│   └── ground_truth_profiles.json
└── test_environment.py         # Main file with evaluate_weights()
```

## Task

See `candidate_task.md` for full instructions.

Briefly:
1. Define which properties are "good matches" for each profile (create `data/my_ground_truth.json`)
2. Implement `optimize_weights()` in `test_environment.py` to maximize NDCG@10 against your defined matches

## Evaluation

- NDCG@10 > 0.4: Pass
- NDCG@10 > 0.5: Excellent
- Documentation and analysis: Important

## Tips

- Read through the code in `matching/` to understand the algorithm
- Test `evaluate_weights()` with different weights first
- Use `scipy.optimize` for systematic optimization
