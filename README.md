# 2650_COMP_5004_2_Assignment-2.1-Programming-Task-1-group-Beta

### Package skeleton & __init__.py

```
2650_COMP_5004_2_Assignment-2.1-Programming-Task-1-group-Beta/
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
├── demo/quickstart.ipynb
├── README.md
└── pyproject.toml
```

## Dataset

https://www.kaggle.com/datasets/yasserh/housing-prices-dataset


## Work division


| Person | Theme | Files | Why these belong together |
|---|---|---|---|
| **Binitta** | Data & Pipelines | `io.py`, `preprocessing.py`, `pipeline.py` | These three describe one continuous journey: a raw CSV becomes a cleaned array, which becomes a chain of reusable transformation steps. Whoever owns this understands the shape of the data end-to-end. |
| **Melvin** | Algorithms | `sort_search.py`, `rank.py`, `optim.py` | All three are about the correctness of a classic algorithm rather than data cleaning: sorting and searching, ranking with tie-breaking rules, and calculus by finite differences. |
| **Cyril** | Stats, Metrics & Benchmarking | `stats.py`, `metrics.py`, `utils.py`, `benchmarking.py` | These modules all produce numbers that describe other numbers — descriptive statistics, prediction-quality scores, and speed comparisons. `utils.py`'s helpers (softmax, distances, safe division) are used most heavily here, so it's a natural fit even though it brings the module count for this theme to four. |
