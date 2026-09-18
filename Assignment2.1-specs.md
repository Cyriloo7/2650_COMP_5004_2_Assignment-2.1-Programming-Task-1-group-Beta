## Core Requirements

### `io.py`
- Load CSV files using `numpy.loadtxt` or `numpy.genfromtxt`
- Support custom delimiters (e.g., comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Return data as NumPy arrays

### `preprocessing.py`
- Implement:
  - `StandardScaler` (z-score standardization)
  - `MinMaxScaler` (scale to a given range)
  - `OneHotEncoder` (for categorical variables)
  - *(Optional)* Simple imputer to replace NaN with a constant
- API:
  - `fit(X) -> self`
  - `transform(X) -> X_out`
  - `fit_transform(X) -> X_out`
- Fully vectorized using NumPy; avoid Python loops

### `sort_search.py`
- **Sorting**:
  - Stable sort wrapper (`np.sort(kind='stable')`)
  - Multi-key sort (sort by multiple columns)
- **Top-k / Partial Sort**:
  - `topk(values, k, largest=True, return_indices=True)` using `np.argpartition`
  - Implement **quickselect** for educational purposes
- **Searching**:
  - `binary_search(sorted_array, x)` returning insertion index and existence boolean

### `rank.py`
- `rank(data, method='average'|'dense'|'ordinal')` — handle ties
- `percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint')`

### `stats.py`
- Basic descriptive statistics:
  - Mean, median, standard deviation, min, max
- Histogram
- Quantiles (with NaN handling)
- Axis-wise stats with clear dimension/shape behaviour

### `metrics.py`
- Classification:
  - `accuracy`
  - `precision`
  - `recall`
  - `f1`
  - `confusion_matrix`
- Regression:
  - `mse(y_true, y_pred)`
- **Bonus**: `roc_curve` and `auc` for binary classification

### `optim.py`
- Finite-difference gradient estimation:
  - `grad(f, x, h=1e-5, method='central'|'forward')`
  - `jacobian(F, x, ...)`

### `pipeline.py`
- Minimal `Transformer`/`Estimator` protocol:
  - Preprocessors: `fit`, `transform`
  - Models: `fit`, `predict` (no ML model implementation required)
- `Pipeline` chaining
- Example:
  ```python
  pipe = Pipeline([
    ('scale', StandardScaler()),
    ('encode', OneHotEncoder())
  ])
  X_tr = pipe.fit_transform(X)
