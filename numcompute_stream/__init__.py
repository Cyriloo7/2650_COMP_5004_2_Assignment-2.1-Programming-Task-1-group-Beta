"""numcompute_stream - Assignment 2.2: a streaming framework built around a buffered decision tree.

It extends the Assignment 2.1 package ``numcompute``, which is reused and not
changed. Every component learns chunk by chunk: the model, the transformers
and the pipeline use ``partial_fit``, metrics use ``update``, and statistics
use ``update_stats``. Plots live in ``numcompute_stream.visualise``, which is
imported separately so the rest of the package does not need matplotlib.
"""

from .io import iter_chunks, load_csv
from .metrics import AUC, F1, Accuracy, ConfusionMatrix, Precision, Recall, RollingMetric
from .pipeline import Pipeline
from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from .stats import StreamingStats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier

__all__ = [
    "load_csv", "iter_chunks", "StandardScaler", "MinMaxScaler", "Imputer", "OneHotEncoder", "StreamingStats",
    "Accuracy", "Precision", "Recall", "F1", "ConfusionMatrix", "AUC", "RollingMetric",
    "DecisionTreeClassifier", "Pipeline", "StreamTrainer",
]
