"""
test_pipeline.py -- Unit tests for numcompute.pipeline.

Provides:
    duck-typed test doubles
        -> AddOne, Doubler, MeanModel, OnlyFit, OnlyTransform: plain classes
           with no base class, used to check the duck-typing contract
    Pipeline tests
        -> construction checks (mismatched step interface, bad steps),
           fit_transform vs fit then transform, predict, named_steps and
           indexing, y forwarding, error messages
    FeatureUnion tests
        -> np.hstack of member outputs, fit_transform vs fit then transform,
           member checks, row-count and 2-D output errors
    integration tests
        -> nesting Pipeline and FeatureUnion with numcompute.preprocessing

Requires:
    pytest >= 7, NumPy >= 1.23, numcompute.pipeline (as pl) and
    numcompute.preprocessing (as pp). Run `python -m pytest` from the repo root.
"""

import numpy as np
import pytest

from numcompute import pipeline as pl
from numcompute import preprocessing as pp


# ----------------------------------------------------------------------------
# Duck-typed test doubles (no base class on purpose)
# ----------------------------------------------------------------------------
class AddOne:
    """Transformer whose fit takes only X and that has no fit_transform."""

    def fit(self, X):
        self.fitted_ = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) + 1


class Doubler:
    """Transformer that records how it was fitted."""

    def __init__(self):
        self.fit_calls = 0
        self.seen_y = None

    def fit(self, X, y=None):
        self.fit_calls += 1
        self.seen_y = y
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 2


class MeanModel:
    """Estimator (fit/predict, no transform) that predicts the mean of y."""

    def fit(self, X, y=None):
        self.mean_ = 0.0 if y is None else float(np.mean(y))
        self.fit_X_ = np.asarray(X)
        return self

    def predict(self, X):
        self.predict_X_ = np.asarray(X)
        return np.full(len(X), self.mean_)


class OnlyFit:
    def fit(self, X, y=None):
        return self


class OnlyTransform:
    def transform(self, X):
        return X


class NotCallable:
    fit = "not a method"
    transform = 5


RNG = np.random.default_rng(42)
X_NUM = RNG.normal(loc=5, scale=3, size=(30, 4))
X_NAN = X_NUM.copy()
X_NAN[RNG.random(X_NAN.shape) < 0.15] = np.nan
Y = RNG.normal(size=30)


# ----------------------------------------------------------------------------
# Pipeline: construction and the step contract
# ----------------------------------------------------------------------------
def test_non_final_step_without_transform_raises_clear_error():
    with pytest.raises(TypeError) as exc:
        pl.Pipeline([("model", MeanModel()), ("scale", pp.StandardScaler())])
    msg = str(exc.value)
    assert "step 'model' (MeanModel) must implement fit and transform" in msg
    assert "missing: transform" in msg
    assert "Every step except the last must be a transformer" in msg


def test_non_final_step_without_fit_raises():
    with pytest.raises(TypeError, match=r"step 'raw' \(OnlyTransform\).*missing: fit"):
        pl.Pipeline([("raw", OnlyTransform()), ("scale", pp.StandardScaler())])


def test_non_callable_attributes_do_not_satisfy_the_contract():
    with pytest.raises(TypeError, match="missing: fit, transform"):
        pl.Pipeline([("bad", NotCallable()), ("scale", pp.StandardScaler())])


def test_last_step_needs_transform_or_predict():
    with pytest.raises(TypeError, match=r"step 'end' \(OnlyFit\) is the last step.*neither"):
        pl.Pipeline([("scale", pp.StandardScaler()), ("end", OnlyFit())])


def test_last_step_needs_fit():
    with pytest.raises(TypeError, match=r"step 'end' \(OnlyTransform\).*missing: fit"):
        pl.Pipeline([("scale", pp.StandardScaler()), ("end", OnlyTransform())])


@pytest.mark.parametrize(
    "steps, exc, pattern",
    [
        ({"scale": pp.StandardScaler()}, TypeError, "list of \\(name, step\\) tuples"),
        ([], ValueError, "at least one"),
        ([pp.StandardScaler()], TypeError, "item 0 must be a \\(name, step\\) tuple"),
        ([("a", pp.StandardScaler(), "extra")], TypeError, "item 0 must be"),
        ([(1, pp.StandardScaler())], TypeError, "name must be a non-empty str"),
        ([("", pp.StandardScaler())], TypeError, "name must be a non-empty str"),
        ([("s", pp.StandardScaler()), ("s", pp.MinMaxScaler())], ValueError, "repeated: \\['s'\\]"),
    ],
)
def test_malformed_steps_raise(steps, exc, pattern):
    with pytest.raises(exc, match=pattern):
        pl.Pipeline(steps)


def test_single_step_pipeline_is_allowed():
    out = pl.Pipeline([("scale", pp.StandardScaler())]).fit_transform(X_NUM)
    np.testing.assert_allclose(out, pp.StandardScaler().fit_transform(X_NUM))


def test_steps_list_is_copied():
    steps = [("scale", pp.StandardScaler())]
    pipe = pl.Pipeline(steps)
    steps.append(("oops", OnlyFit()))
    assert len(pipe) == 1


# ----------------------------------------------------------------------------
# Pipeline: fit / transform / fit_transform / predict
# ----------------------------------------------------------------------------
def test_fit_returns_self():
    pipe = pl.Pipeline([("scale", pp.StandardScaler())])
    assert pipe.fit(X_NUM) is pipe


def test_fit_transform_matches_fit_then_transform():
    def make():
        return pl.Pipeline(
            [("impute", pp.SimpleImputer()), ("scale", pp.StandardScaler()), ("mm", pp.MinMaxScaler())]
        )

    one_pass = make().fit_transform(X_NAN)
    two_calls = make().fit(X_NAN).transform(X_NAN)
    np.testing.assert_allclose(one_pass, two_calls)
    assert not np.isnan(one_pass).any()


def test_fit_transform_matches_for_steps_without_fit_transform():
    one_pass = pl.Pipeline([("add", AddOne()), ("dbl", Doubler())]).fit_transform(X_NUM)
    two_calls = pl.Pipeline([("add", AddOne()), ("dbl", Doubler())]).fit(X_NUM).transform(X_NUM)
    np.testing.assert_allclose(one_pass, two_calls)
    np.testing.assert_allclose(one_pass, (X_NUM + 1) * 2)


def test_pipeline_matches_applying_steps_by_hand():
    imp, sc = pp.SimpleImputer(), pp.StandardScaler()
    by_hand = sc.fit_transform(imp.fit_transform(X_NAN))
    piped = pl.Pipeline([("impute", pp.SimpleImputer()), ("scale", pp.StandardScaler())]).fit_transform(X_NAN)
    np.testing.assert_allclose(piped, by_hand)


def test_transform_uses_statistics_from_fit():
    pipe = pl.Pipeline([("scale", pp.StandardScaler())]).fit(X_NUM)
    X_new = X_NUM[:5] + 100
    np.testing.assert_allclose(
        pipe.transform(X_new), (X_new - X_NUM.mean(axis=0)) / X_NUM.std(axis=0)
    )


def test_predict_transforms_through_all_but_last_then_predicts():
    pipe = pl.Pipeline([("scale", pp.StandardScaler()), ("model", MeanModel())])
    pipe.fit(X_NUM, Y)
    model = pipe.named_steps["model"]
    np.testing.assert_allclose(model.fit_X_, pp.StandardScaler().fit_transform(X_NUM))

    X_new = X_NUM[:3] + 1
    y_pred = pipe.predict(X_new)
    np.testing.assert_allclose(y_pred, np.full(3, Y.mean()))
    np.testing.assert_allclose(model.predict_X_, pipe.named_steps["scale"].transform(X_new))


def test_transform_on_model_ending_pipeline_raises():
    pipe = pl.Pipeline([("scale", pp.StandardScaler()), ("model", MeanModel())]).fit(X_NUM, Y)
    with pytest.raises(TypeError, match=r"Pipeline.transform needs the last step.*'model' \(MeanModel\)"):
        pipe.transform(X_NUM)
    with pytest.raises(TypeError, match="Pipeline.fit_transform"):
        pipe.fit_transform(X_NUM, Y)


def test_predict_on_transformer_ending_pipeline_raises():
    pipe = pl.Pipeline([("scale", pp.StandardScaler())]).fit(X_NUM)
    with pytest.raises(TypeError, match=r"Pipeline.predict needs the last step.*'scale'"):
        pipe.predict(X_NUM)


def test_transform_before_fit_raises_not_fitted():
    pipe = pl.Pipeline([("impute", pp.SimpleImputer()), ("scale", pp.StandardScaler())])
    with pytest.raises(pp.NotFittedError):
        pipe.transform(X_NUM)


def test_y_is_forwarded_only_when_given():
    dbl = Doubler()
    pl.Pipeline([("dbl", dbl), ("model", MeanModel())]).fit(X_NUM, Y)
    assert dbl.seen_y is Y and dbl.fit_calls == 1
    # AddOne.fit takes only X: works because y=None is not passed on.
    pl.Pipeline([("add", AddOne()), ("model", MeanModel())]).fit(X_NUM)


def test_input_is_not_modified():
    X = X_NAN.copy()
    pl.Pipeline([("impute", pp.SimpleImputer()), ("scale", pp.StandardScaler())]).fit_transform(X)
    np.testing.assert_array_equal(X, X_NAN)


def test_spec_example_scale_then_encode():
    pipe = pl.Pipeline([("scale", pp.StandardScaler()), ("encode", pp.OneHotEncoder())])
    X = np.array([[1.0], [2.0], [1.0]])
    out = pipe.fit_transform(X)
    np.testing.assert_array_equal(out, [[1, 0], [0, 1], [1, 0]])


# ----------------------------------------------------------------------------
# Pipeline: named_steps and indexing
# ----------------------------------------------------------------------------
def test_named_steps_dict_access():
    scaler, mm = pp.StandardScaler(), pp.MinMaxScaler()
    pipe = pl.Pipeline([("scale", scaler), ("mm", mm)])
    assert pipe.named_steps == {"scale": scaler, "mm": mm}
    assert pipe.named_steps["scale"] is scaler
    assert list(pipe.named_steps) == ["scale", "mm"]  # order kept
    pipe.fit(X_NUM)
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, X_NUM.mean(axis=0))


def test_indexing_by_name_position_and_slice():
    scaler, mm = pp.StandardScaler(), pp.MinMaxScaler()
    pipe = pl.Pipeline([("scale", scaler), ("mm", mm)])
    assert pipe["scale"] is scaler and pipe[0] is scaler and pipe[-1] is mm
    head = pipe[:1]
    assert isinstance(head, pl.Pipeline) and len(head) == 1 and head[0] is scaler


def test_indexing_errors():
    pipe = pl.Pipeline([("scale", pp.StandardScaler())])
    with pytest.raises(KeyError, match=r"no step named 'nope'; steps are \['scale'\]"):
        pipe["nope"]
    with pytest.raises(IndexError):
        pipe[5]
    with pytest.raises(TypeError, match="indices must be str, int or slice, got float"):
        pipe[1.5]
    assert pipe[np.int64(0)] is pipe["scale"]  # NumPy integers work too


def test_len_and_repr():
    pipe = pl.Pipeline([("scale", pp.StandardScaler()), ("model", MeanModel())])
    assert len(pipe) == 2
    assert repr(pipe).startswith(
        "Pipeline([('scale', StandardScaler(with_mean=True, with_std=True)), ('model',"
    )


# ----------------------------------------------------------------------------
# FeatureUnion
# ----------------------------------------------------------------------------
def test_feature_union_hstacks_member_outputs():
    union = pl.FeatureUnion([("z", pp.StandardScaler()), ("mm", pp.MinMaxScaler())])
    out = union.fit_transform(X_NUM)
    expected = np.hstack(
        [pp.StandardScaler().fit_transform(X_NUM), pp.MinMaxScaler().fit_transform(X_NUM)]
    )
    assert out.shape == (30, 8)
    np.testing.assert_allclose(out, expected)


def test_feature_union_fit_transform_matches_fit_then_transform():
    def make():
        return pl.FeatureUnion([("z", pp.StandardScaler()), ("add", AddOne()), ("dbl", Doubler())])

    np.testing.assert_allclose(make().fit_transform(X_NUM), make().fit(X_NUM).transform(X_NUM))


def test_feature_union_fit_returns_self_and_fits_each_member_once():
    dbl = Doubler()
    union = pl.FeatureUnion([("dbl", dbl), ("z", pp.StandardScaler())])
    assert union.fit(X_NUM, Y) is union
    assert dbl.fit_calls == 1 and dbl.seen_y is Y


def test_feature_union_member_without_transform_raises_clear_error():
    with pytest.raises(TypeError) as exc:
        pl.FeatureUnion([("z", pp.StandardScaler()), ("model", MeanModel())])
    msg = str(exc.value)
    assert "FeatureUnion step 'model' (MeanModel) must implement fit and transform" in msg
    assert "missing: transform" in msg


@pytest.mark.parametrize(
    "transformers, exc",
    [([], ValueError), ([("a", AddOne()), ("a", Doubler())], ValueError), ([("only", OnlyFit())], TypeError)],
)
def test_feature_union_bad_construction(transformers, exc):
    with pytest.raises(exc):
        pl.FeatureUnion(transformers)


def test_feature_union_named_transformers_and_repr():
    z = pp.StandardScaler()
    union = pl.FeatureUnion([("z", z), ("add", AddOne())])
    assert union.named_transformers["z"] is z
    assert list(union.named_transformers) == ["z", "add"]
    assert repr(union).startswith("FeatureUnion([('z', StandardScaler(")


class DropsARow:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:-1]


class Flattens:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X).ravel()


@pytest.mark.parametrize("method", ["fit_transform", "transform"])
def test_feature_union_row_mismatch_raises(method):
    union = pl.FeatureUnion([("z", pp.StandardScaler()), ("drop", DropsARow())]).fit(X_NUM)
    with pytest.raises(ValueError, match="member 'drop' returned 29 rows, expected 30"):
        getattr(union, method)(X_NUM)


def test_feature_union_non_2d_output_raises():
    union = pl.FeatureUnion([("flat", Flattens())])
    with pytest.raises(ValueError, match=r"member 'flat' returned a 1-D output.*2-D array"):
        union.fit_transform(X_NUM)


def test_feature_union_transform_before_fit_raises_not_fitted():
    with pytest.raises(pp.NotFittedError):
        pl.FeatureUnion([("z", pp.StandardScaler())]).transform(X_NUM)


# ----------------------------------------------------------------------------
# Integration: nesting and the package-level API
# ----------------------------------------------------------------------------
def test_pipeline_with_feature_union_step_then_model():
    pipe = pl.Pipeline(
        [
            ("impute", pp.SimpleImputer(strategy="median")),
            ("features", pl.FeatureUnion([("z", pp.StandardScaler()), ("mm", pp.MinMaxScaler())])),
            ("model", MeanModel()),
        ]
    )
    pipe.fit(X_NAN, Y)
    assert pipe.named_steps["model"].fit_X_.shape == (30, 8)
    np.testing.assert_allclose(pipe.predict(X_NAN[:4]), np.full(4, Y.mean()))
    union = pipe.named_steps["features"]
    np.testing.assert_allclose(
        union.named_transformers["mm"].min_,
        np.nanmin(pp.SimpleImputer(strategy="median").fit_transform(X_NAN), axis=0),
    )


def test_feature_union_of_pipelines_mixes_numeric_and_categorical_branches():
    X_cat = np.array([["a"], ["b"], ["a"], ["c"]])
    X_num = np.array([[1.0], [np.nan], [3.0], [5.0]])
    numeric = pl.Pipeline([("impute", pp.SimpleImputer()), ("scale", pp.StandardScaler())])
    categorical = pl.Pipeline([("onehot", pp.OneHotEncoder())])
    features = np.hstack([numeric.fit_transform(X_num), categorical.fit_transform(X_cat)])
    assert features.shape == (4, 4) and not np.isnan(features).any()

    # The same idea as a single object: a union whose members are pipelines.
    union = pl.FeatureUnion(
        [("num", pl.Pipeline([("impute", pp.SimpleImputer())])), ("z", pp.StandardScaler())]
    )
    out = union.fit_transform(np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(out, [[1.0, -1.0], [3.0, 1.0]])


# ----------------------------------------------------------------------------
# Non-contiguous input (X[::2], X.T) through Pipeline and FeatureUnion
# ----------------------------------------------------------------------------
class _MeanCenter:
    """Stateful transformer used only by the non-contiguous tests."""

    def fit(self, X, y=None):
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_


_RNG_NC = np.random.default_rng(7)
_NON_CONTIGUOUS = [
    _RNG_NC.normal(size=(40, 6))[::2],  # X[::2]: every other row, shape (20, 6)
    _RNG_NC.normal(size=(6, 40)).T,  # X.T: transposed view, shape (40, 6)
]
_IDS = ["X[::2]", "X.T"]


@pytest.mark.parametrize("view", _NON_CONTIGUOUS, ids=_IDS)
def test_pipeline_non_contiguous_input_matches_contiguous_copy(view):
    assert not view.flags["C_CONTIGUOUS"]
    copy = np.ascontiguousarray(view)

    def make():
        return pl.Pipeline([("center", _MeanCenter()), ("add", AddOne()), ("dbl", Doubler())])

    np.testing.assert_allclose(make().fit_transform(view), make().fit_transform(copy))
    np.testing.assert_allclose(make().fit(view).transform(view), make().fit(copy).transform(copy))
    np.testing.assert_array_equal(view, copy)  # input left unchanged


@pytest.mark.parametrize("view", _NON_CONTIGUOUS, ids=_IDS)
def test_feature_union_non_contiguous_input_matches_contiguous_copy(view):
    assert not view.flags["C_CONTIGUOUS"]
    copy = np.ascontiguousarray(view)

    def make():
        return pl.FeatureUnion([("center", _MeanCenter()), ("add", AddOne()), ("dbl", Doubler())])

    out = make().fit_transform(view)
    assert out.shape == (view.shape[0], 3 * view.shape[1])
    np.testing.assert_allclose(out, make().fit_transform(copy))
    np.testing.assert_allclose(make().fit(view).transform(view), make().fit(copy).transform(copy))
    np.testing.assert_array_equal(view, copy)  # input left unchanged
