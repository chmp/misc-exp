import numpy as np
import pandas as pd
import sklearn.base

from chmp.ds import find_categorical_columns


def action_p_to_propensity(action, action_p):
    return action * action_p + (1 - action) * (1 - action_p)


class RegressingBinaryClassifier(
    sklearn.base.BaseEstimator, sklearn.base.RegressorMixin
):
    def __init__(self, est):
        self.est = est

    def fit(self, x, y):
        self.est.fit(x, y)
        return self

    def predict(self, x):
        _, score = self.est.predict_proba(x).T
        return score


class CategoricalMeanTargetEncoder(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    def __init__(self, columns=None, pseudo_count=1):
        self.columns = columns
        self.pseudo_count = pseudo_count
        self.columns_ = columns
        self.rates_ = {}
        self.population_mean_ = None

    def fit(self, x, y):
        if self.columns_ is None:
            self.columns_ = find_categorical_columns(x)

        self.population_mean_ = np.nanmean(y)

        for col in self.columns_:
            agg = pd.Series(y, index=x.index).groupby(x[col]).agg(["sum", "count"])
            mean = (agg["sum"] + self.population_mean_ * self.pseudo_count) / (
                agg["count"] + self.pseudo_count
            )
            self.rates_[col] = mean

        return self

    def transform(self, x, y=None):
        for col in self.columns_:
            try:
                mean = self.rates_[col]
                s = x[col].replace(mean)
                s[~x[col].isin(mean.index)] = np.nan
                s = s.astype(float)

                x = x.assign(**{col: s})

            except Exception as e:
                raise RuntimeError(f"cannot transform {col}") from e

        return x


class CategoricalIndexEncoder(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    def __init__(self, columns=None, pseudo_count=1):
        self.columns = columns
        self.pseudo_count = pseudo_count
        self.columns_ = columns
        self.replacements_ = {}
        self.population_mean_ = None

    def fit(self, x, y):
        if self.columns_ is None:
            self.columns_ = find_categorical_columns(x)

        self.population_mean_ = np.nanmean(y)

        for col in self.columns_:
            agg = pd.Series(y, index=x.index).groupby(x[col]).agg(["sum", "count"])
            mean = (agg["sum"] + self.population_mean_ * self.pseudo_count) / (
                agg["count"] + self.pseudo_count
            )
            self.replacements_[col] = pd.Series(
                np.argsort(mean.values), index=mean.index
            )

        return self

    def transform(self, x, y=None):
        for col in self.columns_:
            try:
                repl = self.replacements_[col]
                s = x[col].replace(repl)
                s[~x[col].isin(repl.index)] = np.nan
                s = s.astype(int)

                x = x.assign(**{col: s})

            except Exception as e:
                raise RuntimeError(f"cannot transform {col}") from e

        return x
