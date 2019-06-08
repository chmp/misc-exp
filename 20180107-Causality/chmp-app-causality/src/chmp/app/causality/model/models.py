"""Regression based policy
"""
import numpy as np
import pandas as pd
import sklearn.pipeline
import sklearn.ensemble

from chmp.ds import (
    FuncTransformer,
    FilterLowFrequencyTransfomer,
    find_categorical_columns,
    Loop,
)

from .util import CategoricalMeanTargetEncoder, CategoricalIndexEncoder


def build_standard_sklearn_classifier(features):
    return sklearn.pipeline.Pipeline(
        [
            ("select", FuncTransformer(lambda df: df[list(features)])),
            ("filter", FilterLowFrequencyTransfomer()),
            ("one-hot", CategoricalMeanTargetEncoder()),
            (
                "predict",
                sklearn.ensemble.GradientBoostingClassifier(
                    n_estimators=100, subsample=0.80, max_depth=5
                ),
            ),
        ]
    )


def get_weight_keyword(est):
    if isinstance(est, sklearn.pipeline.Pipeline):
        key, _ = est.steps[-1]
        return f"{key}__sample_weight"

    return "sample_weight"


def binary_offset_tree_transform(reward, action, action_propensity=None):
    # use offset tree transformation (see Beygelzimer & Langford, 2016)
    # NOTE: objective function is invariant to additive and multiplicative reward transformations
    reward = (reward - np.min(reward)) / np.ptp(reward)

    fit_weight = np.abs(reward - 0.5)
    if action_propensity is not None:
        fit_weight = fit_weight / action_propensity

    fit_target = (reward >= 0.5) * action + (reward < 0.5) * (1 - action)

    return fit_weight, fit_target


class RegressionPolicy:
    def predict_value(self, df):
        raise NotImplementedError()

    def predict_action(self, df):
        action, _ = self.predict(df)
        return action

    def predict(self, df):
        # TODO: return actions also a probabilities
        value = self.predict_value(df)
        action = np.concatenate(
            [
                np.asarray(value[:, None, 1] < value[:, None, 0], dtype=np.int),
                np.asarray(value[:, None, 1] >= value[:, None, 0], dtype=np.int),
            ],
            axis=-1,
        )

        return action, value


# TODO: support re-weighting for fitting
class BinaryOutcomeRegressionPolicy(RegressionPolicy):
    """Use a classifier for value estimation.

    .. note::

        the action column has to be part of the features for the classifier to
        obtain reasonable results.

    """

    def __init__(self, est, action_column="action", target_column="outcome"):
        self.est = est
        self.action_column = action_column
        self.target_column = target_column

    def fit(self, df):
        df = df.reset_index(drop=True)
        self.est.fit(df, np.asarray(df[self.target_column]))
        return self

    def predict_value(self, df):
        _, value_0 = self.est.predict_proba(df.assign(**{self.action_column: 0})).T
        _, value_1 = self.est.predict_proba(df.assign(**{self.action_column: 1})).T
        return np.concatenate([value_0[..., None], value_1[..., None]], axis=-1)


class DirectClassifierPolicy:
    def __init__(
        self,
        est,
        action_column="action",
        reward_column="outcome",
        propensity_column=None,
        sample_weight_keyword=None,
        clipping_value=1e-4,
    ):
        if sample_weight_keyword is None:
            sample_weight_keyword = get_weight_keyword(est)

        # TODO: implement propensity estimation
        if propensity_column is None:
            raise RuntimeError("propensity estimation not yet supported")

        self.est = est
        self.action_column = action_column
        self.reward_column = reward_column
        self.propensity_column = propensity_column
        self.sample_weight_keyword = sample_weight_keyword
        self.clipping_value = clipping_value

    def fit(self, df):
        df = df.reset_index(drop=True)

        reward = np.asarray(df[self.reward_column])
        action = df[self.action_column]

        # clip the propensity to reduce variance
        propensity = df[self.propensity_column]
        propensity = np.maximum(self.clipping_value, propensity)

        # fit the classifier
        fit_weight, fit_target = binary_offset_tree_transform(
            reward=reward, action=action, action_propensity=propensity
        )
        self.est.fit(df, fit_target, **{self.sample_weight_keyword: fit_weight})

        return self

    def predict(self, df):
        return self.est.predict_proba(df), None


class DoublyRobustClassifierPolicy:
    """Use doubly robust policy estimation.

    .. note::

        this classifier will enlarge the dataset by a factor of
        ``len(action_values)`` during fit.

    """

    def __init__(
        self,
        value_est,
        action_est,
        action_values,
        action_column="action",
        reward_column="outcome",
        propensity_column=None,
        sample_weight_keyword=None,
        clipping_value=1e-4,
    ):
        if sample_weight_keyword is None:
            sample_weight_keyword = get_weight_keyword(action_est)

        # TODO: implement propensity estimation
        if propensity_column is None:
            raise RuntimeError("propensity estimation not yet supported")

        self.value_est = value_est
        self.action_est = action_est
        self.action_values = action_values

        self.action_column = action_column
        self.reward_column = reward_column
        self.propensity_column = propensity_column
        self.sample_weight_keyword = sample_weight_keyword
        self.clipping_value = clipping_value

    def fit(self, df):
        df = df.reset_index(drop=True)

        reward = np.asarray(df[self.reward_column])
        observed_action = df[self.action_column]

        # clip the propensity to reduce variance
        propensity = df[self.propensity_column]
        propensity = np.maximum(self.clipping_value, propensity)

        # TODO: offer chance of re-weighting?
        self.value_est.fit(df, reward)

        full_df = []
        full_reward = []

        for action in self.action_values:
            df_with_action = df.assign(**{self.action_column: action})
            reward_estimate = self.value_est.predict(df_with_action)

            full_reward.append(
                (reward - reward_estimate) / propensity * (action == observed_action)
                + reward_estimate
            )
            full_df.append(df_with_action)

        full_reward = np.concatenate(full_reward, axis=0)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)

        # fit the classifier
        fit_weight, fit_target = binary_offset_tree_transform(
            reward=full_reward, action=np.asarray(full_df["action"])
        )
        self.action_est.fit(
            full_df, fit_target, **{self.sample_weight_keyword: fit_weight}
        )

        return self

    def predict(self, df):
        values = [
            self.value_est.predict(df.assign(**{self.action_column: action}))
            for action in self.action_values
        ]
        values = np.stack(values).T

        return self.action_est.predict_proba(df), values
