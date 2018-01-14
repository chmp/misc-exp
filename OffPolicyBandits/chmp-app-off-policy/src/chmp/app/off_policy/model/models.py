"""Regression based policy
"""
import numpy as np
import pandas as pd
import sklearn.pipeline
import sklearn.ensemble
import tensorflow as tf

from chmp.ds import FuncTransformer, FilterLowFrequencyTransfomer, find_categorical_columns
from chmp.experiment import Loop
from chmp.ml import PickableTFModel
from chmp.ml.layers import factorized

from .util import CategoricalMeanTargetEncoder, CategoricalIndexEncoder


def build_standard_sklearn_classifier(features):
    return sklearn.pipeline.Pipeline([
        ('select', FuncTransformer(lambda df: df[list(features)])),
        ('filter', FilterLowFrequencyTransfomer()),
        ('one-hot', CategoricalMeanTargetEncoder()),
        ('predict', sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=100, subsample=0.80, max_depth=5,
        )),
    ])


def get_weight_keyword(est):
    if isinstance(est, sklearn.pipeline.Pipeline):
        key, _ = est.steps[-1]
        return f'{key}__sample_weight'

    return 'sample_weight'


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
        action = np.concatenate([
            np.asarray(value[:, None, 1] < value[:, None, 0], dtype=np.int),
            np.asarray(value[:, None, 1] >= value[:, None, 0], dtype=np.int),
        ], axis=-1)

        return action, value


# TODO: support re-weighting for fitting
class BinaryOutcomeRegressionPolicy(RegressionPolicy):
    """Use a classifier for value estimation.

    .. note::

        the action column has to be part of the features for the classifier to
        obtain reasonable results.

    """
    def __init__(self,  est, action_column='action', target_column='outcome'):
        self.est = est
        self.action_column = action_column
        self.target_column= target_column

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
            self, est,
            action_column='action',
            reward_column='outcome',
            propensity_column=None,
            sample_weight_keyword=None,
            clipping_value=1e-4
    ):
        if sample_weight_keyword is None:
            sample_weight_keyword = get_weight_keyword(est)

        # TODO: implement propensity estimation
        if propensity_column is None:
            raise RuntimeError('propensity estimation not yet supported')

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
            reward=reward, action=action, action_propensity=propensity,
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
        self, value_est, action_est, action_values,
        action_column='action',
        reward_column='outcome',
        propensity_column=None,
        sample_weight_keyword=None,
        clipping_value=1e-4,
    ):
        if sample_weight_keyword is None:
            sample_weight_keyword = get_weight_keyword(action_est)

        # TODO: implement propensity estimation
        if propensity_column is None:
            raise RuntimeError('propensity estimation not yet supported')

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
                (reward - reward_estimate) / propensity * (action == observed_action) +
                reward_estimate
            )
            full_df.append(df_with_action)

        full_reward = np.concatenate(full_reward, axis=0)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)

        # fit the classifier
        fit_weight, fit_target = binary_offset_tree_transform(
            reward=full_reward, action=np.asarray(full_df['action'])
        )
        self.action_est.fit(full_df, fit_target, **{self.sample_weight_keyword: fit_weight})

        return self

    def predict(self, df):
        values = [
            self.value_est.predict(df.assign(**{self.action_column: action}))
            for action in self.action_values
        ]
        values = np.stack(values).T

        return self.action_est.predict_proba(df), values


# TODO: fix (by allowing to bind to a session?)
class BinaryOutcomeFactorizationPolicy(RegressionPolicy, PickableTFModel):
    def __init__(self, features, action_column='action', target_column='outcome', epochs=10, batch_size=10):
        super().__init__()

        self.features = features
        self.action_column = action_column
        self.target_column = target_column
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_scores = None

        self.preproc_est = sklearn.pipeline.Pipeline([
            ('select', FuncTransformer(lambda df: df[self.features + [self.action_column]])),
            ('filter', FilterLowFrequencyTransfomer()),
            ('one-hot', CategoricalIndexEncoder()),
        ])

    def fit(self, df, session=None):
        df = df.reset_index(drop=True)
        df = self.preproc_est.fit_transform(df, np.asarray(df[self.target_column]))

        self._build(session.graph)

        session.run(tf.global_variables_initializer())

        outer_loop = Loop()

        self.train_scores = []
        for _ in outer_loop(range(self.epochs)):
            self.train_scores += self._epoch(outer_loop, session, df)
            print()

    def predict_value(self, df, session=None):
        scores = []

        for start in range(0, len(df), self.batch_size):
            end = start + self.batch_size
            batch = df.iloc[start:end]
            score = session.run(self.score_, self._feed_dict(batch))

            scores += [score]

        return np.concatenate(scores, axis=0)

    def _epoch(self, outer_loop, session, df):
        loop = Loop()

        batch_indices = list(range(0, len(df), self.batch_size))
        batch_indices = list(zip(batch_indices[:-1], batch_indices[1:]))

        df = df.sample(frac=1).reset_index(drop=True)

        losses = []
        for start, end in loop(batch_indices):
            batch = df.iloc[start:end]
            loss, _ = session.run([self.loss_, self.train_], self._feed_dict(batch))
            losses += [loss]

            print(f'{outer_loop:[fr} {loop} loss: {loss:.2f}'.ljust(120), end='\r')

        return losses

    def _feed_dict(self, df):
        return {
            placeholder: df[key]
            for key, placeholder in self.features.items()
        }

    def _build(self, df, graph=None):
        categorical_columns = find_categorical_columns(df)
        numeric_columns = [col for col in df if not col in categorical_columns]
        self.category_count = {col: df[col].max() + 1 for col in categorical_columns}

        with self.valid_graph(graph):

            self.features = {
                'outcome': tf.placeholder(tf.float32, shape=[None]),
                'action': tf.placeholder(tf.float32, shape=[None]),
                'weight': tf.placeholder(tf.float32, shape=[None]),
            }
            self.features.update({
                key: tf.placeholder(tf.float32, shape=[None])
                for key in numeric_columns
            })
            self.features.update({
                key: tf.placeholder(tf.int32, shape=[None])
                for key in categorical_columns
            })

            feature_columns = (
                [tf.feature_column.numeric_column('action')] +
                [tf.feature_column.numeric_column(key) for key in numeric_columns] +
                [
                    tf.feature_column.indicator_column(
                        tf.feature_column.categorical_column_with_identity(
                            key, self.category_count[key]
                        ),
                    )
                    for key in categorical_columns
                ]
            )

            inputs_ = tf.feature_column.input_layer(self.features, feature_columns)

            # TODO: implement advantage functions ...
            x_ = inputs_
            x_ = tf.concat([inputs_, tf.layers.dense(x_, 30, activation=tf.nn.elu), factorized(x_, 30)], axis=1)
            x_ = tf.concat([inputs_, tf.layers.dense(x_, 30, activation=tf.nn.elu), factorized(x_, 30)], axis=1)
            x_ = tf.layers.dense(x_, units=1, activation=None)

            self.score_ = tf.sigmoid(x_)

            self.loss_ = tf.losses.sigmoid_cross_entropy(
                tf.reshape(self.features['outcome'], [-1, 1]), x_,
                reduction=tf.losses.Reduction.MEAN,
            )

            self.train_ = tf.train.AdamOptimizer(5e-4).minimize(self.loss_)
