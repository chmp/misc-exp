"""An interpretation of Hierarchical Sampling for Active Learning.

Changes:

* use dirichlet-multinomial to estimate probabilities and errors
* use simple error reduction for splits (i.e., no admissable set)
* search of split to reduce error
"""
import itertools as it
import math
import random

import numba
import numpy as np
import scipy.cluster.hierarchy


class ActiveLearner:
    @classmethod
    def from_data(cls, x, *, n_classes, concentration=0.1, **kwargs):
        if not hasattr(x, 'shape'):
            x = np.asarray(x)

        if len(x.shape) == 1:
            x = x[:, None]

        linkage = scipy.cluster.hierarchy.linkage(x, **kwargs)
        return cls(linkage, n_classes, concentration=concentration)

    def __init__(self, linkage, n_classes, concentration=0.1, max_search_depth=3):
        self.clustering = Clustering(linkage)
        self.n_leaves = len(linkage) + 1
        self.n_nodes = 2 * len(linkage) + 1
        self.n_classes = int(n_classes)
        self.concentration = concentration
        self.max_search_depth = max_search_depth

        # counts, proba, proba-upper, proba-lower, weight, error
        self.class_count = np.zeros((self.n_nodes, self.n_classes), dtype=np.int64)
        self.class_proba = np.zeros((self.n_nodes, self.n_classes), dtype=np.float64)
        self.node_weight = np.zeros(self.n_nodes, dtype=np.int64)
        self.error = np.zeros(self.n_nodes, dtype=np.float64)

        self.pruning = [self.root]

        self._update_summaries()

    @property
    def root(self):
        return ClusterInfo(self, -1)

    @property
    def linkage(self):
        return self.clustering.linkage

    def update(self, examples, classes):
        for example, class_ in zip(examples, classes):
            self._add(example, class_)

        self._update_summaries()

    def add(self, example, class_):
        self._add(example, class_)
        self._update_summaries()

    def _add(self, example, class_):
        if example >= self.n_leaves:
            raise ValueError('can only update counts for leaves')

        _active_learner_add(self.clustering.parents, self.class_count, example, class_)

    def _update_summaries(self):
        self.node_weight[:] = self.class_count.sum(axis=1)

        alpha_i = self.concentration + self.class_count
        alpha_0 = self.n_classes * self.concentration + self.node_weight
        alpha_0 = alpha_0[:, None]

        self.class_proba[:] = alpha_i / alpha_0
        delta = np.sqrt((alpha_i * (alpha_0 - alpha_i)) / (alpha_0 * alpha_0 * (alpha_0 + 1)))
        proba_lower = np.maximum(0, self.class_proba - delta)

        self.error[:] = np.sum(self.class_proba * (1 - proba_lower), axis=1)

    def split(self):
        self.pruning = [
            cluster
            for top_level in self.pruning
            for cluster in min(
                search_splittings(top_level, self.max_search_depth),
                key=calculate_splitting_error,
            )
        ]

    def propose_batch(self, size):
        cluster_weights = np.zeros(len(self.pruning), dtype=np.float64)
        root_weight = max(1, self.root.weight)

        # probably mistake in paper to use class_proba_upper
        for idx, cluster in enumerate(self.pruning):
            cluster_weights[idx] = max(1, cluster.weight) / root_weight * cluster.error

        leaf_weights = np.zeros(self.n_leaves, dtype=np.float64)
        for cluster, p in zip(self.pruning, cluster_weights):
            self.clustering.mark_leaves(cluster.id, p, leaf_weights)

        # select only unseed weights
        leaf_weights = leaf_weights * (self.node_weight[:self.n_leaves] == 0)

        assert np.all(leaf_weights >= 0)

        # normalize
        # TODO: check for nulls
        leaf_weights = leaf_weights / leaf_weights.sum()

        return np.random.choice(np.arange(self.n_leaves), size=size, p=leaf_weights, replace=False)

    def fill_proba(self):
        # use the pruning to fill cluster w/ their probabilities
        cluster_proba = np.zeros((self.n_leaves, self.n_classes), dtype=np.float)
        for cluster in self.pruning:
            for class_ in range(self.n_classes):
                self.clustering.mark_leaves(
                    cluster.id, self.class_proba[cluster.id, class_], cluster_proba[:, class_],
                )

        # get the observed probabilities
        observed_proba = self.class_count[:self.n_leaves]
        observed_proba = observed_proba / np.maximum(1, np.sum(observed_proba, axis=1, keepdims=True))

        # kepp observed probabiities, use cluster as a stand-in
        observed_leaves = self.observed_leaves[:, None]
        return (1 - observed_leaves) * cluster_proba + observed_leaves * observed_proba

    def sample_class(self):
        class_proba = self.fill_proba()
        return random_multinomial(class_proba)

    @property
    def observed_leaves(self):
        return self.node_weight[:self.n_leaves] > 0

    def get_cluster_membership(self):
        result = -np.ones(self.n_leaves, dtype=np.int64)
        for idx, cluster in enumerate(self.pruning):
            self.clustering.mark_leaves(cluster.id, idx, result)

        return result


@numba.jit(
    numba.void(numba.int64[:], numba.int64[:, :], numba.int64, numba.int64),
    nopython=True, nogil=True,
)
def _active_learner_add(parents, class_count, current, class_):
    while current >= 0:
        class_count[current, class_] += 1.0
        current = parents[current]


@numba.jit(numba.int64[:](numba.float64[:, :]))
def random_multinomial(p):
    # TODO: check input?
    res = np.empty(p.shape[0], dtype=np.int64)

    for i in range(p.shape[0]):
        u = random.random()

        j = 0
        for j in range(p.shape[1]):
            u -= p[i, j]
            if u < 0:
                break

        res[i] = j

    return res


def search_splittings(root, max_depth):
    if root.weight < 1:
        return

    yield [root]

    if max_depth == 0:
        return

    if not root.is_leaf:
        left_splits = list(search_splittings(root.left, max_depth=max_depth - 1))
        right_splits = list(search_splittings(root.right, max_depth=max_depth - 1))

        yield from (
            [*a, *b]
            for a, b in it.product(left_splits, right_splits)
        )


def calculate_splitting_error(splitting):
    return (
        sum(cluster.weight * cluster.error for cluster in splitting) /
        sum(cluster.weight for cluster in splitting)
    )


class ClusterInfo:
    def __init__(self, learner, cluster):
        if cluster < 0:
            cluster = learner.n_nodes + cluster

        assert 0 <= cluster < learner.n_nodes

        self.learner = learner
        self.id = cluster

    @property
    def weight(self):
        return self.learner.node_weight[self.id]

    @property
    def error(self):
        return self.learner.error[self.id]

    @property
    def is_leaf(self):
        return self.id < self.learner.n_leaves

    @property
    def left(self):
        assert not self.is_leaf
        return ClusterInfo(
            self.learner,
            self.learner.linkage[self.id - self.learner.n_leaves, 0],
        )

    @property
    def right(self):
        assert not self.is_leaf
        return ClusterInfo(
            self.learner,
            self.learner.linkage[self.id - self.learner.n_leaves, 1],
        )

    @property
    def should_split(self):
        if self.is_leaf:
            return False

        if self.weight == 0:
            return False

        left = self.left
        right = self.right

        if left.weight == 0 or right.weight == 0:
            return False

        combined_error = (
            left.weight / self.weight * left.error +
            right.weight / self.weight * right.error
        )
        return combined_error < self.error


class Clustering:
    def __init__(self, linkage):
        if isinstance(linkage, dict):
            self.linkage = np.zeros((len(linkage), 2), dtype=np.int64)
            for i in range(len(linkage) + 1, 2 * len(linkage) + 1):
                self.linkage[i - len(linkage) - 1, :] = linkage[i]

        else:
            linkage = np.asarray(linkage, dtype=np.int64)
            self.linkage = linkage[:, :2]

        self._parents = None

    @property
    def parents(self):
        if self._parents is None:
            self._parents = np.zeros(2 * len(self.linkage) + 1, dtype=np.int64)
            _mark_parents(self.linkage, 2 * len(self.linkage), -1, self._parents)

        return self._parents

    def mark_leaves(self, cluster_id, value=None, out=None):
        if value is None:
            value = cluster_id

        value_dtype = np.asarray(value).dtype
        if out is None:
            out = -np.ones(len(self.linkage) + 1, dtype=value_dtype)

        else:
            out = np.asarray(out, dtype=value_dtype)

        _mark_leaves(self.linkage, cluster_id, value, out)
        return out


@numba.jit(
    numba.void(numba.int64[:, :], numba.int64, numba.int64, numba.int64[:]),
    nopython=True, nogil=True,
)
def _mark_parents(linkage, child, parent, out):
    if child > 2 * len(linkage):
        raise ValueError()

    out[child] = parent

    if child > len(linkage):
        _mark_parents(linkage, linkage[child - len(linkage) - 1, 0], child, out)
        _mark_parents(linkage, linkage[child - len(linkage) - 1, 1], child, out)


@numba.jit(
    [
        numba.void(numba.int64[:, :], numba.int64, numba.int64, numba.int64[:]),
        numba.void(numba.int64[:, :], numba.int64, numba.float64, numba.float64[:]),
    ],
    nopython=True, nogil=True,
)
def _mark_leaves(linkage, cluster_id, value, out):
    if cluster_id <= len(linkage):
        out[cluster_id] = value

    elif cluster_id <= 2 * len(linkage):
        _mark_leaves(linkage, linkage[cluster_id - len(linkage) - 1, 0], value, out)
        _mark_leaves(linkage, linkage[cluster_id - len(linkage) - 1, 1], value, out)

    else:
        raise ValueError()
