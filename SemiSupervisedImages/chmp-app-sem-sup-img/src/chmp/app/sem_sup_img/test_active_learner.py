import numpy as np

from chmp.app.sem_sup_img.active_learner import ActiveLearner, Clustering


linkage = [
    [2, 3],  # 5
    [0, 1],  # 6
    [4, 6],  # 7
    [5, 7],  # 8
]


def test_clustering__mark_children():
    clustering = Clustering(linkage)

    np.testing.assert_almost_equal(clustering.mark_leaves(0), [+0, -1, -1, -1, -1])
    np.testing.assert_almost_equal(clustering.mark_leaves(1), [-1, +1, -1, -1, -1])
    np.testing.assert_almost_equal(clustering.mark_leaves(2), [-1, -1, +2, -1, -1])
    np.testing.assert_almost_equal(clustering.mark_leaves(3), [-1, -1, -1, +3, -1])
    np.testing.assert_almost_equal(clustering.mark_leaves(4), [-1, -1, -1, -1, +4])

    np.testing.assert_almost_equal(clustering.mark_leaves(5), [-1, -1, +5, +5, -1])
    np.testing.assert_almost_equal(clustering.mark_leaves(6), [+6, +6, -1, -1, -1])
    np.testing.assert_almost_equal(clustering.mark_leaves(7), [+7, +7, -1, -1, +7])
    np.testing.assert_almost_equal(clustering.mark_leaves(8), [+8, +8, +8, +8, +8])


def test_clustering__parents():
    clustering = Clustering(linkage)

    np.testing.assert_almost_equal(
        clustering.parents,
        [+6, +6, +5, +5, +7, +8, +7, +8, -1],
    )


def test_active_learner__add():
    learner = ActiveLearner(
        {5: [2, 3], 6: [0, 1], 7: [4, 6], 8: [5, 7]},
        n_classes=3,
        concentration=0,
    )

    learner.update([0, 2, 4], [1, 2, 2])

    np.testing.assert_almost_equal(learner.node_weight, [1, 0, 1, 0, 1, 1, 1, 2, 3])
    np.testing.assert_almost_equal(learner.majority_class, [1, 0, 2, 0, 2, 2, 1, 1, 2])
    np.testing.assert_almost_equal(learner.class_count, [
        [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1],
        [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2],
    ])

    np.testing.assert_almost_equal(learner.class_proba, [
        [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1],
        [0, 0, 1], [0, 1, 0], [0, 1 / 2, 1 / 2], [0, 1 / 3, 2 / 3],
    ])
