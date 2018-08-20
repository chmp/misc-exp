import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from chmp.app.causality.tabular import prob


def test_prob__conditional_transform():
    df = pd.DataFrame().assign(
        z=lambda df: np.random.binomial(1, p=0.8, size=10_000),
        x=lambda df: np.random.binomial(1, p=0.4 + 0.4 * df['z']),
        y=lambda df: np.random.binomial(1, p=0.1 + 0.4 * df['z'] + 0.4 * df['x']),
    )
    counts = df.groupby(['y', 'z', 'x']).size()

    pdt.assert_almost_equal(
        counts.pipe(prob, ['x'], ['z']).sort_index(),
        (
            counts
            .pipe(prob, ['x'], ['z'], transform=True)
            .groupby(level=['x', 'z'])
            .mean()
            .sort_index()
        ),
    )
