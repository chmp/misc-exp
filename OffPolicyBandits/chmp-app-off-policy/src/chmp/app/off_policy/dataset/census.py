"""Create a off-policy data set from the census classification data set
"""
import argparse
import logging
import os.path

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.pipeline

from chmp.ds import FuncTransformer, OneHotEncoder, DataFrameEstimator, FilterLowFrequencyTransfomer

_logger = logging.getLogger(__name__)

source_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/'

columns = [
    ('age', 'float'), ('class of worker', 'category'), ('industry code', 'category'),
    ('occupation code', 'category'), ('education', 'category'), ('wage per hour', 'float'),
    ('enrolled in edu inst last wk', 'category'), ('marital status', 'category'),
    ('major industry code', 'category'), ('major occupation code', 'category'),
    ('race', 'category'), ('hispanic origin', 'category'), ('sex', 'category'),
    ('member of a labor union', 'category'), ('reason for unemployment', 'category'),
    ('full or part time employment stat', 'category'), ('capital gains', 'float'),
    ('capital losses', 'float'), ('divdends from stocks', 'float'),
    ('tax filer status', 'category'), ('region of previous residence', 'category'),
    ('state of previous residence', 'category'), ('household and family stat', 'category'),
    ('household summary in household', 'category'), ('instance weight', 'float'),
    ('migration code-change in msa', 'category'), ('migration code-change in reg', 'category'),
    ('migration code-move within reg', 'category'), ('live in this house 1 year ago', 'category'),
    ('migration prev res in sunbelt', 'category'), ('num persons worked for employer', 'float'),
    ('parents present', 'category'), ('country of birth father', 'category'),
    ('country of birth mother', 'category'), ('country of birth self', 'category'),
    ('citizenship', 'category'), ('own business or self employed', 'category'),
    ('fill inc questionnaire for veterans admin', 'category'), ('veterans benefits', 'category'),
    ('weeks worked in year', 'float'), ('year', 'category'), ('target', 'category'),
]

features = [name for name, _ in columns]


def create(data_path):
    def rename(df):
        df.columns = [name for name, _ in columns]
        return df

    data_fname = os.path.join(data_path, 'census-income.data.gz')
    test_fname = os.path.join(data_path, 'census-income.test.gz')
    target_fname = os.path.join(data_path, 'census.parquet')

    if os.path.exists(target_fname):
        _logger.info('target %s already exists, skip it', target_fname)
        return

    if not os.path.exists(data_fname) or not os.path.exists(test_fname):
        raise RuntimeError(
            'Could not find input data. Download census-income.data.gz and census-income.test.gz from ' + source_url
        )

    _logger.info('load census data')
    train_data = pd.read_csv(data_fname, sep=',', header=None).pipe(rename).assign(train=1)
    test_data = pd.read_csv(test_fname, sep=',', header=None).pipe(rename).assign(train=0)

    df = generate_census_data(train_data, test_data)

    _logger.info('write %s', target_fname)
    df.to_parquet(target_fname, engine='pyarrow', compression='brotli')


def generate_census_data(train_data, test_data):
    df = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    _logger.info('convert columns')
    for col, typ in columns:
        df = df.assign(**{col: df[col].astype(typ)})

    df = df.assign(target=(df['target'].str.strip() == '50000+.').astype('int'))

    # ensure the targets are more equally distributed
    df = pd.concat([
        df[df['target'] == 0].sample(n=2 * len(df) // 3, replace=True),
        df[df['target'] == 1].sample(n=1 * len(df) // 3, replace=True),
    ], axis=0, ignore_index=True)

    # shuffle the df
    df = df.sample(frac=1.0)

    _logger.info('fit policy')
    est = assemble_census_policy_pipeline()
    est.fit(df.iloc[:10_000], 'target')

    _logger.info('determine action and outcome')
    _, df['action_p'] = est.predict_proba(df).T

    # use the class probability for the action
    df['action_p'] = (
        (df['target'] == 1) * df['action_p'] +
        (df['target'] == 0) * (1 - df['action_p'])
    )

    df['action'] = sample_bernoulli(df['action_p'])

    gain = 0.1 * ((df['target'] > 0.5) == (df['action'] > 0.5))
    mean = 3 * (1 + gain) * (0.5 + df['target'])

    df['value_mean'] = mean
    df['value'] = np.random.normal(mean)

    df['outcome_0_p_det'] = 0.7 * (df['target'] > 0.5) + 0.3 * (df['target'] < 0.5)
    df['outcome_1_p_det'] = 0.8 * (df['target'] > 0.5) + 0.2 * (df['target'] < 0.5)

    df['outcome_p'] = (1 - df['action']) * df['outcome_0_p_det'] + df['action'] * df['outcome_1_p_det']
    df['outcome'] = sample_bernoulli(df['outcome_p'])

    return df


def assemble_census_policy_pipeline(C=5e-4):
    est = sklearn.pipeline.Pipeline([
        ('select', FuncTransformer(lambda x: x.drop(
            ['target', 'capital gains', 'capital losses', 'divdends from stocks', 'instance weight'],
            axis=1,
        ))),
        ('filter-low-frequency', FilterLowFrequencyTransfomer()),
        ('one-hot', OneHotEncoder()),
        ('predict', sklearn.linear_model.LogisticRegression(C=C, random_state=15506)),
    ])
    return DataFrameEstimator(est)


def sample_bernoulli(p):
    u = np.random.uniform(size=np.shape(p))
    return (u < p).astype(float)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    args = parser.parse_args()

    create(args.data_path)
