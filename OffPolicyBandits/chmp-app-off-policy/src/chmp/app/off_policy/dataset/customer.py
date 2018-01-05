import argparse
import logging
import os.path

import numpy as np
import pandas as pd
import patsy

_logger = logging.getLogger(__name__)
_basedir = os.path.abspath(os.path.dirname(__file__))


def create(data_path):
    target_fname = os.path.join(data_path, 'customer.parquet')
    if os.path.exists(target_fname):
        _logger.info('skip %s, does already exist', target_fname)
        return

    _logger.info('create %s', target_fname)
    generate_customer_data(n_samples=100_000).to_parquet(target_fname, engine='pyarrow', compression='brotli')


def generate_customer_data(
        n_samples=100_000,
        n_occupations=5,
        seed=24,
        p_random=0.05,
        train_ratio=0.75,
):
    """Generate a dataset designed to resemble customer datasets.
    """

    np.random.seed(seed)
    age_latent = sigmoid(np.random.normal(size=n_samples))
    gender_latent = sigmoid(np.random.normal(size=n_samples))
    occupation_latent = sample_categorical(n_occupations, size=n_samples, alpha=0.9)

    generic_latent = sigmoid(np.random.normal(size=n_samples))

    salary_latent = (
        spline(
            sample_cauchy(size=(1, n_occupations)) +
            np.random.normal(loc=1, scale=0.75, size=(10, n_occupations)).cumsum(axis=0),
            age_latent,
        ) +
        spline(
            sample_cauchy(size=(1, n_occupations)) +
            np.random.normal(loc=3, scale=0.75, size=(10, n_occupations)).cumsum(axis=0),
            gender_latent,
        )
    )
    salary_latent = salary_latent[np.arange(n_samples), occupation_latent]

    dist_city_latent = (
        spline(
            np.random.laplace(loc=+1, scale=0.4, size=(10, n_occupations)).cumsum(axis=0),
            age_latent,
        )
    )
    dist_city_latent = dist_city_latent[np.arange(n_samples), occupation_latent]

    count_mean = normalize(generic_latent) * np.random.gamma(10, 10 / 5, size=n_samples)
    count = np.random.poisson(count_mean)

    occupation_delta = np.random.laplace(size=n_occupations, loc=0.5, scale=1.5)

    effect_noise_0 = np.random.normal(scale=0.4, size=n_samples)
    effect_noise_1 = np.random.normal(scale=0.4, size=n_samples)

    generic = spline(
        np.random.laplace(loc=+0, scale=0.4, size=10).cumsum(axis=0),
        generic_latent,
    )

    data = pd.DataFrame()

    data['age'] = spline(
        [20, 25, 30, 45, 60, 85],
        normalize(age_latent) + np.random.normal(scale=0.05, size=n_samples),
    )
    data['gender'] = ((gender_latent + np.random.normal(scale=0.05, size=n_samples)) > 0.5).astype(float)
    data['salary'] = spline(
        [30, 35, 45, 60, 70, 80, 90, 100],
        normalize(salary_latent) + np.random.normal(scale=0.05, size=n_samples),
    )
    data['dist_city'] = sigmoid(
        3 * normalize(dist_city_latent) - 1.5 + np.random.normal(scale=0.05, size=n_samples)
    )
    data['occupation'] = random_cat_swaps(occupation_latent, eps=5e-2)

    data['generic'] = generic
    data['count'] = count

    data['outcome_mean_p_det'] = sigmoid(
        -0.0 +
        2.5 * (normalize(age_latent) - 0.35) +
        3 * (normalize(salary_latent) - 0.5)
    )

    data['outcome_delta_p_det'] = sigmoid(
        0.55 +
        -0.15 * occupation_delta[occupation_latent] * normalize(salary_latent) ** 2 +
        -0.5 * normalize(age_latent) ** 2 +
        +1.0 * (normalize(gender_latent) - 0.5) * (normalize(generic_latent) - 0.15)
    )

    data['outcome_0_p_det'] = sigmoid(logit(data['outcome_mean_p_det']) - 0.5 * logit(data['outcome_delta_p_det']))
    data['outcome_1_p_det'] = sigmoid(logit(data['outcome_mean_p_det']) + 0.5 * logit(data['outcome_delta_p_det']))

    data['outcome_0_p'] = sigmoid(
        logit(data['outcome_0_p_det']) - 0.2 * effect_noise_0 * occupation_delta[occupation_latent]
    )
    data['outcome_1_p'] = sigmoid(
        logit(data['outcome_1_p_det']) + 0.2 * effect_noise_1 * occupation_delta[occupation_latent]
    )

    cutoff = logit(p_random)

    data['action_p'] = (data['age'] - 40) / 1.5
    data['action_p'] = sigmoid(np.clip(data['action_p'], -cutoff, +cutoff))
    data['action'] = sample_bernoulli(data['action_p'])

    data['outcome_p'] = (
        (data['action'] == 1) * data['outcome_1_p'] +
        (data['action'] == 0) * data['outcome_0_p']
    )

    data['outcome'] = sample_bernoulli(data['outcome_p'])

    data['train'] = 1
    data.iloc[int(train_ratio * len(data)):, data.columns.get_loc('train')] = 0

    return data


def reject(proposal, accept, max_iter=10):
    res = []

    for _ in range(max_iter):
        cand = proposal()
        n_target = cand.shape[0]

        sel = accept(cand)
        cand = cand[sel]

        res = np.concatenate([res, cand])
        n_current = res.shape[0]

        if n_current >= n_target:
            return res[:n_target]

    raise ValueError()


def sample_bernoulli(p):
    u = np.random.uniform(size=np.shape(p))
    return (u < p).astype(float)


def sigmoid(x):
    x = np.clip(x, -1e2, +1e2)
    return 1. / (1. + np.exp(-x))


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sample_categorical(n_categories, size, alpha=0.5):
    p = np.random.dirichlet([alpha] * n_categories)
    return np.random.choice(np.arange(n_categories), size=size, p=p)


def spline(w, x):
    w = np.asarray(w)
    x = np.asarray(x)
    splines = patsy.bs(
        x,
        df=w.shape[0],
        lower_bound=np.min(x),
        upper_bound=np.max(x),
        include_intercept=True,
    )
    return np.dot(splines, w)


def sample_cauchy(loc=0, scale=1, size=1):
    u = np.random.uniform(size=size)
    return loc + scale * np.tan(np.pi * (u + 0.5))


def normalize(x):
    return (x - np.min(x)) / np.ptp(x)


def random_cat_swaps(x, eps=0.05):
    n_categories = np.max(x) + 1

    u = np.random.uniform(size=np.size(x))
    sel = u < eps

    x = x.copy()
    x[sel] = np.random.randint(0, n_categories, size=sel.sum())
    return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    args = parser.parse_args()

    create(args.data_path)

