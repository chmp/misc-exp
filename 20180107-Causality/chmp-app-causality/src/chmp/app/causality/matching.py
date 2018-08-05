import numpy as np
import pandas as pd


def psmatch(ps, treatment, *, m=1):
    df = pd.DataFrame({"score": ps, "treatment": treatment})
    return match(df, treatment="treatment", covariates=["score"], m=m)


def match(df, *, treatment, covariates, m=1):
    covariates = list(covariates)

    df_0 = df.loc[df[treatment] == 0, covariates]
    df_1 = df.loc[df[treatment] == 1, covariates]

    if len(df_1) < len(df_0):
        needles, haystack = df_1, df_0

    else:
        needles, haystack = df_0, df_1

    inv_cov = np.linalg.inv(np.asarray(df[covariates].cov()))

    needles = np.asarray(needles)
    haystack = np.asarray(haystack)

    needle_indices = np.arange(len(needles))
    np.random.shuffle(needle_indices)

    all_indices = np.arange(len(haystack), dtype=np.int64)
    matched_indices = np.zeros(m * len(needles), dtype=np.int64)
    taken = np.zeros(len(haystack), dtype=np.bool)

    for k in range(len(needles)):
        i = needle_indices[k]

        available_indices = all_indices[~taken]
        dx = haystack[~taken] - needles[i, None, :]

        dd = (dx * (dx @ inv_cov)).sum(axis=1)

        selected = np.argsort(dd)[:m]
        selected_indices = available_indices[selected]
        matched_indices[k * m : (k + 1) * m] = selected_indices
        taken[selected_indices] = True

    assert np.sum(taken) == m * len(needles)

    if len(df_1) < len(df_0):
        return df_0.index[matched_indices], df_1.index[needle_indices]

    else:
        return df_0.index[needle_indices], df_1.index[matched_indices]
