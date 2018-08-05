import bz2
import os.path
import sys

import pandas as pd
import requests

from chmp.ds import Object


def read_rhc_df(fname, download=True):
    if download:
        ensure_rhc_data_exists(fname)

    df = pd.read_csv(fname)

    # fix the index name
    df.columns = ['index', *df.columns[1:]]

    df = df.set_index('index')

    for col in {
        'cat1', 'cat2', 'ca', 'death', 'sex', 'dth30', 'swang1', 'dnr1',
        'ninsclas', 'resp', 'card', 'neuro', 'gastr', 'renal', 'meta',
        'hema', 'seps', 'trauma', 'ortho', 'race', 'income',
    }:
        df[col] = df[col].astype('category')

    meta = Object(
        outcome='death',
        treatment='swang1',
    )

    return df, meta


def ensure_rhc_data_exists(fname):
    if os.path.exists(fname):
        return

    print('download rhc dataset', file=sys.stderr)
    r = requests.get('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/rhc.csv', stream=True)
    r.raise_for_status()

    with bz2.open(fname, 'wt') as fobj:
        for line in r.iter_lines(decode_unicode=True):
            fobj.write(line)

            if not line.endswith('\n'):
                fobj.write('\n')


def normalize_rhc_df(df, meta):
    # Follow preprocessing of "Sample R code for matching"
    # https://www.coursera.org/learn/crash-course-in-causality/resources/0kbXw

    df = pd.DataFrame({
        'ARF': (df['cat1'] == 'ARF').astype(float),
        'CHF': (df['cat1'] == 'CHF').astype(float),
        'Cirr': (df['cat1'] == 'Cirrhosis').astype(float),
        'colcan': (df['cat1'] == 'Colon Cancer').astype(float),
        'Coma': (df['cat1'] == 'Coma').astype(float),
        'COPD': (df['cat1'] == 'COPD').astype(float),
        'lungcan': (df['cat1'] == 'Lung Cancer').astype(float),
        'MOSF': (df['cat1'] == 'MOSF w/Malignancy').astype(float),
        'sepsis': (df['cat1'] == 'MOSF w/Sepsis').astype(float),
        'female': (df['sex'] == 'Female').astype(float),
        'died': (df[meta.outcome] == 'Yes').astype(int),
        'treatment': (df[meta.treatment] == 'RHC').astype(int),
        'age': df['age'],
        'meanbp1': df['meanbp1'],
    })
    meta = Object(
        outcome='died',
        treatment='treatment',
        covariates=sorted({
            "ARF", "CHF", "Cirr", "colcan", "Coma", "lungcan",
            "MOSF", "sepsis", "age", "female", "meanbp1"
        })
    )

    return df, meta


def read_proximity_df(fname):
    """Read the CARD dataset

    In R download and convert it via:

        install.packages('ivpack')
        library('ivpack')
        write.csv(card.data, file='card.csv', na='')

    """
    df = pd.read_csv(fname)
    df.columns = ['csv_index', *df.columns[1:]]
    df = df.drop(columns=['csv_index']).set_index('id')
    df['educ12'] = (df['educ'] > 12).astype('int')

    meta = Object(
        instrument='nearc4',
        treatment='educ12',
        outcome='lwage',
    )

    return df, meta