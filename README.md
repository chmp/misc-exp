# Various Experiments, possibly ML related

## Causality and function approximations

[blog post](https://cprohm.de/article/causality-and-function-approximations.html) |
[source](./20180107-Causality)

How do causal analysis and function approximations interact? This blog post 
demonstrates how results for tabular methods break down for models with finite
capacity.

## Keyword detection from scratch

[blog post](http://cprohm.de/article/keyword-detection-from-scratch.html) |
[source](./20170813-KeywordDetection)

Detecting keywords in speech from data-collection to modelling, includes code
to listen continuously for commands.

## Support

- [chmp](./chmp): support code as a python package

## Getting started

This project uses [pipenv](https://docs.pipenv.org/) to organize dependencies
and common scripts. To setup a virtual environment with all requirements use:

    pipenv sync --dev

After that the following tasks can be performed:
    
    # run any pre-commit tasks (docs, formatting, tests)
    pipenv run precommit
    
    # run tests
    pipenv run test

    # update the documentation
    pipenv run docs
