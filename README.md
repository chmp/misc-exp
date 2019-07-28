# Various Experiments, possibly ML related

[asyncio testing in notebooks](#asyncio-testing-in-notebooks)
| [Prototyping to tested code](#prototyping-to-tested-code)
| [Causality and function approximations](#causality-and-function-approximations)
| [Keyword detection from scratch](#keyword-detection-from-scratch)

[Support](#support)
| [Getting started](#getting-started)
| [View on nbviewer](https://nbviewer.jupyter.org/github/chmp/misc-exp/tree/master/)

## asyncio testing in notebooks

[blog post](https://cprohm.de/article/asyncio-testing-inside-notebooks.html)
[source](./004-AsyncTestingInNotebooks)

How can you test asyncio code inside notebooks? This blog post sketches how
to test asyncio code using pytest inside nobteooks. The post also discusses
how threading can help to run multiple asyncio event loops inside the same
interpreter.

## Prototyping to tested code

[slides](https://htmlpreview.github.io/?https://github.com/chmp/misc-exp/blob/master/20181026-TestingInJupyter/resources/IPyTestIntro.slides.html#/) |
[source](./20181026-TestingInJupyter)

How can pytest be used in Jupyter notebooks? And why does it make sense? This
talk discusses how Jupyter notebooks form an effective environment for
prototyping and how code can be refactored code into modules. A particular
emphasis is placed on testing and the use of
[ipytest](https://github.com/chmp/ipytest).

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

    # run all pre-commit tasks (docs, formatting, tests)
    pipenv run precommit

    # run pre-commit tasks and integration tests
    pipenv run precommit-full

    # run notebook integration tests
    pipenv run integration

    # run tests
    pipenv run test

    # update the documentation
    pipenv run docs
