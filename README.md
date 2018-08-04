# Various Experiments, possibly ML related

## Keyword detection from scratch

[blog post](http://cprohm.de/article/keyword-detection-from-scratch.html) |
[source](./KeywordDetection)

Detecting keywords in speech from data-collection to modelling, includes code
to listen continuously for commands.


## Support

- [chmp](./chmp): support code as a python package
- [bin](./bin): executable helpers

## Getting started

This project uses [pipenv](https://docs.pipenv.org/) to organize dependencies
and common scripts. To setup a virtual environment with all requirements use:

    pipenv sync --dev

After that the following tasks can be performed:

    # run tests
    pipenv run test

    # update the documentation
    pipenv run docs

