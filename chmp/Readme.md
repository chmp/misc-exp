# Collection of utility functions

- [`chmp.distributed`](docs/distributed.md): distributed execution of
  functional pipelines
- [`chmp.ds`](docs/ds.md): data science support
- [`chmp.label`](docs/label.md): support for labeling in IPython notebooks
- [`chmp.ml`](src/chmp/ml): machine learning helpers
- [`chmp.parser`](docs/parser.md): helpers to write parsers using functional
  composition

Note, the individual modules are designed ot be easily copyable outside this
distribution. For example to use the parser combinators just copy the
`__init__.py` into the local project.

## Installation

```bash
# use quotation marks prevent bash from interpreting # / &
pip install -e 'git+https://https://github.com/chmp/misc-exp/#egg=chmp&subdirectory=chmp'
```
