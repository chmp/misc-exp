# Collection of utility functions

- [`chmp.distributed`](src/chmp/distributed): distributed execution of
  functional pipelines
- [`chmp.ds`](docs/ds.md): data science support
- [`chmp.experiment`](src/chmp/experiment): support for ML experiments
- [`chmp.label`](docs/label.md): support for labeling in IPython notebooks
- [`chmp.ml`](src/chmp/ml): machine learning helpers
- [`chmp.parser`](src/chmp/parser): helpers to write parsers using functional
  composition
- [`chmp.reactive`](src/chmp/reactive): prototype of reactive IPython notebooks

Note, the individual modules are designed ot be easily copyable outside this
distribution. For example to use the parser combinators just copy the
`__init__.py` into the local project.

## Installation

```bash
# use quotation marks prevent bash from interpreting # / &
pip install -e 'git+https://https://github.com/chmp/misc-exp/#egg=chmp&subdirectory=chmp'
```
