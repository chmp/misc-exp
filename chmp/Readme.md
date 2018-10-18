# chmp - Support code for machine learning / data science experiments

- [`chmp.distributed`](docs/distributed.md): distributed execution of
  functional pipelines
- [`chmp.ds`](docs/ds.md): data science support
- [`chmp.experiment`](docs/experiment.md): support code to organize and analyse
  machine learning experiments
- [`chmp.label`](docs/label.md): support for labeling in IPython notebooks
- [`chmp.parser`](docs/parser.md): helpers to write parsers using functional
  composition
- [`chmp.torch_util`](docs/torch_utils.md): helpers to write pytorch models

The individual modules are designed ot be easily copyable outside this
distribution. For example to use the parser combinators just copy the
`__init__.py` into the local project.


To install / run tests use:

```bash
# install the package
pip install chmp

# to run tests
pip install pytest
pytest --pyargs chmp
```
