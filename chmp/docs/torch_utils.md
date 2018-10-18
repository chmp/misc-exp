## `chmp.torch_utils.data`


### `chmp.torch_utils.data.data_loader`
`chmp.torch_utils.data.data_loader(cls, *args, batch_size=32, mode='predict', collate_fn=None, num_workers=None, pin_memory=None, worker_init_fn=None, **kwargs)`

Helper to build data loaders for numpy / transform based datasets.

#### Parameters

* **cls** (*any*):
  either a a dataset class or one of `'numpy'` (for
  [NumpyDataset](#numpydataset)) or `'transformed'` (for
  [TransformedDataset](#transformeddataset)).
* **args** (*any*):
  varargs passed to cls to build the dataset
* **kwargs** (*any*):
  kwargs passed to cls to build the dataset
* **mode** (*any*):
  if `'fit'`, shuffle the dataset and only yield batches with
  `batch_size` samples. If `'predict'`, always yield all samples in
  order.

All other arguments (if given) are passed to
[torch.utils.data.DataLoader](https://pytorch.org/docs/stable//data.html#torch.utils.data.DataLoader).


### `chmp.torch_utils.data.pad_sequences`
`chmp.torch_utils.data.pad_sequences(*sequence_batches, dtype='float32', length=None, length_dtype='int64', factory=None)`

Helper to build pad a batches of sequences.


### `chmp.torch_utils.data.NumpyDataset`
`chmp.torch_utils.data.NumpyDataset(data, dtype='float32')`

Helper to build a dataset from numpy arrays.


### `chmp.torch_utils.data.TransformedDataset`
`chmp.torch_utils.data.TransformedDataset(transform, *bases, dtype='float32')`

Helper to build a dataset from transformed items of a base dataset.


### `chmp.torch_utils.data.apply_dtype`
`chmp.torch_utils.data.apply_dtype(dtype, arg)`


### `chmp.torch_utils.data.pack`
`chmp.torch_utils.data.pack(obj)`

Pack arguments of different types into a list.

The `pack` / `unpack` pair is used to ensure that even with mixed
arguments internal the logic always sees a list of arrays.

#### Returns

a tuple of `keys, values`. The `keys` are an opaque hashable
object that can be used to unpack the values. `values` will be
a tuple of flattend arguments.


### `chmp.torch_utils.data.unpack`
`chmp.torch_utils.data.unpack(key, values)`

Unpack previously packed parameters.

Given `keys` and `values` as returned by `pack` reconstruct
build objects of the same form as the arguments to `pack`.

#### Returns

a tuple of the same structure as the arguments to `pack`.



## `chmp.torch_utils.model`


### `chmp.torch_utils.model.Model`
`chmp.torch_utils.model.Model(module, optimizer='Adam', loss=None, regularization=None, optimizer_kwargs=None)`

Keras-like API around a torch models.

#### Parameters

* **module** (*any*):
  the module that defines the model prediction
* **optimizer** (*any*):
  the optimizer to use. Either a callable or string specifying an
  optimizer in torch.optim.
* **optimizer_kwargs** (*any*):
  keyword arguments passed to the optimizer before building it.
* **loss** (*any*):
  the `loss` function to use, with signature
  `(pred, target) -> loss`. If `None`, the module is assumed to
  return the loss itself.
* **regularization** (*any*):
  if given a callable, with signature `(module) -> loss`, that should
  return a regularization loss

For all functions `x` and `y` can not only be `numpy` arrays, but
also structured data, such as dicts or lists / tuples. The former are
passed to the module as keyword arguments, the latter as varargs.

For example:

```
# NOTE: this module does not define parameters
class Model(torch.nn.Module):
    def forward(self, a, b):
        return a + b


model = TorchModel(module=Model, loss=MSELoss())
model.fit(x={"a": [...], "b": [...]}, y=[...])
```


#### `chmp.torch_utils.model.Model.fit`
`chmp.torch_utils.model.Model.fit(x, y, *, epochs=1, batch_size=32, dtype='float32', verbose=True, callbacks=None, metrics=None, validation_data=None)`


#### `chmp.torch_utils.model.Model.predict`
`chmp.torch_utils.model.Model.predict(x, batch_size=32, dtype='float32', verbose=False)`


#### `chmp.torch_utils.model.Model.fit_transformed`
`chmp.torch_utils.model.Model.fit_transformed(transform, *bases, epochs=1, batch_size=32, dtype='float32', verbose=True, callbacks=None, metrics=None, validation_data=None)`


#### `chmp.torch_utils.model.Model.predict_transformed`
`chmp.torch_utils.model.Model.predict_transformed(transform, *bases, batch_size=32, dtype='float32', verbose=False)`


#### `chmp.torch_utils.model.Model.fit_data`
`chmp.torch_utils.model.Model.fit_data(data, *, epochs=1, callbacks=None, verbose=True, metrics=None, validation_data=None)`


#### `chmp.torch_utils.model.Model.predict_data`
`chmp.torch_utils.model.Model.predict_data(data, *, verbose=False)`


### `chmp.torch_utils.model.Callback`
`chmp.torch_utils.model.Callback()`

Event handler to monitor / modify training runs.

Most event handlers have a `begin_*`, `end_*` structure, with a
`logs` argument. For each `end_*` call, the same dictionary as for the
`begin_*` call is passed. This mechanism allows to modify the object to
collect statistics.


### `chmp.torch_utils.model.History`
`chmp.torch_utils.model.History()`

Record any epoch statistics generated during training.


### `chmp.torch_utils.model.LossHistory`
`chmp.torch_utils.model.LossHistory()`

Record the loss history per batch.


### `chmp.torch_utils.model.LearningRateScheduler`
`chmp.torch_utils.model.LearningRateScheduler(cls, **kwargs)`



## `chmp.torch_utils.attention`


### `chmp.torch_utils.attention.Transformer`
`chmp.torch_utils.attention.Transformer(key_module, query_module=None, value_module=None, flatten=False, search_x=None, search_y=None)`

A attention / transformer model.

Masks be two-dimensional and compatible with `n_query, n_search`. This
model also supports soft-masks. They must never be `0`. The hard masks
must be binary `{0, 1}`.


#### `chmp.torch_utils.attention.Transformer.compute_weights`
`chmp.torch_utils.attention.Transformer.compute_weights(search_x, query_x, mask, soft_mask=None)`

Compute weights with shape `(batch_size, n_samples, n_keys)`.



## `chmp.torch_utils.bayes`


### `chmp.torch_utils.bayes.fixed`
`chmp.torch_utils.bayes.fixed(value)`

decorator to mark a parameter as not-optimized.


### `chmp.torch_utils.bayes.optimized`
`chmp.torch_utils.bayes.optimized(value)`

Decorator to mark a parameter as optimized.


### `chmp.torch_utils.bayes.optional_parameter`
`chmp.torch_utils.bayes.optional_parameter(arg, *, default=<class 'chmp.torch_utils.bayes.optimized'>)`

Make sure arg is a tensor and optionally a parameter.

Values wrapped with `fixed` are returned as a tensor, `values` wrapped
with `optimized``are returned as parameters. When arg is not one of
``fixed` or `optimized` it is wrapped with `default`.

Usage:

```
class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()

        # per default a will be optimized during training
        self.a = optional_parameter(a, default=optimized)

        # per default B will not be optimized during training
        self.b = optional_parameter(b, default=fixed)
```


### `chmp.torch_utils.bayes.KLDivergence`
`chmp.torch_utils.bayes.KLDivergence(n_observations)`

A regularizer using the KL divergence of the model.


### `chmp.torch_utils.bayes.SimpleBayesModel`
`chmp.torch_utils.bayes.SimpleBayesModel(module, n_observations, **kwargs)`


#### `chmp.torch_utils.bayes.SimpleBayesModel.fit`
`chmp.torch_utils.bayes.SimpleBayesModel.fit(x, y, *, epochs=1, batch_size=32, dtype='float32', verbose=True, callbacks=None, metrics=None, validation_data=None)`


#### `chmp.torch_utils.bayes.SimpleBayesModel.predict`
`chmp.torch_utils.bayes.SimpleBayesModel.predict(x, batch_size=32, dtype='float32', verbose=False)`


#### `chmp.torch_utils.bayes.SimpleBayesModel.fit_transformed`
`chmp.torch_utils.bayes.SimpleBayesModel.fit_transformed(transform, *bases, epochs=1, batch_size=32, dtype='float32', verbose=True, callbacks=None, metrics=None, validation_data=None)`


#### `chmp.torch_utils.bayes.SimpleBayesModel.predict_transformed`
`chmp.torch_utils.bayes.SimpleBayesModel.predict_transformed(transform, *bases, batch_size=32, dtype='float32', verbose=False)`


#### `chmp.torch_utils.bayes.SimpleBayesModel.fit_data`
`chmp.torch_utils.bayes.SimpleBayesModel.fit_data(data, *, epochs=1, callbacks=None, verbose=True, metrics=None, validation_data=None)`


#### `chmp.torch_utils.bayes.SimpleBayesModel.predict_data`
`chmp.torch_utils.bayes.SimpleBayesModel.predict_data(data, *, verbose=False)`


### `chmp.torch_utils.bayes.VariationalNormal`
`chmp.torch_utils.bayes.VariationalNormal(shape, loc, scale)`

Variational approximation to a Normal distributed sample.


### `chmp.torch_utils.bayes.VariationalHalfCauchy`
`chmp.torch_utils.bayes.VariationalHalfCauchy(shape, tau)`

Variational approximation to Half-Cauchy distributed sample.


### `chmp.torch_utils.bayes.NormalModule`
`chmp.torch_utils.bayes.NormalModule(*args, **kwargs)`


### `chmp.torch_utils.bayes.GammaModule`
`chmp.torch_utils.bayes.GammaModule(*args, **kwargs)`


### `chmp.torch_utils.bayes.LogNormalModule`
`chmp.torch_utils.bayes.LogNormalModule(*args, **kwargs)`


### `chmp.torch_utils.bayes.ExponentialModule`
`chmp.torch_utils.bayes.ExponentialModule(*args, **kwargs)`


### `chmp.torch_utils.bayes.NormalModelConstantScale`
`chmp.torch_utils.bayes.NormalModelConstantScale(transform=None, scale=1.0)`


### `chmp.torch_utils.bayes.WeightsHS`
`chmp.torch_utils.bayes.WeightsHS(shape, tau_0, regularization=None)`

A module that generates weights with a Horeshoe Prior.

#### Parameters

* **shape** (*any*):
  the shape of sample to generate
* **tau_0** (*any*):
  the scale of the the global scale prior. Per default, this parameter
  is not optimized. Pass as `optimized(inital_tau_0)` to fit the
  parameter with maximum likelihood.
* **regularization** (*any*):
  if given, the regularization strength.

To implement a linear regression model with Horseshoe prior, use:

```
class LinearHS(NormalModelConstantScale):
    def __init__(self, in_features, out_features, tau_0, bias=True):
        super().__init__()

        self.weights = WeightsHS((in_features, out_features), tau_0=tau_0)
        self.bias = torch.nn.Parameter(torch.zeros(1)) if bias else 0

    def transform(self, x):
        return self.bias + linear(x, self.weights())

    def kl_divergence(self):
        return self.weights.kl_divergence()
```

Sources:

- * **The basic implementation (incl. the posterior approximation) is taken**:
  from C. Louizos, K. Ullrich, and M. Welling " Bayesian Compression for
  Deep Learning" (2017).
- * **The regularization concept is taken from J. Piironen and A. Vehtari**:
  "Sparsity information and regularization in the horseshoe and other
  shrinkage priors" (2107).

