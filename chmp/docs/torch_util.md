## `chmp.torch_util`

Helper to construct models with pytorch.


### `chmp.torch_util.TorchModel`
`chmp.torch_util.TorchModel(module, optimizer='Adam', loss=None, regularization=None, optimizer_kwargs=None)`

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


#### `chmp.torch_util.TorchModel.fit`
`chmp.torch_util.TorchModel.fit(x=None, y=None, *, batch_size=None, epochs=1, shuffle=True, verbose=False, callbacks=None, dtype='float32')`


#### `chmp.torch_util.TorchModel.fit_generator`
`chmp.torch_util.TorchModel.fit_generator(generator=None, *, steps_per_epoch=1, epochs=None, verbose=True, callbacks=None, dtype='float32')`

Fit the model on a dynamically generated dataset.

#### Parameters

* **generator** (*any*):
  A generator yielding `batch_x, batch_y` pairs.
* **steps_per_epoch** (*any*):
  The number batches that make up an epoch.
* **epochs** (*any*):
  The number of epochs to evaluate. If `None`, the generator must be
  finite.
* **verbose** (*any*):


#### Returns

itself.


#### `chmp.torch_util.TorchModel.predict`
`chmp.torch_util.TorchModel.predict(x=None, *, batch_size=None, verbose=False, dtype='float32')`


#### `chmp.torch_util.TorchModel.predict_generator`
`chmp.torch_util.TorchModel.predict_generator(generator, *, steps=None, verbose=True, dtype='float32', strip_target=False)`

Predict on a generator.

#### Parameters

* **generator** (*any*):
  an iterable, that will be used to get batches for prediction.
* **steps** (*any*):
  the number of times the generator should be called. if `steps` is
  `None`, the generator all items of the generator will be
  processed. Therefore, the generator should only yield a finite
  number of items in this case.
* **verbose** (*any*):
  if `True` print progress during prediction.
* **strip_target** (*any*):
  if `True`, the generator is assumed to also yield targets, that
  should be ignored. Note: in this case also the dtype is assumed to
  include target information.

#### Returns

the predictions as a numpy array.


### `chmp.torch_util.Callback`
`chmp.torch_util.Callback()`

Event handler to monitor / modify training runs.

Most event handlers have a `begin_*`, `end_*` structure, with a
`logs` argument. For each `end_*` call, the same dictionary as for the
`begin_*` call is passed. This mechanism allows to modify the object to
collect statistics.


### `chmp.torch_util.LearningRateScheduler`
`chmp.torch_util.LearningRateScheduler(cls, **kwargs)`


### `chmp.torch_util.History`
`chmp.torch_util.History()`

Record any epoch statistics generated during training.


### `chmp.torch_util.LossHistory`
`chmp.torch_util.LossHistory()`

Record the loss history per batch.


### `chmp.torch_util.TerminateOnNaN`
`chmp.torch_util.TerminateOnNaN()`

Raise an exception when the loss becomes nan.


### `chmp.torch_util.Transformer`
`chmp.torch_util.Transformer(key_module, query_module=None, value_module=None, flatten=False)`

A attention / transformer model.

Masks be two-dimensional and compatible with `n_query, n_search`. This
model also supports soft-masks. They must never be `0`. The hard masks
must be binary `{0, 1}`.


#### `chmp.torch_util.Transformer.compute_weights`
`chmp.torch_util.Transformer.compute_weights(search_x, query_x, mask, soft_mask=None)`

Compute weights with shape `(batch_size, n_samples, n_keys)`.


### `chmp.torch_util.iter_batch_indices`
`chmp.torch_util.iter_batch_indices(n_samples, *, batch_size=None, indices=None, shuffle=False, only_complete=True)`


### `chmp.torch_util.iter_batched`
`chmp.torch_util.iter_batched(data, *, batch_size=None, indices=None, only_complete=True, shuffle=False)`

Iterate over data in batches.

#### Parameters

* **data** (*any*):
  the data to iterate over has to be a dicitonary
* **only_complete** (*any*):
  if True yield only batches that have exactly `batch_size` items


### `chmp.torch_util.factorized_quadratic`
`chmp.torch_util.factorized_quadratic(x, weights)`

A factorized quadratic interaction.

#### Parameters

* **x** (*any*):
  shape `(batch_size, in_features)`
* **weights** (*any*):
  shape `(n_factors, in_features, out_features)`


### `chmp.torch_util.identity`
`chmp.torch_util.identity(x)`


### `chmp.torch_util.linear`
`chmp.torch_util.linear(x, weights)`

A linear interaction.

#### Parameters

* **x** (*any*):
  shape `(batch_size, in_features)`
* **weights** (*any*):
  shape `(n_factors, in_features, out_features)`


### `chmp.torch_util.masked_softmax`
`chmp.torch_util.masked_softmax(logits, mask, eps=1e-06, dim=-1)`

Compute a softmax with certain elements masked out.


### `chmp.torch_util.Add`
`chmp.torch_util.Add(*children)`

Appply all modules in parallel and add their outputs.


### `chmp.torch_util.DiagonalScaleShift`
`chmp.torch_util.DiagonalScaleShift(shift=None, scale=None)`

Scale and shift the inputs along each dimension independently.


### `chmp.torch_util.Flatten`
`chmp.torch_util.Flatten()`


### `chmp.torch_util.Lambda`
`chmp.torch_util.Lambda(func)`


### `chmp.torch_util.find_module`
`chmp.torch_util.find_module(root, predicate)`

Find a (sub) module using a predicate.

#### Parameters

* **predicate** (*any*):
  a callable with arguments `(name, module)`.

#### Returns

the first module for which the predicate is true or raises
a `RuntimeError`.


### `chmp.torch_util.ExponentialModule`
`chmp.torch_util.ExponentialModule(*args, **kwargs)`


### `chmp.torch_util.LogNormalModule`
`chmp.torch_util.LogNormalModule(*args, **kwargs)`


### `chmp.torch_util.GammaModule`
`chmp.torch_util.GammaModule(*args, **kwargs)`


### `chmp.torch_util.KLDivergence`
`chmp.torch_util.KLDivergence(n_observations)`

A regularizer using the KL divergence of the model.


### `chmp.torch_util.NllLoss`
`chmp.torch_util.NllLoss(distribution)`

Negative log likelihood loss for pytorch distributions.

Usage:

```
loss = NllLoss(torch.distribtuions.Normal)
loc, scale = parameter_module(x)
loss((loc, scale), y)
```


### `chmp.torch_util.NormalModelConstantScale`
`chmp.torch_util.NormalModelConstantScale(transform=None, scale=1.0)`


### `chmp.torch_util.NormalModule`
`chmp.torch_util.NormalModule(*args, **kwargs)`


### `chmp.torch_util.SimpleBayesTorchModel`
`chmp.torch_util.SimpleBayesTorchModel(module, n_observations, **kwargs)`


#### `chmp.torch_util.SimpleBayesTorchModel.fit`
`chmp.torch_util.SimpleBayesTorchModel.fit(x=None, y=None, *, batch_size=None, epochs=1, shuffle=True, verbose=False, callbacks=None, dtype='float32')`


#### `chmp.torch_util.SimpleBayesTorchModel.fit_generator`
`chmp.torch_util.SimpleBayesTorchModel.fit_generator(generator=None, *, steps_per_epoch=1, epochs=None, verbose=True, callbacks=None, dtype='float32')`

Fit the model on a dynamically generated dataset.

#### Parameters

* **generator** (*any*):
  A generator yielding `batch_x, batch_y` pairs.
* **steps_per_epoch** (*any*):
  The number batches that make up an epoch.
* **epochs** (*any*):
  The number of epochs to evaluate. If `None`, the generator must be
  finite.
* **verbose** (*any*):


#### Returns

itself.


#### `chmp.torch_util.SimpleBayesTorchModel.predict`
`chmp.torch_util.SimpleBayesTorchModel.predict(x=None, *, batch_size=None, verbose=False, dtype='float32')`


#### `chmp.torch_util.SimpleBayesTorchModel.predict_generator`
`chmp.torch_util.SimpleBayesTorchModel.predict_generator(generator, *, steps=None, verbose=True, dtype='float32', strip_target=False)`

Predict on a generator.

#### Parameters

* **generator** (*any*):
  an iterable, that will be used to get batches for prediction.
* **steps** (*any*):
  the number of times the generator should be called. if `steps` is
  `None`, the generator all items of the generator will be
  processed. Therefore, the generator should only yield a finite
  number of items in this case.
* **verbose** (*any*):
  if `True` print progress during prediction.
* **strip_target** (*any*):
  if `True`, the generator is assumed to also yield targets, that
  should be ignored. Note: in this case also the dtype is assumed to
  include target information.

#### Returns

the predictions as a numpy array.


### `chmp.torch_util.WeightsHS`
`chmp.torch_util.WeightsHS(shape, tau_0, regularization=None)`

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


### `chmp.torch_util.optional_parameter`
`chmp.torch_util.optional_parameter(arg, *, default=<class 'chmp.torch_util._probabilistic.optimized'>)`

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


### `chmp.torch_util.fixed`
`chmp.torch_util.fixed(value)`

decorator to mark a parameter as not-optimized.


### `chmp.torch_util.optimized`
`chmp.torch_util.optimized(value)`

Decorator to mark a parameter as optimized.

