## `chmp.torch_util`

Helper to construct models with pytorch.


### `chmp.torch_util.Transformer`
`chmp.torch_util.Transformer(key_module=None, query_module=None, value_module=<function noop_value_module at 0x11bd0d9d8>, flatten=False)`


#### `chmp.torch_util.Transformer.compute_weights`
`chmp.torch_util.Transformer.compute_weights(search_x, query_x, mask)`

Compute weights with shape `(batch_size, n_samples, n_keys)`.


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


### `chmp.torch_util.TorchModel`
`chmp.torch_util.TorchModel(module, optimizer, loss, regularization=None)`

Helper to add simple numpy integration to torch model.

#### Parameters

* **module** (*any*):
  the module that defines the model prediction
* **optimizer** (*any*):
  a factory for the optimizer to use
* **loss** (*any*):
  the loss function to use, with signature `(pred, target) -> loss`
* **regularization** (*any*):
  if given a callable, with signature `(module) -> loss`, that should
  return a regularization loss


#### `chmp.torch_util.TorchModel.fit_generator`
`chmp.torch_util.TorchModel.fit_generator(generator)`

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


#### `chmp.torch_util.TorchModel.predict_generator`
`chmp.torch_util.TorchModel.predict_generator(generator)`

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


#### Returns

the predictions as a numpy array.


### `chmp.torch_util.iter_batch_indices`
`chmp.torch_util.iter_batch_indices(n_samples)`


### `chmp.torch_util.iter_batched`
`chmp.torch_util.iter_batched(data)`

Iterate over data in batches.

#### Parameters

* **data** (*any*):
  the data to iterate over has to be a dicitonary
* **only_complete** (*any*):
  if True yield only batches that have exactly `batch_size` items


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


### `chmp.torch_util.NormalModelConstantScale`
`chmp.torch_util.NormalModelConstantScale(transform)`


### `chmp.torch_util.NormalModule`
`chmp.torch_util.NormalModule(*args, **kwargs)`


### `chmp.torch_util.WeightsHS`
`chmp.torch_util.WeightsHS(shape, tau_0)`

A module that generates weights with a Horeshoe Prior.


### `chmp.torch_util.fixed`
`chmp.torch_util.fixed(value)`

decorator to mark a parameter as not-optimized

