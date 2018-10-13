## `chmp.torch_util`

Helper to construct models with pytorch.


### `chmp.torch_util.Transformer`
`chmp.torch_util.Transformer(key_module=None, query_module=None, value_module=<function noop_value_module at 0x1115829d8>, flatten=False)`

A attention / transformer model.

Note: this model also supports soft-masks. They must never be `0`. The
hard masks must be binary `{0, 1}`.

Masks be two-dimensional and compatible with `n_query, n_search`.


#### `chmp.torch_util.Transformer.compute_weights`
`chmp.torch_util.Transformer.compute_weights(search_x, query_x, mask, soft_mask=None)`

Compute weights with shape `(batch_size, n_samples, n_keys)`.


### `chmp.torch_util.Callback`
`chmp.torch_util.Callback()`


### `chmp.torch_util.LossHistory`
`chmp.torch_util.LossHistory()`


### `chmp.torch_util.TerminateOnNaN`
`chmp.torch_util.TerminateOnNaN()`


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


### `chmp.torch_util.LearningRateScheduler`
`chmp.torch_util.LearningRateScheduler(cls, **kwargs)`


### `chmp.torch_util.TorchModel`
`chmp.torch_util.TorchModel(module, optimizer='Adam', loss=None, regularization=None, optimizer_kwargs=None)`


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


### `chmp.torch_util.fixed`
`chmp.torch_util.fixed(value)`

decorator to mark a parameter as not-optimized

