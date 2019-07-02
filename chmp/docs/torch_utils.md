# `chmp.torch_utils`

## `chmp.torch_utils`

Helper to construct models with pytorch.


### `chmp.torch_utils.fixed`
`chmp.torch_utils.fixed(value)`

decorator to mark a parameter as not-optimized.


### `chmp.torch_utils.optimized`
`chmp.torch_utils.optimized(value)`

Decorator to mark a parameter as optimized.


### `chmp.torch_utils.optional_parameter`
`chmp.torch_utils.optional_parameter(arg, *, default=<class 'chmp.torch_utils.optimized'>)`

Make sure arg is a tensor and optionally a parameter.

Values wrapped with `fixed` are returned as a tensor, `values` wrapped
with `optimized``are returned as parameters. When arg is not one of
``fixed` or `optimized` it is wrapped with `default`.

Usage:

```python
class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()

        # per default a will be optimized during training
        self.a = optional_parameter(a, default=optimized)

        # per default B will not be optimized during training
        self.b = optional_parameter(b, default=fixed)
```


### `chmp.torch_utils.t2n`
`chmp.torch_utils.t2n(obj, dtype=None)`

Torch to numpy.


### `chmp.torch_utils.n2t`
`chmp.torch_utils.n2t(obj, dtype=None, device=None)`

Numpy to torch.


### `chmp.torch_utils.call_torch`
`chmp.torch_utils.call_torch(func, arg, *args, dtype=None, device=None, batch_size=64)`

Call a torch function with numpy arguments and numpy results.


### `chmp.torch_utils.linear`
`chmp.torch_utils.linear(x, weights)`

A linear interaction.

#### Parameters

* **x** (*any*):
  shape `(batch_size, in_features)`
* **weights** (*any*):
  shape `(n_factors, in_features, out_features)`


### `chmp.torch_utils.factorized_quadratic`
`chmp.torch_utils.factorized_quadratic(x, weights)`

A factorized quadratic interaction.

#### Parameters

* **x** (*any*):
  shape `(batch_size, in_features)`
* **weights** (*any*):
  shape `(n_factors, in_features, out_features)`


### `chmp.torch_utils.masked_softmax`
`chmp.torch_utils.masked_softmax(logits, mask, eps=1e-06, dim=-1)`

Compute a softmax with certain elements masked out.


### `chmp.torch_utils.find_module`
`chmp.torch_utils.find_module(root, predicate)`

Find a (sub) module using a predicate.

#### Parameters

* **predicate** (*any*):
  a callable with arguments `(name, module)`.

#### Returns

the first module for which the predicate is true or raises
a `RuntimeError`.


### `chmp.torch_utils.DiagonalScaleShift`
`chmp.torch_utils.DiagonalScaleShift(shift=None, scale=None)`

Scale and shift the inputs along each dimension independently.


### `chmp.torch_utils.Add`
`chmp.torch_utils.Add(*children)`

Apply all modules in parallel and add their outputs.


### `chmp.torch_utils.Do`
`chmp.torch_utils.Do(func, **kwargs)`

Call a function as a pure side-effect.


### `chmp.torch_utils.LookupFunction`
`chmp.torch_utils.LookupFunction(input_min, input_max, forward_values, backward_values)`

Helper to define a lookup function incl. its gradient.

Usage:

```python
import scipy.special

x = np.linspace(0, 10, 100).astype('float32')
iv0 = scipy.special.iv(0, x).astype('float32')
iv1 = scipy.special.iv(1, x).astype('float32')

iv = LookupFunction(x.min(), x.max(), iv0, iv1)

a = torch.linspace(0, 20, 200, requires_grad=True)
g, = torch.autograd.grad(iv(a), a, torch.ones_like(a))
```


### `chmp.torch_utils.Transformer`
`chmp.torch_utils.Transformer(key_module, query_module=None, value_module=None, flatten=False, search_x=None, search_y=None)`

A attention / transformer Module.

Masks be two-dimensional and compatible with `n_query, n_search`. This
model also supports soft-masks. They must never be `0`. The hard masks
must be binary `{0, 1}`.


#### `chmp.torch_utils.Transformer.compute_weights`
`chmp.torch_utils.Transformer.compute_weights(search_x, query_x, mask, soft_mask=None)`

Compute weights with shape `(batch_size, n_samples, n_keys)`.


### `chmp.torch_utils.kl_divergence__gamma__log_normal`
`chmp.torch_utils.kl_divergence__gamma__log_normal(p, q)`

Compute the kl divergence with a Gamma prior and LogNormal approximation.

Taken from C. Louizos, K. Ullrich, M. Welling "Bayesian Compression for Deep Learning"
https://arxiv.org/abs/1705.08665

