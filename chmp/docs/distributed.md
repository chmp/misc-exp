## `chmp.distributed`

Helpers for distributed data processing.

Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.


### `chmp.distributed.DaskCounter`
`chmp.distributed.DaskCounter()`

[undocumented]


### `chmp.distributed.DaskDict`
`chmp.distributed.DaskDict()`

[undocumented]


### `chmp.distributed.RuleSet`
`chmp.distributed.RuleSet(rules, rules=())`

[undocumented]


### `chmp.distributed.apply`
`chmp.distributed.apply()`

Apply the given transformation to a `dask.bag.Bag`.


### `chmp.distributed.chained`
`chmp.distributed.chained(*funcs)`

Represent the composition of functions.

When the resulting object is called with a single argument, the passed
object is transformed by passing it through all given functions.
For example:

```
a = chained(
    math.sqrt,
    math.log,
    math.cos,
)(5.0)
```

is equivalent to:

```
a = 5.0
a = math.sqrt(a)
a = math.log(a)
a = math.cos(a)
```


### `chmp.distributed.extension_rules`

[undocumented]


### `chmp.distributed.mean`
`chmp.distributed.mean()`

Calculate the mean of a list of values.


### `chmp.distributed.repartition`
`chmp.distributed.repartition()`

Express repartition of a `dask.bag.Bag`, for non bags it is a nop-op.

#### Parameters

* **n** (*int*):
  the number of partitions


### `chmp.distributed.rules`

[undocumented]


### `chmp.distributed.reduction_rules`

[undocumented]


### `chmp.distributed.std`
`chmp.distributed.std(*args, **kwargs)`

[undocumented]


### `chmp.distributed.stdlib_rules`

[undocumented]


### `chmp.distributed.toolz_rules`

[undocumented]


### `chmp.distributed.var`
`chmp.distributed.var(*args, **kwargs)`

[undocumented]

