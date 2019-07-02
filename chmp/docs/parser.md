## `chmp.parser`

Build custom parsers fast using parser combinators.

Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.


### `chmp.parser.inspect_parser`
`chmp.parser.inspect_parser(parser)`

Recursively inspect a parser.


### `chmp.parser.sequential`
`chmp.parser.sequential(*args, **kwargs)`

Match a sequence of parsers exactly.


### `chmp.parser.first`
`chmp.parser.first(*args, **kwargs)`

Return the result of the first parser to match.


### `chmp.parser.repeat`
`chmp.parser.repeat(*args, **kwargs)`

Match 0 or more occurrences.

If multiple parsers are given, they are combined with `sequential`.


### `chmp.parser.ignore`
`chmp.parser.ignore(*args, **kwargs)`

Ignore the result of parser.


### `chmp.parser.optional`
`chmp.parser.optional(*args, **kwargs)`

If the parser matches return its result, otherwise the default.


### `chmp.parser.apply`
`chmp.parser.apply(*args, **kwargs)`

Apply a transformation to the full result of the given parser.


### `chmp.parser.regex`
`chmp.parser.regex(*args, **kwargs)`

Match a single token against the regext.

If successful, the result of this parse will be the groupdict of the match.
Therefore, groups of interested should be named:

```python
>>> p.parse(p.regex(r"(?P<number>\d+)"), ["123"])
[{'number': '123'}]
```

