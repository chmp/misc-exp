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


### `chmp.parser.ignore`
`chmp.parser.ignore(*args, **kwargs)`

Ignore the result of parser.

