"""Build custom parsers fast using parser combinators.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
# distributed as chmp.parser
import functools as ft
import inspect
import operator as op
import re


class ParseError(Exception):
    pass


def parse(parser, tokens, partial=False, offset=0):
    rest, result, debug = parser(tokens, offset)

    info = _find_latest_error(debug)

    parse_error = 'parse error at position {}: {}'.format(
        info.get('offset', offset),
        info.get('message', 'unknown error'),
    )

    if result is None:
        raise ParseError(parse_error)

    if not partial and rest:
        raise ParseError('sequence not fully consumed: {!r}, {}'.format(_excerpt(rest), parse_error))

    return result


def _find_latest_error(debug):
    max_info = debug
    max_offset = debug.get('offset', 0)

    for child in debug.get('children', []):
        max_child = _find_latest_error(child)

        if max_child.get('offset', 0) >= max_offset:
            max_info = max_child
            max_offset = max_child.get('offset', 0)

    return max_info


def inspect_parser(parser):
    """Recursively inspect a parser.
    """
    if hasattr(parser, '_parser_parameters'):
        result = extract_parameters(parser)
        result['name'] = parser.__name__
        return {k: inspect_parser(v) for k, v in result.items()}

    elif isinstance(parser, (list, tuple)):
        return type(parser)(inspect_parser(item) for item in parser)

    elif isinstance(parser, dict):
        return {k: inspect_parser(v) for k, v in parser.items()}

    else:
        return parser


def extract_parameters(parser):
    nonlocals = inspect.getclosurevars(parser).nonlocals

    result = {}
    for k in parser._parser_parameters:
        try:
            result[k] = nonlocals[k]

        except KeyError:
            raise ValueError('could not extract {} from {}'.format(k, parser))

    for k, extractor in parser._parser_extractors.items():
        try:
            result[k] = extractor(parser)

        except Exception as e:
            raise ValueError('could not extract {} from {}'.format(k, parser)) from e

    return result


def parameters(*names, **extractors):
    def decorator(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            parser = func(*args, **kwargs)
            parser._parser_parameters = tuple(names)
            parser._parser_extractors = dict(extractors)

            return parser

        return wrapper

    return decorator


def partial_arg(partial_name, arg_index):
    def extract_partial_arg(obj):
        nonloals = inspect.getclosurevars(obj).nonlocals
        partial = nonloals[partial_name]
        return partial.args[arg_index]

    return extract_partial_arg


def _debug(state, offset, consumed, *, children=(), **kwargs):
    return dict(state=state, offset=offset, consumed=consumed, children=list(children), **kwargs)


def _add_child(debug, d):
    debug['consumed'] += d.get('consumed', 0)
    debug['children'].append(d)


def _ok(offset, consumed, **kwargs):
    return _debug('ok', offset, consumed, **kwargs)


def _fail(offset, consumed, **kwargs):
    return _debug('fail', offset, consumed, **kwargs)


@parameters()
def noop():
    def noop_parser(tokens, offset):
        return tokens, [], _ok(offset, 0)

    return noop_parser


@parameters('predicate')
def predicate(predicate, message=None):
    if message is None:
        message = 'predicate did not match'

    def predicate_parser(tokens, offset):
        head = list(tokens[:1])
        tail = tokens[1:]

        if not head or not predicate(head[0]):
            return tokens, None, _fail(offset, 0, message=message)

        return tail, head, _ok(offset, 1)

    return predicate_parser


@parameters(needle=partial_arg('predicate', 0))
def ne(needle):
    return predicate(ft.partial(op.ne, needle), message='found {!r}'.format(needle))


@parameters(needle=partial_arg('predicate', 0))
def eq(needle):
    return predicate(ft.partial(op.eq, needle), message='did not find {!r}'.format(needle))


@parameters()
def any():
    return predicate(lambda _: True)


@parameters('parsers', 'flatten')
def sequential(parser, *parsers, flatten=True):
    """Match a sequence of parsers exactly.
    """
    parsers = (parser,) + parsers

    def sequential_parser(tokens, offset):
        result = []
        append = result.append if flatten is False else result.extend

        rest = tokens
        debug = _debug(None, offset, 0)

        for parser in parsers:
            rest, m, d = parser(rest, offset)
            _add_child(debug, d)
            offset += d.get('consumed', 0)

            if m is None:
                return tokens, None, dict(debug, state='fail')

            append(m)

        return rest, result, dict(debug, state='ok')

    return sequential_parser


@parameters('parsers')
def first(parser, *parsers):
    """Return the result of the first parser to match.
    """
    parsers = (parser,) + parsers

    def first_parser(tokens, offset):
        debug = _debug(None, offset, 0)
        for parser in parsers:
            rest, result, d = parser(tokens, offset)
            debug['children'].append(d)

            if result is not None:
                return rest, result, dict(debug, state='ok', consumed=d.get('consumed', 0))

        return tokens, None, dict(debug, state='fail', message='no matching parser')

    return first_parser


@parameters('parser')
def repeat(parser, *parsers, flatten=True):
    """Match 0 or more occurrences.
    """
    if parsers:
        parser = sequential(parser, *parsers)

    def repeat_parser(tokens, offset):
        result = []
        append = result.append if flatten is False else result.extend

        debug = _ok(offset, 0)
        while len(tokens):
            tokens, m, d = parser(tokens, offset)
            _add_child(debug, d)

            if m is None:
                break

            offset += d.get('consumed', 0)

            append(m)

        return tokens, result, debug

    return repeat_parser


@parameters('parser')
def ignore(parser):
    """Ignore the result of parser.
    """

    def ignore_parser(tokens, offset):
        try:
            rest, result, d = parser(tokens, offset)

        except Exception as e:
            raise ParseError('could not run parser {}'.format(parser)) from e

        if result is None:
            return tokens, None, _fail(offset, 0, children=[d])

        return rest, [], _ok(offset, d.get('consumed', 0), children=[d])

    return ignore_parser


@parameters('needle')
def sequence_eq(needle):
    n = len(needle)

    def sequence_eq_parser(tokens, offset):
        if len(tokens) < n or tokens[:n] != needle:
            return tokens, None, _fail(offset, 0, message='did not find {!r}'.format(needle))

        return tokens[n:], [tokens[:n]], _ok(offset, n)

    return sequence_eq_parser


@parameters('parser')
def no_match(parser):
    def no_match_parser(tokens, offset):
        _, match, _ = parser(tokens, offset)
        if match is None:
            return tokens, [], _ok(offset, 0)
        else:
            return tokens, None, _fail(offset, 0)

    return no_match_parser


@parameters('transform', 'parser')
def apply(transform, parser):
    def apply_parser(tokens, offset):
        rest, result, d = parser(tokens, offset)

        if result is None:
            return tokens, None, _fail(offset, 0, children=[d])

        return rest, transform(result), _ok(offset, d.get('consumed', 0), children=[d])

    return apply_parser


@parameters('transform', 'parser')
def map(transform, parser):
    return apply(lambda m: [transform(i) for i in m], parser)


@parameters()
def end_of_sequence():
    def end_of_sequence_parser(tokens, offset):
        if not tokens:
            return tokens, [], _ok(offset, 0)

        else:
            return tokens, None, _fail(offset, 0, message='trailing tokens {!r}'.format(_excerpt(tokens)))

    return end_of_sequence_parser


@parameters('pattern')
def regex(pattern, flags=0):
    pattern = re.compile(pattern, flags=flags)

    def _match_regex(lines, offset):
        if not lines:
            return lines, None, {}

        head = lines[0]
        tail = lines[1:]

        m = pattern.match(head)

        if m is None:
            return lines, None, {}

        return tail, [m.groupdict()], {}

    return _match_regex


def _excerpt(s, n=40):
    s = str(s)
    return (s[:n - 3] + '...') if len(s) > n else s
