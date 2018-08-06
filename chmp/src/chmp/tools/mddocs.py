#!/usr/bin/env python3
"""Support a subset of sphinx features in plain markdown files.
"""
from __future__ import print_function, division, absolute_import

import importlib
import inspect
import itertools as it
import logging
import os
import os.path
import sys

try:
    import docutils

except ImportError:
    print('this script requires docutils to be installed', file=sys.stderr)
    raise SystemExit(1)

from docutils.core import publish_string
from docutils.nodes import Element
from docutils.parsers.rst import roles
from docutils.writers import Writer

from chmp import parser as p

_logger = logging.getLogger(__name__)


def transform_directories(src, dst):
    setup_rst_roles()

    src_dir = os.path.abspath(src)
    docs_dir = os.path.abspath(dst)

    for fname in relwalk(src_dir):
        if not fname.endswith('.md'):
            continue

        source = os.path.abspath(os.path.join(src_dir, fname))
        target = os.path.abspath(os.path.join(docs_dir, fname))

        # NOTE: always generate docs, to include newest docstrings

        _logger.info('transform %s -> %s', source, target)

        with open(source, 'rt') as fobj:
            content = fobj.read()

        content = transform(content, source)

        with open(target, 'wt') as fobj:
            fobj.write(content)


def setup_rst_roles():
    roles.register_canonical_role('class', rewrite_reference)
    roles.register_canonical_role('func', rewrite_reference)


def rewrite_reference(name, rawtext, text, lineno, inliner, options=None, content=None):
    # TODO: support titles
    return [TitledReference(rawtext, reference=text, title=text)], []


class TitledReference(Element):
    pass


def relwalk(absroot, relroot='.'):
    for fname in os.listdir(absroot):
        relpath = os.path.join(relroot, fname)
        abspath = os.path.join(absroot, fname)

        if fname in {'.', '..'}:
            continue

        if os.path.isfile(abspath):
            yield relpath

        elif os.path.isdir(abspath):
            yield from relwalk(abspath, relpath)


def transform(content, source):
    content_lines = [line.rstrip() for line in content.splitlines()]
    result = p.parse(parser, content_lines)
    lines = []

    directive_map = {
        'include': include,
        'autofunction': autofunction,
        'autoclass': autoclass,
        'automethod': automethod,
        'automodule': automodule,
        'literalinclude': literalinclude,
    }

    for part in result:
        if part['type'] == 'verbatim':
            lines += [part['line']]

        elif part['type'] in directive_map:
            lines += directive_map[part['type']](part, source)

        else:
            raise NotImplementedError('unknown parse fragmet {}'.format(part['type']))

    return '\n'.join(lines)


def build_parser():
    simple_directives = [
        'autofunction', 'include', 'autoclass', 'automethod', 'literalincldue',
    ]

    end_of_directive = p.first(
        p.map(
            lambda line: {'type': 'verbatim', 'line': line},
            p.eq(''),
        ),
        p.end_of_sequence(),
    )

    def make_simple_parser(name):
        return p.sequential(
            p.map(
                lambda line: {'type': name, 'line': line},
                p.predicate(lambda line: line.startswith('.. {}::'.format(name))),
            ),
            end_of_directive,
        )

    simple_parsers = [make_simple_parser(name) for name in simple_directives]

    automodule_parser = p.sequential(
        p.build_object(
            p.map(
                lambda line: {'type': 'automodule', 'line': line},
                p.predicate(lambda line: line.startswith('.. automodule::'))
            ),
            p.repeat(
                p.first(
                    p.map(lambda line: {'members': True}, p.eq('    :members:')),
                ),
            ),
        ),
        end_of_directive,
    )

    return p.repeat(
        p.first(
            *simple_parsers,
            automodule_parser,
            p.map(
                lambda line: {'type': 'verbatim', 'line': line},
                p.first(
                    p.fail_if(
                        lambda line: line.startswith('..'),
                        lambda line: 'unknown directive {!r}'.format(line),
                    ),
                    p.any(),
                )
            ),
        ),
    )

parser = build_parser()


def autofunction(part, source):
    return autoobject(part)


def automethod(part, source):
    return autoobject(part, depth=2, skip_args=1)


def autoclass(part, source):
    return autoobject(part)


def automodule(part, source):
    yield from autoobject(part, header=2, depth=0)


def autoobject(part, depth=1, header=3, skip_args=0):
    line = part['line']

    _, what = line.split('::')

    if '(' in what:
        signature = what
        what, _1, _2 = what.partition('(')

    else:
        signature = None

    obj = import_object(what, depth=depth)
    yield from document_object(obj, what, signature=signature, header=header, skip_args=skip_args)

    if part.get('members') is True:
        for k in get_member_names(obj):
            v = getattr(obj, k)
            yield from document_object(v, what + '.' + k)


def document_object(obj, what, *, signature=None, header=3, skip_args=0):
    if signature is None:
        if inspect.isfunction(obj):
            signature = format_signature(what, obj, skip=skip_args)

        elif inspect.isclass(obj):
            signature = format_signature(what, obj.__init__, skip=1 + skip_args)

        else:
            signature = ''

    yield '{} `{}`'.format('#' * header, what.strip())

    if signature:
        yield '`{}`'.format(signature)

    yield ''
    yield render_docstring(obj)


def format_signature(label, func, skip=0):
    args = inspect.getfullargspec(func)
    args, varargs, keywords, defaults = args[:4]

    args = args[skip:]
    if not defaults:
        defaults = []

    varargs = [] if varargs is None else [varargs]
    keywords = [] if keywords is None else [keywords]

    args = (
        ['{}'.format(arg) for arg in args[:len(defaults)]] +
        ['{}={!r}'.format(arg, default) for arg, default in zip(args[-len(defaults):], defaults)] +
        ['*{}'.format(arg) for arg in varargs] +
        ['**{}'.format(arg) for arg in keywords]
    )

    return '{}({})'.format(label.strip(), ', '.join(args))


def literalinclude(part, source):
    line = part['line']

    _, what = line.split('::')
    what = what.strip()

    what = os.path.abspath(os.path.join(os.path.dirname(source), what))
    _, ext = os.path.splitext(what)

    type_map = {
        '.py': 'python',
        '.sh': 'bash',
    }

    with open(what, 'r') as fobj:
        content = fobj.read()

    yield '```' + type_map.get(ext.lower(), '')
    yield content
    yield '```'


def include(part, source):
    line = part['line']
    _, what = line.split('::')
    what = what.strip()

    what = os.path.abspath(os.path.join(os.path.dirname(source), what))

    with open(what, 'r') as fobj:
        content = fobj.read()

    yield content


def render_docstring(obj):
    doc = obj.__doc__ or '[undocumented]'
    doc = unindent(doc)

    return publish_string(
        doc,
        writer=MarkdownWriter(),
        settings_overrides={'output_encoding': 'unicode'}
    )


class MarkdownWriter(Writer):
    def translate(self):
        self.output = ''.join(self._translate(self.document))

    def _translate(self, node):
        func = '_translate_{}'.format(type(node).__name__)
        try:
            func = getattr(self, func)

        except AttributeError:
            raise NotImplementedError('cannot translate %r (%r)' % (node, node.astext()))

        return func(node)

    def _translate_children(self, node):
        for c in node.children:
            yield from self._translate(c)

    _translate_document = _translate_children

    def _translate_paragraph(self, node):
        yield from self._translate_children(node)
        yield '\n\n'

    def _translate_literal_block(self, node):
        yield '```\n'
        yield node.astext()
        yield '\n'
        yield '```\n'
        yield '\n'

    def _translate_Text(self, node):
        yield node.astext()

    def _translate_literal(self, node):
        yield '`{}`'.format(node.astext())

    def _translate_bullet_list(self, node):
        for c in node.children:
            prefixes = it.chain(['- ', ], it.repeat('  '))

            child_content = ''.join(self._translate_children(c))
            child_content = child_content.splitlines()
            child_content = (l.strip() for l in child_content)
            child_content = (l for l in child_content if l)
            child_content = '\n'.join(p + l for p, l in zip(prefixes, child_content))
            child_content = child_content + '\n'

            yield child_content

        yield '\n'

    def _translate_field_list(self, node):
        by_section = {}

        for c in node.children:
            name, body = c.children
            parts = name.astext().split()
            section, *parts = parts

            # indent parameter descriptions (as part of a list)
            indent = '  ' if section in {'param', 'ivar'} else ''

            body = ''.join(self._translate_children(body))
            body.strip()
            body = '\n'.join(indent + line for line in body.splitlines())
            body = body.rstrip()

            if section in {'param', 'ivar'}:
                if len(parts) == 2:
                    type, name = parts

                elif len(parts) == 1:
                    name, = parts
                    type = 'any'

                else:
                    raise RuntimeError()

                value = f'* **{name}** (*{type}*):\n{body}\n'

            elif section == 'returns':
                value = f'{body}\n'

            else:
                raise NotImplementedError('unknown section %s' % section)

            by_section.setdefault(section, []).append(value)

        known_sections = ['param', 'returns', 'ivar']
        section_titles = {'param': 'Parameters', 'returns': 'Returns', 'ivar': 'Instance variables'}

        for key in known_sections:
            if key in by_section:
                yield f'#### {section_titles[key]}\n\n'

                for item in by_section[key]:
                    yield f'{item}'

                yield '\n'

        unknown_sections = set(by_section) - set(known_sections)
        if unknown_sections:
            raise ValueError('unknown sections %s' % unknown_sections)

    def _translate_TitledReference(self, node):
        yield '[{0}](#{1})'.format(
            node.attributes['title'],
            node.attributes['reference'].replace('.', '').lower(),
        )

    def _translate_strong(self, node):
        yield '**'
        yield from self._translate_children(node)
        yield '**'

    def _translate_reference(self, node):
        yield from self._translate_children(node)


def unindent(doc):
    def impl():
        lines = doc.splitlines()
        indent = find_indent(lines)

        if lines:
            yield lines[0]

        for line in lines[1:]:
            yield line[indent:]

    return '\n'.join(impl())


def find_indent(lines):
    for line in lines[1:]:
        if not line.strip():
            continue

        return len(line) - len(line.lstrip())

    return 0


def import_object(what, depth=1):
    parts = what.split('.')

    if depth > 0:
        mod = '.'.join(parts[:-depth]).strip()
        what = parts[-depth:]

    else:
        mod = '.'.join(parts).strip()
        what = []

    obj = importlib.import_module(mod)

    for p in what:
        obj = getattr(obj, p)

    return obj


def get_member_names(obj):
    """Return all members names of the given object"""
    # TODO: handle classes

    if hasattr(obj, '__all__'):
        return obj.__all__

    return [
        k
        for k, v in vars(obj).items()
        if (
            getattr(v, '__module__', None) == obj.__name__ and
            getattr(v, '__doc__', None) is not None
        )
    ]
