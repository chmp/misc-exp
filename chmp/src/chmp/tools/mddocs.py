#!/usr/bin/env python3
"""Support a subset of sphinx features in plain markdown files.
"""
from __future__ import print_function, division, absolute_import

import importlib
import inspect
import io
import itertools as it
import logging
import os
import os.path
import sys

try:
    import docutils

    assert docutils  # to silence pyflakes

except ImportError:
    print("this script requires docutils to be installed", file=sys.stderr)
    raise SystemExit(1)

from docutils.core import publish_string
from docutils.nodes import Element, Admonition
from docutils.parsers.rst import roles, directives
from docutils.parsers.rst.directives import admonitions
from docutils.writers import Writer

from chmp import parser as p

_logger = logging.getLogger(__name__)


def transform_directories(src, dst, continue_on_error=False, inventory=None):
    """Transform whole directory while rendering a subset of sphinx markup."""
    src_dir = os.path.abspath(src)
    docs_dir = os.path.abspath(dst)

    for fname in relwalk(src_dir):
        if not fname.endswith(".md"):
            continue

        source = os.path.abspath(os.path.join(src_dir, fname))
        target = os.path.abspath(os.path.join(docs_dir, fname))

        # NOTE: always generate docs, to include newest docstrings

        _logger.info("transform %s -> %s", source, target)
        try:
            transform_file(source, target, inventory=inventory)

        except Exception:
            if continue_on_error:
                _logger.error("could not transform %s", source, exc_info=True)
                continue

            else:
                raise


def transform_file(source, target, inventory=None):
    """Transform one file while rendering a subset of sphinx markup."""
    setup_rst_roles()

    with open(source, "rt") as fobj:
        content = fobj.read()

    content = transform(content, source, inventory=inventory)

    with open(target, "wt") as fobj:
        fobj.write(content)


def setup_rst_roles():
    roles.register_canonical_role("class", rewrite_reference)
    roles.register_canonical_role("func", rewrite_reference)


def rewrite_reference(name, rawtext, text, lineno, inliner, options=None, content=None):
    # TODO: support titles
    return [TitledReference(rawtext, reference=text, title=text)], []


class TitledReference(Element):
    pass


def relwalk(absroot, relroot="."):
    for fname in os.listdir(absroot):
        relpath = os.path.join(relroot, fname)
        abspath = os.path.join(absroot, fname)

        if fname in {".", ".."}:
            continue

        if os.path.isfile(abspath):
            yield relpath

        elif os.path.isdir(abspath):
            yield from relwalk(abspath, relpath)


def transform(content, source, inventory=None):
    if inventory is None:
        inventory = {}

    reference_resolver = ChainResolver(
        InventoryResolver(inventory=inventory), GithubLinkResolver()
    )

    content_lines = [line.rstrip() for line in content.splitlines()]
    result = p.parse(parser, content_lines)
    lines = []

    directive_map = {
        "include": include,
        "autofunction": autofunction,
        "autoclass": autoclass,
        "automethod": automethod,
        "automodule": automodule,
        "literalinclude": literalinclude,
    }

    for part in result:
        if part["type"] == "verbatim":
            lines += [part["line"]]

        elif part["type"] in directive_map:
            lines += directive_map[part["type"]](
                part, source, reference_resolver=reference_resolver
            )

        else:
            raise NotImplementedError("unknown parse fragmet {}".format(part["type"]))

    return "\n".join(lines)


def build_parser():
    simple_directives = [
        "autofunction",
        "include",
        "autoclass",
        "automethod",
        "literalinclude",
    ]

    end_of_directive = p.first(
        p.map(lambda line: {"type": "verbatim", "line": line}, p.eq("")),
        p.end_of_sequence(),
    )

    def make_simple_parser(name):
        return p.sequential(
            p.map(
                lambda line: {"type": name, "line": line},
                p.predicate(lambda line: line.startswith(".. {}::".format(name))),
            ),
            end_of_directive,
        )

    simple_parsers = [make_simple_parser(name) for name in simple_directives]

    automodule_parser = p.sequential(
        p.build_object(
            p.map(
                lambda line: {"type": "automodule", "line": line},
                p.predicate(lambda line: line.startswith(".. automodule::")),
            ),
            p.repeat(
                p.first(p.map(lambda line: {"members": True}, p.eq("    :members:")))
            ),
        ),
        end_of_directive,
    )

    return p.repeat(
        p.first(
            *simple_parsers,
            automodule_parser,
            p.map(
                lambda line: {"type": "verbatim", "line": line},
                p.first(
                    p.fail_if(
                        lambda line: line.startswith(".."),
                        lambda line: "unknown directive {!r}".format(line),
                    ),
                    p.any(),
                ),
            ),
        )
    )


parser = build_parser()


def autofunction(part, source, *, reference_resolver):
    return autoobject(part, reference_resolver=reference_resolver)


def automethod(part, source, *, reference_resolver):
    return autoobject(part, depth=2, skip_args=1, reference_resolver=reference_resolver)


def autoclass(part, source, *, reference_resolver):
    return autoobject(part, reference_resolver=reference_resolver)


def automodule(part, source, *, reference_resolver):
    yield from autoobject(
        part, header=2, depth=0, reference_resolver=reference_resolver
    )


def autoobject(part, *, reference_resolver, depth=1, header=3, skip_args=0):
    what, signature = extract_label_signature(part["line"])

    obj = import_object(what, depth=depth)
    yield from document_object(
        obj,
        what,
        signature=signature,
        header=header,
        skip_args=skip_args,
        reference_resolver=reference_resolver,
    )

    if part.get("members") is True:
        for child, child_kwargs in get_members(what, obj, header=header, skip_args=0):
            yield from document_object(
                child, **child_kwargs, reference_resolver=reference_resolver
            )


def extract_label_signature(autodoc_line):
    """Extract the object name and signature of the object being document.

    For example::

        >>> extract_label_signature(':: foo(a, b)')
        'foo', 'foo(a, b)'
        >>> extract_label_signature(':: foo')
        'foo', None
    """
    _, what = autodoc_line.split("::")

    if "(" in what:
        signature = what.strip()
        what, *_ = what.partition("(")

    else:
        signature = None

    # NOTE: if given, the signature is already stripped
    return what.strip(), signature


def get_members(parent_k, parent, header, skip_args=0):
    for k in get_member_names(parent):
        v = getattr(parent, k)
        full_key = f"{parent_k}.{k}"

        yield v, dict(label=full_key, header=header + 1, skip_args=skip_args)

        if inspect.isclass(v):
            yield from get_members(full_key, v, header=header + 1, skip_args=1)


def document_object(
    obj, label, *, reference_resolver, signature=None, header=3, skip_args=0
):
    """Document an object.

    :param label:
        the name of the object being documented, as given in the autodoc
        command. It is exclusively used as a label.
    :param signature:
        if given, an explicit signature to use. Otherwise the signature is
        determined from the object being documented.
    :param header:
        the header level to use for this object
    :param skip_args:
        the number of arguments to skip for function-like objects. In
        particular, use ``skip_args=1`` to skip the ``self`` argument of
        methods.
    :returns:
         a generator of lines.
    """
    if signature is None:
        if inspect.isfunction(obj):
            signature = format_signature(label, obj, skip=skip_args)

        elif inspect.isclass(obj):
            signature = format_signature(label, obj.__init__, skip=1 + skip_args)

        else:
            signature = ""

    yield "{} `{}`".format("#" * header, label.strip())

    if signature:
        yield "`{}`".format(signature)

    yield ""
    yield render_docstring(obj, reference_resolver=reference_resolver)


def format_signature(label, func, skip=0):
    args = inspect.getfullargspec(func)
    args, varargs, keywords, defaults, kwonlyargs, kwonlydefaults = args[:6]

    args = args[skip:]
    if not defaults:
        defaults = []

    if not kwonlydefaults:
        kwonlydefaults = {}

    fragments = []

    fragments.extend("{}".format(arg) for arg in args[: len(args) - len(defaults)])
    fragments.extend(
        "{}={!r}".format(arg, default)
        for arg, default in zip(args[len(args) - len(defaults) :], defaults)
    )

    if varargs is not None:
        fragments.append("*{}".format(varargs))

    elif kwonlyargs:
        fragments.append("*")

    fragments.extend(
        "{}".format(arg) for arg in kwonlyargs if not arg in kwonlydefaults
    )

    fragments.extend(
        "{}={!r}".format(arg, kwonlydefaults[arg])
        for arg in kwonlyargs
        if arg in kwonlydefaults
    )

    if keywords is not None:
        fragments.append("**{}".format(keywords))

    return "{}({})".format(label.strip(), ", ".join(fragments))


def literalinclude(part, source, reference_resolver=None):
    line = part["line"]

    _, what = line.split("::")
    what = what.strip()

    what = os.path.abspath(os.path.join(os.path.dirname(source), what))
    _, ext = os.path.splitext(what)

    type_map = {".py": "python", ".sh": "bash"}

    with open(what, "r") as fobj:
        content = fobj.read()

    yield "```" + type_map.get(ext.lower(), "")
    yield content
    yield "```"


def include(part, source, reference_resolver=None):
    line = part["line"]
    _, what = line.split("::")
    what = what.strip()

    what = os.path.abspath(os.path.join(os.path.dirname(source), what))

    with open(what, "r") as fobj:
        content = fobj.read()

    yield content


def render_docstring(obj, reference_resolver):
    """Render the docstring for an object.

    For classes the docstring of the class and init are merged.
    """
    doc = obj.__doc__ or ""
    doc = unindent(doc)

    # merge the docstring of the __init__ method with the main docstring
    if inspect.isclass(obj) and obj.__init__.__doc__ is not None:
        doc = doc + "\n\n" + unindent(obj.__init__.__doc__)

    return publish_string(
        doc,
        writer=MarkdownWriter(reference_resolver=reference_resolver),
        settings_overrides={"output_encoding": "unicode"},
    )


class MarkdownWriter(Writer):
    def __init__(self, reference_resolver):
        super().__init__()
        self.reference_resolver = reference_resolver

    def translate(self):
        self.output = "".join(self._translate(self.document))

    def _translate(self, node):
        func = "_translate_{}".format(type(node).__name__)
        try:
            func = getattr(self, func)

        except AttributeError:
            raise NotImplementedError(
                "cannot translate %r (%r)" % (node, node.astext())
            )

        return func(node)

    def _translate_children(self, node):
        for c in node.children:
            yield from self._translate(c)

    _translate_document = _translate_children

    def _translate_section(self, node):
        yield from self._translate_children(node)

    def _translate_title(self, node):
        yield "#### %s\n" % node.astext()
        yield "\n"

    def _translate_paragraph(self, node):
        yield from self._translate_children(node)
        yield "\n\n"

    def _translate_literal_block(self, node):
        yield "```python\n"
        yield node.astext()
        yield "\n"
        yield "```\n"
        yield "\n"

    def _translate_Text(self, node):
        yield node.astext()

    def _translate_literal(self, node):
        yield "`{}`".format(node.astext())

    def _translate_bullet_list(self, node):
        for c in node.children:
            prefixes = it.chain(["- "], it.repeat("  "))

            child_content = "".join(self._translate_children(c))
            child_content = child_content.splitlines()
            child_content = (l.strip() for l in child_content)
            child_content = (l for l in child_content if l)
            child_content = "\n".join(p + l for p, l in zip(prefixes, child_content))
            child_content = child_content + "\n"

            yield child_content

        yield "\n"

    def _translate_field_list(self, node):
        by_section = {}

        for c in node.children:
            name, body = c.children
            parts = name.astext().split()
            section, *parts = parts

            # indent parameter descriptions (as part of a list)
            indent = "  " if section in {"param", "ivar", "raises"} else ""

            body = "".join(self._translate_children(body))
            body.strip()
            body = "\n".join(indent + line for line in body.splitlines())
            body = body.rstrip()

            if section in {"param", "ivar", "raises"}:
                if len(parts) == 2:
                    type, name = parts

                elif len(parts) == 1:
                    name, = parts
                    type = "any"

                else:
                    raise RuntimeError(f"did not understand {section} {parts}")

                value = f"* **{name}** (*{type}*):\n{body}\n"

            elif section == "returns":
                value = f"{body}\n"

            else:
                raise NotImplementedError("unknown section %r" % section)

            by_section.setdefault(section, []).append(value)

        section_titles = {
            "param": "Parameters",
            "returns": "Returns",
            "ivar": "Instance variables",
            "raises": "Raises",
        }

        for key in section_titles:
            if key in by_section:
                yield f"#### {section_titles[key]}\n\n"

                for item in by_section[key]:
                    yield f"{item}"

                yield "\n"

        unknown_sections = set(by_section) - set(section_titles)
        if unknown_sections:
            raise ValueError("unknown sections %r" % unknown_sections)

    def _translate_definition_list(self, node):
        for child in node.children:
            children_by_type = {}

            for subchild in child.children:
                node_name = type(subchild).__name__
                children_by_type[node_name] = subchild

            name = children_by_type["term"].astext()
            definition = children_by_type["definition"]

            body = "".join(self._translate_children(definition))
            body.strip()
            body = "\n".join("  " + line for line in body.splitlines())
            body = body.rstrip()

            if "classifier" not in children_by_type:
                yield f"* **{name}**:\n{body}\n"

            else:
                arg_type = "".join(
                    self._translate_children(children_by_type["classifier"])
                )
                yield f"* **{name}** (*`{arg_type}`*):\n{body}\n"

        yield "\n"

    def _translate_TitledReference(self, node):
        yield "[{0}]({1})".format(
            node.attributes["title"],
            self.reference_resolver.resolve(node.attributes["reference"]),
        )

    def _translate_strong(self, node):
        yield "**"
        yield from self._translate_children(node)
        yield "**"

    def _translate_reference(self, node):
        yield from self._translate_children(node)

    def _translate_title_reference(self, node):
        yield from self._translate_children(node)

    def _translate_note(self, node):
        return self._util_translate_admonition("Note", node)

    def _translate_warning(self, node):
        return self._util_translate_admonition("Warning", node)

    def _translate_system_message(self, node):
        return self._util_translate_admonition("System Message", node)

    def _translate_seealso(self, node):
        return self._util_translate_admonition("See also", node)

    def _translate_todo(self, node):
        return self._util_translate_admonition("Todo", node)

    def _util_translate_admonition(self, title, node):
        body = "".join(self._translate_children(node))
        body = body.splitlines()

        yield f"> **{title}:**\n"
        yield ">\n"
        yield from ("> " + line for line in body)
        yield "\n\n"


def unindent(doc):
    def impl():
        lines = doc.splitlines()
        indent = find_indent(lines)

        if lines:
            yield lines[0]

        for line in lines[1:]:
            yield line[indent:]

    return "\n".join(impl())


def find_indent(lines):
    for line in lines[1:]:
        if not line.strip():
            continue

        return len(line) - len(line.lstrip())

    return 0


def import_object(what, depth=1):
    parts = what.split(".")

    if depth > 0:
        mod = ".".join(parts[:-depth]).strip()
        what = parts[-depth:]

    else:
        mod = ".".join(parts).strip()
        what = []

    obj = importlib.import_module(mod)

    for part in what:
        obj = getattr(obj, part)

    return obj


def get_member_names(obj):
    """Return all members names of the given object"""
    if inspect.ismodule(obj):
        if hasattr(obj, "__all__"):
            return obj.__all__

        return [
            k
            for k, v in vars(obj).items()
            if (
                getattr(v, "__module__", None) == obj.__name__
                and getattr(v, "__doc__", None) is not None
                and not k.startswith("_")
            )
        ]

    elif inspect.isclass(obj):
        if hasattr(obj, "__members__"):
            return obj.__members__

        return [
            k
            for k, v in vars(obj).items()
            if (
                callable(v)
                and getattr(v, "__doc__", None) is not None
                and not k.startswith("_")
            )
        ]

    else:
        raise ValueError(f"cannot get members of {obj!r}")


# register additional admonitions
class todo(Admonition, Element):
    pass


class Todo(admonitions.BaseAdmonition):
    node_class = todo


directives.register_directive("todo", Todo)


class seealso(Admonition, Element):
    pass


class SeeAlso(admonitions.BaseAdmonition):
    node_class = seealso


directives.register_directive("seealso", SeeAlso)


class ChainResolver:
    def __init__(self, *resolvers):
        self.resolvers = resolvers

    def resolve(self, ref, kind=None):
        for resolver in self.resolvers:
            try:
                return resolver.resolve(ref, kind)
            except KeyError:
                pass

        raise KeyError()


class GithubLinkResolver:
    def resolve(self, ref, kind=None):
        return "#{}".format(ref.replace(".", "").lower())


class InventoryResolver:
    def __init__(self, inventory):
        self.inventory = inventory
        self.order = [
            "py:module",
            "py:class",
            "py:function",
            "py:exception",
            "py:method",
            "py:classmethod",
            "py:staticmethod",
            "py:attribute",
            "py:data",
        ]

    def resolve(self, ref, kind=None):
        if kind is None:
            kind = self.order

        elif isinstance(kind, str):
            kind = [kind]

        for sec in kind:
            try:
                *_, uri, _ = self.inventory.get(sec, {})[ref]
                return uri

            except KeyError:
                continue

        raise KeyError("could not find {}".format(ref))


def load_inventory(uris):
    import requests
    from sphinx.util.inventory import InventoryFile

    uris = list(uris)
    inventory = {}

    for base_uri in uris:
        object_inv_uri = f"{base_uri}/objects.inv"
        r = requests.get(object_inv_uri)
        r.raise_for_status()

        with io.BytesIO(r.content) as fobj:
            sub_inventory = InventoryFile.load(fobj, base_uri, lambda a, b: f"{a}/{b}")

        for section, data in sub_inventory.items():
            inventory.setdefault(section, {}).update(data)

    return dict(uris=uris, inventory=inventory)
