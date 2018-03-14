import contextlib
import json
import operator as op
import hashlib

import vmprof


@contextlib.contextmanager
def collect_profile(fname='profile.dat'):
    with open(fname, 'w+b') as fobj:
        vmprof.enable(fobj.fileno(), memory=False)
        try:
            yield Profile(fname)

        finally:
            vmprof.disable()


class Profile:
    def __init__(self, fname):
        self.fname = fname
        self._stats = None

    @property
    def stats(self):
        if self._stats is None:
            self._stats = vmprof.read_profile(self.fname)

        return self._stats

    @property
    def tree(self):
        return self.stats.get_tree()

    def show(self):
        plot_profile(self, show=True)


def plot_profile(stats, show=False):
    from bokeh.models import HoverTool
    from bokeh.plotting import figure, show as do_show

    if isinstance(stats, Profile):
        stats = stats.stats

    ds = build_data(stats)

    p = figure(
        active_scroll='wheel_zoom',
        x_axis_label='Runtime [ms]', y_axis_label='Stack depth',
        plot_width=800, plot_height=450,
    )
    p.tools.append(
        HoverTool(tooltips=[
            ("Name", "@name"),
            ('File', '@file'),
            ('Count', '@count'),
            ('Type', '@type'),
            ('Time', '@time'),
        ], point_policy='follow_mouse')
    )
    p.rect('x', 'y', 'width', 'height', color='color', source=ds, line_color='black')

    if show:
        do_show(p)

    else:
        return p


def build_data(stats, cmap='autumn', skip_empty=True):
    from bokeh.models import ColumnDataSource
    from matplotlib import cm

    tree = stats.get_tree()
    time_factor = stats.get_runtime_in_microseconds() / tree.count

    if skip_empty:
        while len(tree.children) == 1:
            tree, = tree.children.values()

    cmap = cm.get_cmap(cmap)

    data = {}
    for d in _build_data(
            tree, offset=0, depth=0, parent_count=tree.count, cmap=cmap, time_factor=time_factor
    ):
        for k, v in d.items():
            data.setdefault(k, []).append(v)

    return ColumnDataSource(data=data)


def _build_data(node, *, offset, parent_count, depth, cmap, time_factor):
    r, g, b, a = cmap(random(node.name))
    type, name, lineno, file = node.name.split(':')

    yield dict(
        x=time_factor * (offset + node.count / 2) / 1e3,
        y=depth,
        width=time_factor * node.count / 1e3,
        height=0.95,
        color='rgba({:.0f}, {:.0f}, {:.0f}, {:.0f})'.format(255 * r, 255 * g, 255 * b, 255 * a),
        name=name[:30],
        file='{}:{}'.format(file[-30:], lineno),
        count=node.count,
        type=type,
        time=format_time(time_factor * node.count),
    )

    for child in sorted(node.children.values(), key=op.attrgetter('count'), reverse=True):
        yield from _build_data(
            child, offset=offset, depth=depth + 1, parent_count=node.count, cmap=cmap,
            time_factor=time_factor,
        )
        offset += child.count


def format_time(time):
    if time < 1e3:
        return '{.1f} µs'.format(time)

    if time < 1e6:
        return '{:.1f} ms'.format(time / 1e3)

    return '{:.1f} s'.format(time / 1e6)


def sha1(obj):
    """Create a hash for a json-encode-able object
    """
    return int(str_sha1(obj)[:15], 16)


def str_sha1(obj):
    s = json.dumps(obj, indent=None, sort_keys=True, separators=(',', ':'))
    s = s.encode('utf8')
    return hashlib.sha1(s).hexdigest()


def random(obj):
    """Return a random float in the range [0, 1)"""
    maximum_15_digit_hex = float(0xFFF_FFFF_FFFF_FFFF)
    return min(sha1(obj) / maximum_15_digit_hex, 0.9999999999999999)
