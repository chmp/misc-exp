import contextlib
import operator as op

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

    hover = HoverTool(tooltips=[
        ("Name", "@name"),
        ('File', '@file'),
        ('Count', '@count'),
        ('Type', '@type'),
        ('Time', '@time'),
    ])

    p = figure(active_scroll='wheel_zoom', x_axis_label='Runtime [ms]', y_axis_label='Stack depth')
    p.tools.append(hover)
    p.rect('x', 'y', 'width', 'height', color='color', source=ds)

    if show:
        do_show(p)

    else:
        return p


def build_data(stats, cmap='plasma_r', skip_empty=True):
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
    r, g, b, a = cmap(node.count / parent_count)
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