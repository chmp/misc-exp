import numpy as np

from . import _hexworld
from ._hexworld import *


def to_buffer(a):
    if a.dtype == 'int32':
        return _hexworld._to_Int32Buffer(a)
    
    elif a.dtype == 'int64':
        return _hexworld._to_Int64Buffer(a)

    raise ValueError(f"unknown dtype {a.dtype!r}")


def default_display(col, row, value):
    return str(value)


def show(buffer, display=default_display):
    screen = make_screen(buffer)
    draw_grid(buffer, screen)
    draw_content(buffer, screen, display)

    return '\n'.join(''.join(row) for row in screen)


def plot(buffer, colors):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    buffer = np.asarray(buffer)

    c60 = np.cos(60 / 360 * 2 * np.pi)
    s60 = np.sin(60 / 360 * 2 * np.pi)
    dh = 2 * s60
    dw = 3 * c60

    plt.figure(figsize=(
        (dw * buffer.shape[1] - c60 + 2 * c60) / 4,
        (s60 + dh * buffer.shape[0]) / 4,
    ))

    for i in range(buffer.shape[0]):
        for j in range(buffer.shape[1]):
            color = colors.get(buffer[i, j], '0.5')

            x = dw * j
            y = -dh * i - (j % 2) * s60

            plt.gca().add_artist(mpl.patches.Polygon([
                (x - 1, y + 0), (x - c60, y + s60), (x + c60, y + s60), 
                (x + 1, y + 0), (x + c60, y - s60), (x - c60, y - s60),
            ], closed=True, facecolor=color, edgecolor='k', clip_on=False))

    plt.xlim(-2 * c60, dw * buffer.shape[1] - c60)
    plt.ylim(-dh * buffer.shape[0], +s60)
    plt.box(False)
    plt.xticks([])
    plt.yticks([])


def make_screen(buffer):
    w = 4 * (buffer.width  + buffer.width % 2) + 1
    h = 2 * (buffer.height + 1);

    return [[' ' for _ in range(w)] for _ in range(h)]


def draw_grid(buffer, screen):
    w = 4 * (buffer.width  + buffer.width % 2) + 1
    h = 2 * (buffer.height + 1);

    for i in range(0, h, 2):
        first_line = i == 0

        for j in range(0, w - 1, 8):
            lower_corner = (i == (h - 2)) and (j == 0)

            screen[i + 0][j:j+8] = '\\___/   ' if not first_line else ' ___    '
            screen[i + 1][j:j+8] = '/   \\___' if not lower_corner else  '    \\___'

        if not first_line:
            screen[i + 0][-1] = '\\'
            screen[i + 1][-1] = '/'


def draw_content(buffer, screen, display=default_display):
    for col in range(buffer.width):
        for row in range(buffer.height):
            row_offset = col %2 + 1
            col_offset = 1 + 4 * col
        
            s = display(col, row, buffer[col, row])
            s = s.center(3)[:3]

            screen[2 * row + row_offset][col_offset:col_offset + 3] = s
