import queue as _queue

import numpy as np
import numba

from scipy.ndimage.measurements import label

from .util import DEFAULT_SAMPLERATE


class StreamProcessor(object):
    """Segment a continuous stream of samples.

    Usage::

        proc = StreamProcessor()

        for block in blocks:
            proc.process(block)

        proc.finish()

        # consume detected blocks from proc.queue

    """
    def __init__(self, queue=None, samplerate=DEFAULT_SAMPLERATE):
        if queue is None:
            queue = _queue.Queue()

        self.queue = queue
        self.in_speech = False
        self.buffer = []
        self.samplerate = samplerate

    def process(self, indata):
        was_in_speech = self.in_speech
        self.in_speech, blocks = compute_speechiness(
            indata,
            samplerate=self.samplerate,
            in_speech=self.in_speech,
        )

        # a single block found that continues to be in-speech
        if was_in_speech and self.in_speech and len(blocks) == 1:
            self.buffer.extend(indata[b] for b in blocks)

        else:
            # the first block finishes the speech
            if was_in_speech:
                self.buffer.extend(indata[b] for b in blocks[:1])
                self.finish()
                blocks = blocks[1:]

            # if in-speech, the last block is not yet finished
            if self.in_speech:
                b = blocks[-1]
                self.buffer.append(indata[b])
                blocks = blocks[:-1]

            # all other blocks can be emitted directly
            for b in blocks:
                self.queue.put(indata[b], block=False)

    def finish(self):
        if not self.buffer:
            return

        self.queue.put(np.concatenate(self.buffer), block=False)
        self.buffer = []


def compute_speechiness(sample, samplerate=DEFAULT_SAMPLERATE, in_speech=False):
    """Given a sample detect blocks of probable speech.

    :param numpy.ndarray sample:
        the sample as a one-dimensional array.

    :param int samplerate:
        the sample rate in Hertz

    :param bool in_speech:
        whether the previous sample ended in a probable speech block

    :returns:
        a tuple `(in_speech, blocks)` with a boolean flag whether the sample
        ended with a probable speech block and a list of detected speech
        blocks. Each block is returned as an array of indices.
    """
    speechiness = np.convolve(abs(sample), np.ones(400) / 400, mode='same')
    result = np.zeros_like(speechiness, dtype=np.int8)
    in_speech = _compute_speechiness(
        np.asarray(speechiness, dtype=np.float32),
        0.01,
        samplerate // 4,
        out=result,
        in_speech=in_speech,
    )

    indices, num_features = label(result)

    blocks = []
    for feature in range(1, num_features + 1):
        block, = np.nonzero(indices == feature)
        blocks.append(block)

    return in_speech, blocks


@numba.jit('int8(float32[:], float32, int64, int8[:], int8)', nopython=True, nogil=True)
def _compute_speechiness(speechiness, threshold, advance, out, in_speech=False):
    if not in_speech:
        idx = 0
        start = 0
        end = 0

    else:
        idx = 0
        start = 0
        end = advance

    while idx < len(speechiness):
        above_threshold = (speechiness[idx] > threshold)

        if not in_speech and above_threshold:
            start = max(0, idx - advance)
            end = min(len(speechiness) - 1, start + advance)
            in_speech = True

        if in_speech and above_threshold:
            end = idx + advance
            idx += 1

        elif in_speech and not above_threshold:
            idx += 1
            out[start:end] = 1
            in_speech = False

        idx += 1

    if in_speech:
        out[start:len(out) - 1] = 1

    return out[len(out) - 1] == 1
