import logging

import numpy as np
import torch
import torch.nn.functional as F

from python_speech_features import mfcc

from .segmentation import compute_speechiness
from .util import DEFAULT_SAMPLERATE, label_encoding, load_sample


_logger = logging.getLogger(__name__)
n_features = 13


def batch_transform_samples(samples):
    batch = [
        mfcc(sample, DEFAULT_SAMPLERATE, winlen=10e-3, winstep=5e-3, numcep=n_features)
        for sample in samples
    ]
    samples, lengths = pad_sequences(batch)

    return (
        torch.as_tensor(samples, dtype=torch.float32),
        torch.as_tensor(lengths, dtype=torch.long),
    )


def transform_x(desc):
    sample = load_sample(desc['file'])
    return mfcc(sample, DEFAULT_SAMPLERATE, winlen=10e-3, winstep=5e-3, numcep=n_features)


def transform_y(desc):
    return label_encoding.get(desc['label'], -1)


def batch_transform_x(descs):
    batch = [transform_x(desc) for desc in descs]
    samples, lengths = pad_sequences(batch)

    return (
        torch.as_tensor(samples, dtype=torch.float32),
        torch.as_tensor(lengths, dtype=torch.long),
    )


def batch_transform_y(descs):
    batch = [transform_y(desc) for desc in descs]
    return torch.as_tensor(np.asarray(batch, dtype=np.int64))


def batch_transform_xy(desc):
    return batch_transform_x(desc), batch_transform_y(desc)


def extract_single_block(sample, block=0):
    if sample.ndim == 2:
        sample = np.mean(sample, axis=1)

    _, blocks = compute_speechiness(sample)

    if len(blocks) == 0:
        raise ValueError('no sample found')

    elif len(blocks) > 1:
        print('more than one sample found')

    return sample[blocks[block]]


class KeywordModel(torch.nn.Module):
    """Transform a sequence of MFC features into class logits."""

    def __init__(
            self,
            n_features,
            n_classes,
            kernel_size=7,
            dilation_rate=2,
            n_hidden=64,
            n_layers=2,
    ):
        super().__init__()
        self.cnn_stack = build_cnn_stack(
            n_features, n_hidden,
            n_hidden=n_hidden,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
        )
        self.gru_pooling = GRUPooling(n_hidden, n_hidden)
        self.final = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, inputs, lengths):
        # Convert into format expected by CNNs
        # (batch, length, channel) -> (batch, channel, length)
        res = inputs.permute(0, 2, 1)
        res = self.cnn_stack(res)

        # pack + sort, apply GRU, unsort afterwards
        res, unsort_indices = pack_conv_output(res, lengths)
        res = self.gru_pooling(res)
        res = res[unsort_indices]

        res = F.relu(res)
        res = self.final(res)

        return res


def build_cnn_stack(n_input, n_output, *, kernel_size=7, dilation_rate=2, n_hidden=64, n_layers=2):
    input_dims = [n_input] + (n_layers - 1) * [n_hidden]
    output_dims = (n_layers - 1) * [n_hidden] + [n_output]

    cnn_stack = []

    for input_dim, output_dim in zip(input_dims, output_dims):
        cnn_stack += [
            # Pad such that the sequence stays at the same size
            torch.nn.ConstantPad1d((dilation_rate * (kernel_size - 1), 0), 0),
            torch.nn.Conv1d(input_dim, output_dim, dilation=dilation_rate, kernel_size=kernel_size),
            torch.nn.ReLU(),
        ]

    return torch.nn.Sequential(*cnn_stack)


class GRUPooling(torch.nn.Module):
    """Appply a GRU and summarize a sequence by the last hidden state."""

    def __init__(self, in_features, out_features, n_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(in_features, out_features, num_layers=n_layers)

    def forward(self, seq):
        _, hn = self.gru(seq)
        return hn[-1]


def pack_conv_output(inputs, lengths, sorted=False):
    # conv output has shape (batch, channels, length)
    # needs to be packed into (length, batch, channels)

    if not sorted:
        _, sort_indices = torch.sort(lengths, descending=True)
        inputs = inputs[sort_indices]
        lengths = lengths[sort_indices]
        _, unsort_indices = torch.sort(sort_indices)

    else:
        unsort_indices = torch.arange(len(inputs))

    seq = torch.nn.utils.rnn.pack_padded_sequence(
        inputs.permute(2, 0, 1),
        lengths,
    )

    return seq, unsort_indices


def pad_sequences(
    *sequence_batches, dtype="float32", length=None, length_dtype="int64", factory=None
):
    """Helper to build pad a batches of sequences."""
    if length is None:
        length = _pad_sequences_determine_max_lengths(sequence_batches)

    if factory is None:
        factory = np.zeros

    length = ensure_tuple(length, len(sequence_batches))
    dtype = ensure_tuple(dtype, len(sequence_batches))

    tail_shapes = _pad_sequences_determine_tail_shapes(sequence_batches)
    batch_size = _pad_sequences_determine_batch_size(sequence_batches)

    result = []

    for sequence_batch, l, dt, ts in zip(sequence_batches, length, dtype, tail_shapes):
        sequence_padded = factory((batch_size, l, *ts), dtype=dt)
        sequence_length = factory(batch_size, dtype=length_dtype)

        for i, sequence in enumerate(sequence_batch):
            sequence_padded[i, : len(sequence)] = sequence
            sequence_length[i] = len(sequence)

        result += [sequence_padded, sequence_length]

    return tuple(result)


def _pad_sequences_determine_max_lengths(sequence_batches):
    return max(
        sequence.shape[0]
        for sequence_batch in sequence_batches
        for sequence in sequence_batch
    )


def _pad_sequences_determine_tail_shapes(sequence_batches):
    tail_shapes = [
        {tuple(sequence.shape[1:]) for sequence in sequence_batch}
        for sequence_batch in sequence_batches
    ]

    for idx, tail_shape in enumerate(tail_shapes):
        if len(tail_shape) != 1:
            raise RuntimeError(f"Inconsistent tail shapes in {idx}: {tail_shape}")

    # unpack the tail shapes (note the ",")
    tail_shapes = [tail_shape for tail_shape, in tail_shapes]

    return tail_shapes


def _pad_sequences_determine_batch_size(sequence_batches):
    batch_size = {len(sequence_batch) for sequence_batch in sequence_batches}

    if len(batch_size) != 1:
        raise RuntimeError(f"Inconsistent batch sizes {batch_size}")

    batch_size, = batch_size

    return batch_size


def ensure_tuple(obj, length):
    if not isinstance(obj, tuple):
        return tuple(obj for _ in range(length))

    return obj


def get_number_of_samples(values):
    sample_counts = {len(item) for item in values if item is not None}
    if len(sample_counts) != 1:
        raise ValueError("inconsistent batch sizes")

    n_samples, = sample_counts
    return n_samples
