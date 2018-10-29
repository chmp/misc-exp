import numpy as np
import torch.utils.data

from ._util import default_batch_size, pack, unpack, apply_dtype

__all__ = [
    "data_loader",
    "pad_sequences",
    "NumpyDataset",
    "TransformedDataset",
    "apply_dtype",
    "pack",
    "unpack",
]


def data_loader(
    cls,
    *args,
    batch_size=default_batch_size,
    mode="predict",
    collate_fn=None,
    num_workers=None,
    pin_memory=None,
    worker_init_fn=None,
    **kwargs,
):
    """Helper to build data loaders for numpy / transform based datasets.

    :param cls:
        either a a dataset class or one of ``'numpy'`` (for
        :class:`NumpyDataset`) or ``'transformed'`` (for
        :class:`TransformedDataset`). `batch-transformed` is also valid value,
        in which case the data the transform receives a full batch and should
        also perform collation.
    :param args:
        varargs passed to `cls` to build the dataset
    :param kwargs:
        kwargs passed to `cls` to build the dataset
    :param mode:
        if ``'fit'``, shuffle the dataset and only yield batches with
        ``batch_size`` samples. If ``'predict'``, always yield all samples in
        order.

    All other arguments (if given) are passed to
    :class:`torch.utils.data.DataLoader`.
    """
    if mode not in {"fit", "predict"}:
        raise ValueError(f"Unknown mode {mode}")

    if isinstance(cls, str):
        if cls == "numpy":
            cls = NumpyDataset

        elif cls == "transformed":
            cls = TransformedDataset

        elif cls == "batch-transformed":
            transform, arg = args
            return data_loader(
                "transformed",
                lambda x: x,
                arg,
                collate_fn=transform,
                batch_size=batch_size,
                mode=mode,
                num_workers=num_workers,
                pin_memory=pin_memory,
                dtype=None,
            )

        else:
            raise ValueError("Unknown transform mode {}".format(cls))

    data_loader_kwargs = dict(batch_size=batch_size)

    if collate_fn is not None:
        data_loader_kwargs.update(collate_fn=collate_fn)

    if num_workers is not None:
        data_loader_kwargs.update(num_workers=num_workers)

    if pin_memory is not None:
        data_loader_kwargs.update(pin_memory=pin_memory)

    if worker_init_fn is not None:
        data_loader_kwargs.update(worker_init_fn=worker_init_fn)

    if mode == "fit":
        data_loader_kwargs.update(shuffle=True, drop_last=True)

    elif mode == "predict":
        data_loader_kwargs.update(shuffle=False, drop_last=False)

    dataset = cls(*args, **kwargs)
    return torch.utils.data.DataLoader(dataset, **data_loader_kwargs)


class TransformedDataset(torch.utils.data.Dataset):
    """Helper to build a dataset from transformed items of a base dataset."""

    def __init__(self, transform, *bases, dtype="float32"):
        self.bases = bases
        self.transform = transform
        self.dtype = dtype
        self.length = get_number_of_samples(bases)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        res = self.transform(*(base[idx] for base in self.bases))
        res = apply_dtype(self.dtype, res)
        return res


class NumpyDataset(torch.utils.data.Dataset):
    """Helper to build a dataset from numpy arrays."""

    def __init__(self, data, dtype="float32"):
        self.keys, self.values = pack(data)
        self.length = get_number_of_samples(self.values)
        self.dtype = dtype

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        values = [v[idx] for v in self.values]
        res = unpack(self.keys, values)
        res = apply_dtype(self.dtype, res)
        return res


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
