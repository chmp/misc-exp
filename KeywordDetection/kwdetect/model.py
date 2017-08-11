import logging
import random
import warnings

import numpy as np
import tensorflow as tf

from python_speech_features import mfcc

from .io import listdata, load_sample
from .segmentation import compute_speechiness
from .util import DEFAULT_SAMPLERATE, label_encoding, PickableTFModel, label_decoding

_logger = logging.getLogger(__name__)


def load_data(data_or_path, batch_size=10, numcep=13, iter=iter):
    """Load labled data as dense batches.

    :param Union[Iterable[dict],str] data_or_path:
        the path where the data is stored or an iterable of file descriptions
        obtained from :func:`kwdetect.io.listdata`.

    :param Optional[int] batchsize:
        the number of samples per batch. If `None`, return a single large batch
        containing all data.

    :param int numcep:
        the number of cepstra to compute.

    :param callable iter:
        a callable that is used to get the iterable for the outer loops.
        Pass ``tqdm`` for a progress bar.

    :returns:
        a list of ``(batch-data, batch-seq-lengths, batch-labels)`` triples.

    """
    if isinstance(data_or_path, str):
        data = [d for d in listdata(data_or_path) if d['label'] in label_encoding]

    else:
        data = list(data_or_path)

    random.shuffle(data)

    if batch_size is None:
        batch_size = len(data)

    n_batches = len(data) // batch_size

    for batch in iter(range(n_batches)):
        batch_data = []
        for d in data[batch_size * batch:batch_size * (batch + 1)]:
            sample = load_sample(d['file'])

            _, blocks = compute_speechiness(sample, DEFAULT_SAMPLERATE)
            if len(blocks) == 1:
                sample = sample[blocks[0]]

            else:
                warnings.warn(f'more than one block for {d["file"]}')

            batch_data.append((d['label'], sample))

        padded, lengths = prepare_samples([s for _, s in batch_data], numcep=numcep)
        labels = np.asarray([label_encoding.get(l, -1) for l, _ in batch_data])

        yield padded, lengths, labels


def prepare_samples(samples, numcep=13):
    samples = [
        mfcc(s, DEFAULT_SAMPLERATE, winlen=10e-3, winstep=5e-3, numcep=numcep)
        for s in samples
    ]
    max_len = max(len(s) for s in samples)

    padded = np.zeros((len(samples), max_len, numcep), dtype=np.float32)
    lengths = np.zeros(len(samples), dtype=np.float32)

    for i, s in enumerate(samples):
        padded[i, :len(s), :] = s
        lengths[i] = len(s)

    return padded, lengths


def predict_single(model, sample, session, block=0):
    sample = extract_single_block(sample, block=block)
    inputs, lengths = prepare_samples([sample])

    label, = model.predict({'inputs': inputs, 'lengths': lengths}, session=session)
    label = label_decoding[label]

    return label


def extract_single_block(sample, block=0):
    if sample.ndim == 2:
        sample = np.mean(sample, axis=1)

    _, blocks = compute_speechiness(sample)

    if len(blocks) == 0:
        raise ValueError('no sample found')

    elif len(blocks) > 1:
        print('more than one sample found')

    return sample[blocks[block]]


def model_fn(features, labels, mode, params):
    return _model_fn(labels=labels, mode=mode, **features, **params)


def _model_fn(labels, mode, inputs, lengths, n_layers=2, n_hidden=64, learning_rate=1e-5):
    logits = predict_logits(inputs, lengths, n_layers=n_layers, n_hidden=n_hidden)

    if mode == tf.estimator.ModeKeys.PREDICT:
        proba = tf.nn.softmax(logits)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=dict(proba=proba),
        )

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
    )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={})

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train,
        eval_metric_ops={},
    )


class Model(PickableTFModel):
    __params__ = (
        'kernel_size',
        'dilation_rate',
        'n_hidden',
        'n_layers',
        'n_features',
        'learning_rate',
    )

    def __init__(
            self,
            kernel_size=7,
            dilation_rate=2,
            n_hidden=64,
            n_layers=2,
            n_features=13,
            learning_rate=1e-5,
            graph=None,
    ):
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_features = n_features
        self.learning_rate = learning_rate

        self._build(graph=graph)

    def _build(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()

        with graph.as_default():
            self.variables = (
                set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) |
                set(tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
            )

            self.inputs_ = tf.placeholder(tf.float32, [None, None, self.n_features], name='inputs')
            self.lengths_ = tf.placeholder(tf.int64, [None], name='lengths')
            self.labels_ = tf.placeholder(tf.int64, [None], name='labels')

            self.logits_ = predict_logits(
                self.inputs_, self.lengths_,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilation_rate,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
            )

            with tf.name_scope('loss'):
                self.loss_ = tf.losses.sparse_softmax_cross_entropy(
                    labels=self.labels_,
                    logits=self.logits_,
                )

            with tf.name_scope('training'):
                self.optimizer_ = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_ = self.optimizer_.minimize(self.loss_)

                with tf.control_dependencies([self.train_]):
                    self.train_loss_ = tf.losses.sparse_softmax_cross_entropy(
                        labels=self.labels_,
                        logits=self.logits_,
                    )

            with tf.name_scope('predict-proba'):
                self.proba_ = tf.nn.softmax(self.logits_)

            with tf.name_scope('predict-class'):
                self.class_ = tf.argmax(self.logits_, dimension=-1)

            self.variables = (
                set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) |
                set(tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
            ) - self.variables

    def predict(self, inputs, session=None):
        if session is None:
            session = tf.get_default_session()

        return session.run(self.class_, {self.inputs_: inputs['inputs'], self.lengths_: inputs['lengths']})

    def predict_proba(self, inputs, session=None):
        if session is None:
            session = tf.get_default_session()

        return session.run(self.proba_, {self.inputs_: inputs['inputs'], self.lengths_: inputs['lengths']})

    def fit_partial(self, inputs, labels, session=None):
        if session is None:
            session = tf.get_default_session()

        return session.run(self.train_loss_, {
            self.inputs_: inputs['inputs'],
            self.lengths_: inputs['lengths'],
            self.labels_: labels,
        })


def predict_logits(inputs, sequence_length, kernel_size=7, dilation_rate=2, n_hidden=64, n_layers=2):
    x = inputs

    for layer in range(n_layers):
        with tf.name_scope(f'dense_convolution_layer_{layer + 1}'):
            x = padded_conv1d(
                x, n_hidden,
                kernel_size=kernel_size,
                dilation_rate=1,
            )
            x = tf.nn.relu(x)

        with tf.name_scope(f'dilated_convolution_layer_{layer + 1}'):
            x = padded_conv1d(
                x, n_hidden,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
            )
            x = tf.nn.relu(x)

    with tf.name_scope('gru-pooling-layer'):
        _, x = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(n_hidden),
            x,
            sequence_length=sequence_length,
            dtype=tf.float32,
        )

    with tf.name_scope('dense-layer'):
        x = tf.layers.dense(x, 4)

    return x


def padded_conv1d(inputs, filters, kernel_size, dilation_rate=1, strides=1, **kwargs):
    assert strides == 1
    padding = dilation_rate * (kernel_size - 1)
    padding = tf.constant([[0, 0], [padding, 0], [0, 0]])

    return tf.layers.conv1d(
        tf.pad(inputs, padding),
        filters,
        kernel_size,
        dilation_rate=dilation_rate,
        strides=strides,
        **kwargs
    )
