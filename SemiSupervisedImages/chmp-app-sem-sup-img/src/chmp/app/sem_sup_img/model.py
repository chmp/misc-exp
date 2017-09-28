"""Convolutional Auto-Encoder
"""
import itertools as it
import logging
import typing

import tensorflow as tf

from chmp.experiment import Config as _Config
from chmp.ml import PickableTFModel, inject_session

_logger = logging.getLogger(__name__)


class ConvolutionalAutoEncoder(PickableTFModel):
    @typing.no_type_check
    class Config(_Config):
        batch_size: int = 10
        epochs: int = 10

        width: int = 160
        features: int = 4096

        strides: typing.Tuple[int] = (2, 2, 2, 2)
        kernel_sizes: typing.Tuple[int] = (4, 4, 4, 4)
        encoder_filters: typing.Tuple[int] = (16, 32, 64, 64)
        decoder_filters: typing.Tuple[int] = (64, 64, 32, 16)

        @classmethod
        def check(cls, ns):
            if len(ns.strides) != len(ns.kernel_sizes):
                raise ValueError('strides and kernel lens do not match')

            if len(ns.strides) != len(ns.encoder_filters):
                raise ValueError('strides and encoder_filter lens do not match')

            if len(ns.strides) != len(ns.decoder_filters):
                raise ValueError('strides and decoder_filters lens do not match')

    def __init__(self, *, config, graph=None):
        super().__init__()
        self.config = config
        self._build(graph=graph)

    @inject_session
    def fit_partial(self, inputs, labels=None, session=None):
        return session.run(self.train_loss_, {self.input_: inputs})

    @inject_session
    def eval_loss(self, inputs, session=None):
        return session.run(self.loss_, {self.input_: inputs})

    @inject_session
    def transform(self, inputs, session=None):
        return session.run(self.encoded_, {self.input_: inputs})

    def _build(self, graph=None):
        with self.build_context(graph=graph):
            self.input_ = tf.placeholder(
                tf.float32,
                shape=(None, self.config.width, self.config.width, 1),
                name='input',
            )

            self.encoded_ = self._encode(self.input_)
            self.decoded_ = self._decode(self.encoded_)

            self.loss_ = tf.losses.mean_squared_error(self.input_, self.decoded_)

            self.optimizer_ = tf.train.AdamOptimizer()
            self.train_ = self.optimizer_.minimize(self.loss_)

            with tf.control_dependencies([self.train_]):
                self.train_loss_ = tf.identity(self.loss_)

    def _encode(self, x_):
        _logger.debug('encode')
        x_ = self._apply_stacked_convolutions(x_, 'encoder', self.config.encoder_filters, self.config)

        conv_shape = x_.shape.as_list()
        conv_shape = conv_shape[1:]
        conv_size = prod(conv_shape)

        # TODO: calc shape instead of using attribute
        self._conv_params = conv_size, conv_shape

        x_ = tf.reshape(x_, [-1, conv_size])
        return tf.layers.dense(x_, units=self.config.features, activation=tf.nn.sigmoid)

    def _decode(self, x_):
        _logger.debug('decode')
        conv_size, conv_shape = self._conv_params

        x_ = tf.layers.dense(x_, units=conv_size, activation=tf.nn.relu)
        x_ = tf.reshape(x_, [-1] + list(conv_shape))

        x_ = self._apply_stacked_convolutions(
            x_, 'decoder', self.config.decoder_filters, self.config,
            decode=True,
        )

        return tf.layers.conv2d(
            x_,
            name='decoder_final',
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=tf.nn.sigmoid,
            padding='same',
        )

    @staticmethod
    def _apply_stacked_convolutions(x_, name, filters, config, decode=False):
        layer_func = tf.layers.conv2d if not decode else tf.layers.conv2d_transpose

        _logger.debug('shape before conv stack %s', x_.shape)
        for i, nf, st, ks in zip(it.count(), filters, config.strides, config.kernel_sizes):
            x_ = layer_func(
                x_,
                name=f'{name}_{i}',
                filters=nf,
                kernel_size=(ks, ks),
                strides=(st, st),
                activation=tf.nn.relu,
                padding='same',
            )
            _logger.debug('shape after convolution %s', x_.shape)

            if decode:
                x_ = tf.layers.conv2d(
                    x_,
                    name=f'{name}_{i}_smooth',
                    filters=nf,
                    kernel_size=(st, st),
                    activation=tf.nn.relu,
                    padding='same',
                )
                _logger.debug('shape after smooth %s', x_.shape)

        _logger.debug('shap after conv stack %s', x_.shape)
        return x_


def prod(items, initial=1):
    for item in items:
        initial = initial * item
    return initial
