import numpy as np
import tensorflow as tf


default_params = dict(
    structure=[
        (32, 4, 1),
        (32, 3, 2),
        (32, 3, 1),
        (32, 3, 2),
        (32, 3, 1),
    ],
    latent_structure=[128, 64],
    max_steps=30_000,
    batch_size=20,
)


class Estimator(tf.estimator.Estimator):
    def __init__(
        self, *,
        structure=None, latent_structure=None, model_dir=None, config=None, warm_start_from=None
    ):
        if structure is None:
            structure = default_params['structure']

        if latent_structure is None:
            latent_structure  = default_params['latent_structure']

        super().__init__(
            model_fn=model_fn,
            model_dir=model_dir,
            config=config,
            warm_start_from=warm_start_from,
            params=dict(structure=structure, latent_structure=latent_structure),
        )


def model_fn(features, labels, mode, params):
    params = dict(default_params, **params)
    structure = params['structure']
    latent_structure = params['latent_structure']

    input_ = features['input']

    encoded_ = conv_encode(input_, structure)
    decoded_, latent_, (latent_mean_, latent_sigma_) = latent_bottleneck(encoded_, latent_structure)
    decoded_ = conv_decode(decoded_, structure)
    output_, (output_mean_, output_sigma_) = reconstruct(decoded_)

    _, *input_shape = input_.shape.as_list()
    _, *output_shape = output_.shape.as_list()
    assert input_shape == output_shape, f'{input_shape!r} != {output_shape!r}'

    loss_, reconstruction_loss_, prior_loss_, entropy_loss_ = build_loss(
        input_, (latent_mean_, latent_sigma_), (output_mean_, output_sigma_),
    )

    # TODO: check formula
    reconstruction_error_ = tf.reduce_mean((input_ - output_mean_) ** 2.0)
    reconstruction_error_expected_ = reconstruction_error_ + tf.reduce_mean(output_sigma_ ** 2.0)

    with tf.name_scope('losses'):
        tf.summary.scalar('reconstruction_error', reconstruction_error_)
        tf.summary.scalar('reconstruction_error_expected', reconstruction_error_expected_)

        tf.summary.scalar('reconstruction_loss', reconstruction_loss_)
        tf.summary.scalar('prior_loss', prior_loss_)
        tf.summary.scalar('entropy_loss', entropy_loss_)

    with tf.name_scope('means'):
        tf.summary.scalar('mean_latent_mean', tf.reduce_mean(latent_mean_))
        tf.summary.scalar('mean_latent_sigma', tf.reduce_mean(latent_sigma_))
        tf.summary.scalar('mean_output_mean', tf.reduce_mean(output_mean_))
        tf.summary.scalar('mean_output_sigma', tf.reduce_mean(output_sigma_))

    with tf.name_scope('histograms'):
        tf.summary.histogram('input', input_)

        tf.summary.histogram('latent', latent_)
        tf.summary.histogram('latent_mean', latent_mean_)
        tf.summary.histogram('latent_sigma', latent_sigma_)

        tf.summary.histogram('output', output_)
        tf.summary.histogram('output_mean', output_mean_)
        tf.summary.histogram('output_sigma', output_sigma_)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=dict(
            latent=latent_,
            latent_mean=latent_mean_,
            output=output_,
            output_mean=output_mean_,
        ))

    elif mode == tf.estimator.ModeKeys.EVAL:
        raise NotImplementedError()

    elif mode == tf.estimator.ModeKeys.TRAIN:
        opt_ = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
        train_ = opt_.minimize(loss_, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, train_op=train_, loss=loss_)

    else:
        raise NotImplementedError()


def conv_encode(x_, structure):
    for idx, (filters, kernel_size, stride) in enumerate(structure):
        x_ = tf.layers.conv2d(
            x_,
            filters, kernel_size, strides=(stride, stride),
            activation=tf.nn.elu,
            name=f'encode_{idx}'
        )

    return x_


def conv_decode(x_, structure):
    structure = list(enumerate(structure))

    for idx, (filters, kernel_size, stride) in structure[::-1]:
        x_ = tf.layers.conv2d_transpose(
            x_,
            filters, kernel_size, strides=(stride, stride),
            activation=tf.nn.elu,
            name=f'decode_{idx}')

        if stride != 1:
            # TODO: find better way to set the kernel size
            x_ = tf.contrib.layers.conv2d_in_plane(
                x_, 3, padding='SAME',
                # NOTE: the name argument is not supported
                # name=f'decode_smooth_{idx}',
            )

    return x_


def latent_bottleneck(x_, latent_structure):
    latent_structure = list(enumerate(latent_structure))

    _, *feature_shape = x_.shape.as_list()
    x_ = tf.reshape(x_, [-1, np.prod(feature_shape)])

    _, latent_size = latent_structure[-1]

    for idx, units in latent_structure[:-1]:
        x_ = tf.layers.dense(x_, latent_size, activation=tf.nn.elu, name=f'dense_encode_{idx}')

    latent_mean_ = tf.layers.dense(x_, latent_size, name='latent_mean')
    latent_sigma_ = tf.nn.softplus(1e-4 + tf.layers.dense(x_, latent_size, name='latent_sdev'))
    latent_ = latent_mean_ + latent_sigma_ * tf.random_normal(tf.shape(latent_mean_))

    x_ = latent_
    for idx, units in latent_structure[-2::-1]:
        x_ = tf.layers.dense(x_, units, activation=tf.nn.elu, name=f'dense_decode_{idx}')

    x_ = tf.layers.dense(latent_, np.prod(feature_shape), activation=tf.nn.elu, name='decoded')
    x_ = tf.reshape(x_, [-1, *feature_shape])

    return x_, latent_, (latent_mean_, latent_sigma_)


def reconstruct(decoded_):
    output_mean_ = tf.layers.dense(decoded_, 1, name='output_mean')
    output_sigma_ = tf.nn.softplus(1e-3 + tf.layers.dense(decoded_, 1, name='output_sdev'))

    output_ = output_mean_ + output_sigma_ * tf.random_normal(tf.shape(output_mean_))
    return output_, (output_mean_, output_sigma_)


def build_loss(input_, latent_params_, output_params_):
    latent_mean_, latent_sigma_ = latent_params_
    output_mean_, output_sigma_ = output_params_

    # mean over batch size and image size
    with tf.control_dependencies([
        tf.assert_positive(output_sigma_), tf.assert_positive(latent_sigma_)
    ]):
        reconstruction_loss_ = tf.reduce_mean(
            -0.5 * (input_ - output_mean_) ** 2.0 / output_sigma_ ** 2.0 - tf.log(output_sigma_)
        )

        # mean over  batch size + latent size
        prior_loss_ = 0.5 * tf.reduce_mean(- latent_mean_ ** 2.0 - latent_sigma_ ** 2.0)

        entropy_loss_ = 0.5 * tf.reduce_mean(1 + tf.log(latent_sigma_ ** 2.0))

    # NOTE: scale the reconstruction loss to remove the constants introduced in mean
    loss_ = (
        prior_loss_ +
        entropy_loss_ + reconstruction_loss_ *
        tf.cast(tf.size(output_mean_) / tf.size(latent_mean_), tf.float32)
    )

    # NOTE: the loss is the negative of L as defined in ...
    return -loss_, reconstruction_loss_, prior_loss_, entropy_loss_
