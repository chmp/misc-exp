import collections

_graph_ids = collections.defaultdict(lambda: collections.defaultdict(int))


def factorized(inputs, units, activation=None, use_bias=True, trainable=True, name='factorized'):
    """Factorization machine kernel modeled after ``tf.layers.dense``.
    """
    import tensorflow as tf

    unique_name = _unique_name(name)

    with tf.variable_scope(unique_name, reuse=tf.AUTO_REUSE):
        if use_bias:
            bias = tf.get_variable(
                'bias',
                shape=[units], dtype=tf.float32, initializer=tf.zeros_initializer(),
                trainable=trainable,
            )

        else:
            bias = 0

        filters = tf.get_variable(
            'filters',
            shape=[inputs.get_shape()[-1], units], dtype=tf.float32, trainable=trainable,
        )

    result = tf.matmul(inputs, filters) ** 2.0 - tf.matmul(inputs ** 2.0, filters ** 2.0)
    result = bias + 0.5 * result

    if activation is not None:
        result = activation(result)

    return result


def _unique_name(name):
    import tensorflow as tf

    g = tf.get_default_graph()
    idx = _graph_ids[g][name]
    _graph_ids[g][name] = idx + 1
    return f'{name}_{idx}'
