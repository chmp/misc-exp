"""Helpers for Bayesian Modelling.
"""
import inspect


class NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, exc, exc_type, traceback):
        pass


class Model(NoOpContext):
    """Definition of a model

    Usage::

        with bayes.Model() as model:
            @model.observe
            def _(s):
                s.x = tf.placeholder(dtype=floatx, shape=[None, 1], name='x')
                s.y = tf.placeholder(dtype=floatx, shape=[None], name='y')

            @model.define
            def _(s, lam=0.5):
                s.p.w = tf.distributions.Normal(loc=0.0, scale=1.0 / lam)
                s.p.y = tf.distributions.Normal(loc=tf.squeeze(s.x @ s.w[:, None]), scale=1.0)

            @model.inference
            def _(s):
                _, n_features = get_shape(s.x)

                s.q.w = tf.distributions.Normal(
                    loc=tf.get_variable('w_loc', shape=[n_features], dtype=floatx),
                    scale=tf.nn.softplus(tf.get_variable('w_scale', shape=[n_features], dtype=floatx)),
                )
    """
    def __init__(self):
        self._scope = {'observed': {}}
        self._observed = None
        self._definition = None
        self._inference = None
        self._built = False

    def observe(self, func):
        self._observed = func
        return func

    def define(self, func):
        self._definition = func
        return func

    def inference(self, func):
        self._inference = func
        return func

    def __getitem__(self, key):
        self._ensure_observed()

        if isinstance(key, tuple):
            return tuple(self._scope['observed'][k] for k in key)

        return self._scope['observed'][key]

    def build(self, scope=None, latent_strategy=None, ensure_loss=True):
        import tensorflow as tf

        if scope is None:
            scope = {}

        if latent_strategy is None:
            latent_strategy = sample_latent

        self._ensure_observed()

        scope = dict(self._scope, **scope)
        scope = Scope(scope, latent_strategy=latent_strategy)

        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            self._inference(scope)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            scope._scope['loss'] = self._definition(scope)

        if ensure_loss and scope._scope['loss'] is None:
            scope._scope['loss'] = _build_kl_loss(scope._scope)

        return scope.get()

    def _ensure_observed(self):
        if self._built:
            return

        if self._observed is None:
            self._built = True
            return

        with DictWrapper(self._scope['observed']) as s:
            self._observed(s)

        self._built = True


class DictWrapper(NoOpContext):
    def __init__(self, d):
        super().__setattr__('_target', d)

    def __setattr__(self, k, v):
        self._target[k] = v

    def __getattr__(self, k):
        try:
            return self._target[k]

        except KeyError:
            raise AttributeError(k)


class Scope:
    def __init__(self, scope, latent_strategy=None):
        if latent_strategy is None:
            latent_strategy = sample_latent

        self._scope = dict(scope)
        self._scope.setdefault('q', {})
        self._scope.setdefault('p', {})
        self._scope.setdefault('latent', {})
        self._latent_strategy = latent_strategy

    def __getattr__(self, key):
        scope = self._scope

        if key in scope['latent']:
            return scope['latent'][key]

        if key in scope['observed']:
            return scope['observed'][key]

        if key in scope['q']:
            self._latent_strategy(scope, key)
            return scope['latent'][key]

        raise AttributeError(key)

    def get(self):
        return self._scope

    @property
    def p(self):
        return DictWrapper(self._scope['p'])

    @property
    def q(self):
        return DictWrapper(self._scope['q'])


def build(model, *defs, latent_strategy=None):
    scope = model.build(latent_strategy=latent_strategy)

    if not defs:
        return scope

    res = []
    for f in defs:
        spec = inspect.getfullargspec(f)

        # TODO: raise error for unsupported features
        args = [_lookup_dist(scope, arg) for arg in spec.args]
        res.append(f(*args))

    if len(defs) == 1:
        return res[0]

    return tuple(res)


def _lookup_dist(scope, k):
    return scope['p'][k] if k in scope['observed'] else scope['q'][k]


def sample_prior(scope, key):
    scope['latent'][key] = scope['p'][key].sample()


def sample_latent(scope, key):
    scope['latent'][key] = scope['q'][key].sample()


def sample_latent_no_grad(scope, key):
    import tensorflow as tf

    scope['latent'][key] = tf.stop_gradient(scope['q'][key].sample())


def average_latent(scope, key):
    scope['latent'][key] = scope['q'][key].mean()


def build_reparam_loss(model):
    import tensorflow as tf

    # TODO: raise warning if non-re-parametrizable
    scope = model if isinstance(model, dict) else model.build(latent_strategy=sample_latent)
    loss = tf.reduce_mean(scope['loss'])
    return loss, loss


def build_score_loss(model, var_list=None):
    import tensorflow as tf

    scope = model if isinstance(model, dict) else model.build(latent_strategy=sample_latent_no_grad)

    if var_list is None:
        var_list = tf.trainable_variables()

    grad_q = 0
    for k, q in scope['q'].items():
        v = scope['latent'][k]
        grad_q += q.log_prob(v)

    return (
        tf.reduce_mean(scope['loss']),
        tf.reduce_mean(scope['loss'] + tf.stop_gradient(scope['loss']) * grad_q),
    )


def relax_bernoulli(p, temperature=1.0):
    """Create a relaxed sample from a Bernoulli distribution.

    :param tf.distributions.Bernoulli p:
        the bernoulli distribution from which to sample

    :param float temperature:
        the temperature used for relaxed quantities

    :returns:
        a triple of sampled variable, relaxed variable and relaxed variable
        conditioned on the non-relaxed variable.
    """
    import tensorflow as tf

    u = tf.random_uniform(tf.shape(p.probs))
    z = tf.log(p.probs / (1.0 - p.probs)) + tf.log(u / (1. - u))
    b = tf.cast(z > 0, dtype=z.dtype)
    b = tf.stop_gradient(b)

    b_relaxed = tf.sigmoid(z / temperature)

    nu = tf.random_uniform(tf.shape(b))
    nu_cond = (
        (nu * (1 - p.probs)) * (1 - b) +
        (1 - p.probs * nu) * b
    )
    z_cond = tf.log(p.probs / (1.0 - p.probs)) + tf.log(nu_cond / (1. - nu_cond))

    b_cond_relaxed = tf.sigmoid(z_cond / temperature)

    return b, b_relaxed, b_cond_relaxed


def relax_categorical(p, temperature=1.0):
    """Create a relaxed sample from a OneHotCategorical distribution.

    :param Union[tf.distributions.Mutltinomial,tf.contrib.distributions.OneHotCategorical] p:
        the categorical distribution from which to sample. If specified as a
        Multinomial, the total count has to be 1.

    :param float temperature:
        the temperature used for relaxed quantities

    :returns:
        a triple of sampled variable, relaxed variable and relaxed variable
        conditioned on the non-relaxed variable.
    """
    import tensorflow as tf

    if isinstance(p, tf.distributions.Multinomial):
        control_deps = [tf.assert_equal(p.total_count, 1.0, message='can only relax categoricals')]
        event_size = tf.shape(p.probs)[-1]

    else:
        control_deps = []
        event_size = p.event_size

    z = tf.log(p.probs) - tf.log(-tf.log(tf.random_uniform(tf.shape(p.probs))))

    b = tf.argmax(z, axis=-1)
    b = tf.one_hot(b, event_size)

    with tf.control_dependencies(control_deps):
        b = tf.stop_gradient(b)

    b_relaxed = tf.nn.softmax(z / temperature)

    alpha = (1.0 - p.probs) / p.probs
    theta_b = tf.reduce_sum(p.probs * b, keep_dims=True, axis=-1)

    u_i_exp = (1 - b)
    u_b_exp = b + (1 - b) * p.probs / theta_b

    u_b = tf.random_uniform(tf.shape(p.probs)) ** (1.0 / (1.0 + alpha))
    u_b = tf.reduce_sum(u_b * b, keep_dims=True, axis=-1)

    u_i = tf.random_uniform(tf.shape(p.probs))

    u_cond = (u_i ** u_i_exp) * (u_b ** u_b_exp)

    z_cond = tf.log(p.probs) - tf.log(-tf.log(u_cond))

    b_cond_relaxed = tf.nn.softmax(z_cond / temperature)

    return b, b_relaxed, b_cond_relaxed


def build_relax_loss(model):
    """Build the RELAX loss.

    Described in Will Grathwohl et al., "Backpropagation through the Void:
    Optimizing control variates for black-box gradient estimation", 2017,
    found at ``https://arxiv.org/abs/1711.00123``.

    :param Model model:
        the model to build the REBAR loss for

    :returns:
        a pair of loss and train loss

    """
    import tensorflow as tf

    scope = model.build(latent_strategy=relax_latent_strategy)

    scope_cond_relaxed = dict(scope, latent=scope['latent_cond_relaxed'].copy(), p={}, loss=None)
    scope_cond_relaxed = model.build(scope=scope_cond_relaxed, latent_strategy=raise_latent_strategy)

    scope_relaxed = dict(scope, latent=scope['latent_relaxed'].copy(), p={}, loss=None)
    scope_relaxed = model.build(scope=scope_relaxed, latent_strategy=raise_latent_strategy)

    grad_q = 0
    for k, q in scope['q'].items():
        v = scope['latent'][k]
        grad_q += q.log_prob(v)

    loss = (
        scope['loss'] +
        tf.stop_gradient(scope['loss'] - scope_cond_relaxed['loss']) * grad_q +
        scope_relaxed['loss'] - scope_cond_relaxed['loss']
    )

    return tf.reduce_mean(scope['loss']), tf.reduce_mean(loss)


def relax_latent_strategy(scope, key):
    import tensorflow as tf

    p = scope['q'][key]

    if isinstance(p, tf.distributions.Bernoulli):
        v, v_relaxed, v_cond_relaxed = relax_bernoulli(p)

    elif isinstance(p, tf.distributions.Multinomial):
        v, v_relaxed, v_cond_relaxed = relax_categorical(p)

    elif isinstance(p, tf.contrib.distributions.OneHotCategorical):
        v, v_relaxed, v_cond_relaxed = relax_categorical(p)

    elif isinstance(p, tf.distributions.Categorical):
        raise NotImplementedError('use Multinomial with total_count = 1 or OneHotCategorical')

    else:
        v = v_relaxed = v_cond_relaxed = p.sample()
        v = tf.stop_gradient(v)

    scope.setdefault('latent', {})[key] = v
    scope.setdefault('latent_relaxed', {})[key] = v_relaxed
    scope.setdefault('latent_cond_relaxed', {})[key] = v_cond_relaxed


def raise_latent_strategy(scope, key):
    """Raise for non-existing latent variables"""
    raise RuntimeError(f'latent variable {key} does not exit')


def _build_kl_loss(scope):
    loss = 0
    for k, p in scope['p'].items():
        if k in scope['latent']:
            v = scope['latent'][k]

        else:
            v = scope['observed'][k]

        loss += p.log_prob(v)

    for q in scope['q'].values():
        loss += q.entropy()

    return -loss
