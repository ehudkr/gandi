import tensorflow as tf
from tensorflow.contrib import layers

# _activation_types = {"tanh": tf.nn.tanh,
#                      "sigmoid": tf.nn.sigmoid,
#                      "relu": tf.nn.relu}


def create_linear_layer(x, layer_dim, weights_initialize="xavier", scope_name=None):
    """
    Creates and return one fully-connected linear layer of the neural net.
    :param x: layer's input.
    :param layer_dim: the current size (dimension) of the hidden layer.
    :param weights_initialize: Type of weights initialization (Xavier or Normal)
    :param scope_name: Name to use in tf.name_scope
    :return: Tensor which is the multiplication of the input with the weights and bias addition.
    """
    input_dim = x.get_shape()[1]
    # Choose initializer for the weights:
    if weights_initialize == "normal":
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=(1.0 / tf.sqrt(input_dim)))
    else:  # xavier as default
        w_initializer = layers.xavier_initializer()

    # Create the graph layer:
    with tf.name_scope(scope_name or "hidden"):
        W = tf.get_variable(name=scope_name+"weights", shape=[input_dim, layer_dim],
                            initializer=w_initializer)
        b = tf.get_variable(name=scope_name+"biases", shape=[layer_dim],
                            initializer=tf.constant_initializer(value=0))
        Wxb = tf.matmul(x, W) + b
        return Wxb


def create_minibatch_feature_layer(x, num_kernel=5, kernel_dim=3, scope_name=None):
    """
    In the paper "Improved Techniques for Training GANs" (https://arxiv.org/abs/1606.03498), the authors present a
    technique to avoid mode-collapse while training called minibatch-discrimination (section 3.2). This method is
    implemented here - concatenating the differences of the previous layer in the batch to the current layer input.
    :param x: layer's input.
    :param num_kernel:
    :param kernel_dim:
    :param scope_name: Name to use in tf.name_scope
    :return: hidden layer - the concatenation of the input and the minibatch features calculated.
    """
    h = create_linear_layer(scope_name=(scope_name or "minibatch"),
                            x=x, layer_dim=(num_kernel * kernel_dim))
    h = tf.reshape(h, shape=[-1, num_kernel, kernel_dim])
    diffs = tf.expand_dims(h, axis=3) - tf.expand_dims(tf.transpose(h, perm=[1, 2, 0]), axis=0)
    diffs = tf.abs(diffs)
    diff = tf.reduce_sum(diffs, axis=2)
    diff = tf.exp(-diff)
    minibatch_features = tf.reduce_sum(diff, axis=2)
    return tf.concat(values=[x, minibatch_features], axis=1)


def create_optimizer(loss, var_list, learning_rate, minimize=True, decay=None, global_step=tf.Variable(0),
                     **opt_kwargs):
    """
    create the optimizer for some neural net
    :param loss: loss function to minimize/maximize
    :param var_list: Collection of trainable variables (tf.GraphKeys.TRAINABLE_VARIABLES) in some scope.
    :param learning_rate: either a number or a tensor (i.e. represents a decay in rate) to plug in
                          tf.train.<SOME OPTIMIZER>
    :param minimize: either to minimize or to maximize the cost function.
    :param decay: if learning_rate is a number, it can be used as initial learning rate and be exponentially decayed.
    :param global_step: counter variable used in the decay calculation.
    :param opt_kwargs: arguments to the optimizer. Should have "opt_type" key to choose between GradientDescent ("gd"),
                       Momentum ("momentum") and Adam ("adam"). and in addition, the relevant arguments to the type
                       chosen. For example:
                       if opt_type == "momentum", than opt_kwargs can contain "momentum" and "use_nestrov" keys that
                       matches the parameters of tf.train.MomentumOptimizer. if not specified, default values are used.
    :return: optimizer object.
    """
    if decay == "exp":
        learning_rate = create_exp_decay_learning_rate(learning_rate, global_step=global_step)

    opt_type = opt_kwargs.get("opt_type", "adam")
    if opt_type == "gd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opt_type == "momentum":
        momentum = opt_kwargs.get("momentum", 0.95)
        use_nestrov = opt_kwargs.get("use_nestrov", True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=use_nestrov)
    else:       # default is adam
        learning_rate = learning_rate if learning_rate is not None else opt_kwargs.get("adam_lr", 0.001)
        beta1 = opt_kwargs.get("beta1", 0.9)
        beta2 = opt_kwargs.get("beta2", 0.999)
        epsilon = opt_kwargs.get("epsilon", 1e-08)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1, beta2=beta2, epsilon=epsilon)
    if minimize:
        optimizer = optimizer.minimize(loss=loss, global_step=global_step, var_list=var_list)
    else:
        optimizer = optimizer.minimize(loss=-loss, global_step=global_step, var_list=var_list)
    return optimizer


def create_exp_decay_learning_rate(initial_learning_rate, global_step=tf.Variable(0),
                                   decay_steps=150, decay_rate=0.95, staircase=False):
    """
    Create a tensor representing the learning rates using exponential decay.
    Following the formula: learning_rate * decay_rate ** (global_step / decay_steps)
    :param initial_learning_rate: learning rate to begin with
    :param global_step: a counter. used in the decay calculation.
    :param decay_steps: how many steps between each decay (i.e. drop in learning rate)
    :param decay_rate: A scalar. decay rate.
    :param staircase: if True, the decrease will perform in discrete intervals (i.e. global_step / decay_steps
                      calculated as integer rather than float).
    :return: A Tensor - the decayed learning rate.
    """
    learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=decay_rate,
                                               staircase=staircase)
    return learning_rate

