import tensorflow as tf
# from tensorflow.contrib import layers

import NNbuilds


def architecture_1(net_input, minibatch_size):
    """
    Possible architecture for a discriminator.
    :param net_input: the input to the neural network.
    :param minibatch_size: the size of the minibatch.
    :return: the last layer (output layer) of the network after and before activation (sigmoid squeeze).
    """
    h1_size = 5
    h2_size = 5

    # Hidden 1 layer:
    h1 = NNbuilds.create_linear_layer(x=net_input, layer_dim=h1_size,
                                      scope_name="D_hidden1")
    h1 = tf.nn.tanh(h1)
    # Minibatch features layer or regular hidden layer:
    if minibatch_size > 1:
        h2 = NNbuilds.create_minibatch_feature_layer(x=h1, scope_name="D_minibatch2")
    else:
        h2 = NNbuilds.create_linear_layer(x=h1, layer_dim=h2_size,
                                          scope_name="D_hidden2")
    # output layer:
    o_logit = NNbuilds.create_linear_layer(x=h2, layer_dim=Discriminator.output_dim, scope_name="D_output")
    o = tf.nn.sigmoid(o_logit, name="sigmoid_activation")     # mentioning type of activation name is important if the
    #                                                         # generator is playing maximum likelihood game
    return o, o_logit


def architecture_2(net_input, input_dim, minibatch_size):
    raise NotImplementedError


class Discriminator:
    """
    Discriminator tries to discriminate between samples from the true data distribution (self.x) and the generated
    distribution (self.G i.e. it is actually G(z)).
    Hopefully prediction values closer to 1 suggest the input is real, and values closer to 0 suggest input is generated
    """
    output_dim = 1
    _arch_types = {"architecture_1": architecture_1,
                   "architecture_2": architecture_2}

    @staticmethod
    def classify_probabilities(prediction, cutoff=0.5):
        pred = prediction.copy()
        mask = pred >= cutoff
        pred[mask] = 1
        pred[~mask] = 0
        return pred

    def __init__(self, input_dim, minibatch_size, G_output, arch_num=1, var_scope_name="D"):
        # set object's variables:
        self.input_dim = input_dim
        self.minibatch_size = minibatch_size
        self.var_scope_name = var_scope_name

        # build the graph according to the given architecture_num.
        # NOTE ::: D1 inputs real data samples. D2 inputs generated samples.
        # D2 reuses D1's variables so they are basically the same net. This is because you can't use same network for
        # different inputs.
        with tf.variable_scope(var_scope_name) as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name=var_scope_name+"_sample_input")
            self.G_output = G_output

            arch_type = Discriminator._arch_types.get("architecture_{}".format(arch_num))

            self.D1, self.D1_logit = arch_type(self.x, minibatch_size)
            scope.reuse_variables()
            self.D2, self.D2_logit = arch_type(self.G_output, minibatch_size)

        # declare but keep empty until the initialize_graph is called. see it's documentation as for why.
        self.loss = None
        self.params = None
        self.opt = None

    def initialize_graph(self, loss_type, global_step=None):
        """
        Since the discriminator and generator are interlaced (the generator optimizes the output / loss of D (D2), I
        needed to separate the initialization process into two (so first create a graph and then create the loss and
        optimize it).
        So a proper calling to the network follows along these lines:
        G = Generator(input_dim, minibatch_size, var_scope_name="G")
        D = Discriminator(input_dim, minibatch_size, G.G, var_scope_name="D")
        D.initialize_graph()        # since D.loss and D.D2 must be initialized before G final initialization.
        G.initialize_graph(D=D)     # uses either D.D2 or D.loss.
        :return: None
        """
        with tf.variable_scope(self.var_scope_name):
            if loss_type == "game":
                self.loss = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
            elif loss_type == "cross_entropy":
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logit,
                                                                                   labels=tf.ones_like(self.D1_logit)))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logit,
                                                                                   labels=tf.zeros_like(self.D2_logit)))
                self.loss = real_loss + fake_loss   # "real" stands for the true data, "fake" for the generated
            else:
                raise ValueError
            self.params = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.var_scope_name)
            global_step = global_step or tf.Variable(0)
            self.opt = NNbuilds.create_optimizer(loss=self.loss, var_list=self.params, learning_rate=None,
                                                 minimize=True, global_step=global_step, opt_type="adam", opt_kwargs={})
            tf.summary.scalar(name="cost_{}".format(loss_type), tensor=self.loss)

    def predict(self, samples, session, pred_type="prob", cutoff=0.5, complement=False):
        """
        use the discriminator to predict the values of a given sample (i.e. apply the discriminator neural net function
        on the samples). depending on the size of the sample and the size of the minibatch trained, not all samples may
        be processed.
        :param samples: the samples to evaluate
        :param session: the current GAN session used to train the discriminator and the generator.
        :param pred_type: either to predict in classes (categorical) or in probabilities ("regressional").
        :param cutoff: if categorical is used, what cutoff value to use to obtain the "rounding".
        :param complement: if True, the return values are 1 minus the original prediction values (complementing to 1).
        :type complement: bool
        :return: prediction vector of size num_samples corresponding to the given samples.
        """
        pred = session.run(fetches=self.D1, feed_dict={self.x: samples.reshape(-1, self.input_dim)})

        if complement:
            pred = 1 - pred

        if pred_type == "prob":
            return pred
        elif pred == "class":
            Discriminator.classify_probabilities(pred, cutoff=cutoff)
        else:
            raise NotImplementedError("Choose a legit type of prediction")



#
# ##
# ####
# ###### LEFTOVERS ###### #
# def architecture_1(net_input, input_dim, minibatch_size):
#     h1_size = 5
#     h2_size = 5
#     output_size = 1
#
#     # hidden 1:
#     # with tf.name_scope("D_hidden1"):
#     #     # W = tf.Variable(tf.truncated_normal(shape=[input_dim, h1_size],
#     #     #                                     mean=0.0,
#     #     #                                     stddev=(1.0 / tf.sqrt(input_dim))),
#     #     #                 name="weights")
#     #     # b = tf.Variable(tf.zeros(shape=[h1_size]), name="biases")
#     #     W = tf.get_variable(name="weights", shape=[input_dim, h1_size],
#     #                         initializer=layers.xavier_initializer())
#     #     b = tf.get_variable(name="biases", shape=[h1_size],
#     #                         initializer=tf.constant_initializer(value=0))
#     #     h1 = tf.nn.tanh(tf.matmul(net_input, W) + b)
#     h1 = Discriminator.create_linear_layer(scope_name="D_hidden1",
#                                            x=net_input, input_dim=input_dim,
#                                            layer_dim=h1_size)
#     h1 = tf.nn.tanh(h1)
#
#
#     # hidden 2:
#     with tf.name_scope("D_hidden2"):
#         W = tf.get_variable(name="weights", shape=[h1_size, h2_size],
#                             initializer=layers.xavier_initializer())
#         b = tf.get_variable(name="biases", shape=[h2_size],
#                             initializer=tf.constant_initializer(value=0))
#         h2 = tf.nn.tanh(tf.matmul(h1, W) + b)
#
#     # output layer:
#     with tf.name_scope("D_output"):
#         W = tf.get_variable(name="weights", shape=[h2_size, output_size],
#                             initializer=layers.xavier_initializer())
#         b = tf.get_variable(name="biases", shape=[output_size],
#                             initializer=tf.constant_initializer(value=0))
#         o = tf.nn.sigmoid(tf.matmul(h2, W) + b)
#
#     return o

# def create_linear_layer(x, layer_dim, weights_initialize="xavier", scope_name=None):
#     input_dim = x.get_shape()[1]
#     # Choose initializer for the weights:
#     if weights_initialize == "normal":
#         w_initializer = tf.random_normal_initializer(mean=0, stddev=(1.0 / tf.sqrt(input_dim)))
#     else:  # xavier as default
#         w_initializer = layers.xavier_initializer()
#
#     # Create the graph:
#     with tf.name_scope(scope_name or "D_hidden"):
#         # W = tf.Variable(tf.truncated_normal(shape=[input_dim, h1_size],
#         #                                     mean=0.0,
#         #                                     stddev=(1.0 / tf.sqrt(input_dim))),
#         #                 name="weights")
#         # b = tf.Variable(tf.zeros(shape=[h1_size]), name="biases")
#         W = tf.get_variable(name="weights", shape=[input_dim, layer_dim],
#                             initializer=w_initializer)
#         b = tf.get_variable(name="biases", shape=[layer_dim],
#                             initializer=tf.constant_initializer(value=0))
#         Wxb = tf.matmul(x, W) + b
#         return Wxb
#         # h = Discriminator._activation_types.get(activation)(Wxb)
#         # if h is not None:
#         #     return h
#         # else:
#         #     raise NotImplementedError("Choose legit activation function")

