import tensorflow as tf

import NNbuilds


def architecture_1(net_input, output_dim):
    """
    Possible architecture for a generator.
    :param net_input: the input to the neural network.
    :param output_dim: the size of the output.
    :return: the last layer (output layer) of the network.
    """
    h1_size = 5
    output_dim = output_dim

    h1 = NNbuilds.create_linear_layer(x=net_input, layer_dim=h1_size, scope_name="G_hidden1")
    h1 = tf.nn.softplus(h1)

    o = NNbuilds.create_linear_layer(x=h1, layer_dim=output_dim, scope_name="G_output")
    tf.summary.histogram(name="G_output_distribution", values=o)
    return o


def architecture_2(net_input, output_dim):
    """
    Possible architecture for a generator.
    :param net_input: the input to the neural network.
    :param output_dim: the size of the output.
    :return: the last layer (output layer) of the network.
    """
    h1_size = 10
    h2_size = 10
    h3_size = 5
    output_dim = output_dim

    h1 = NNbuilds.create_linear_layer(x=net_input, layer_dim=h1_size, scope_name="G_hidden1")
    h1 = tf.nn.softplus(h1)

    h2 = NNbuilds.create_linear_layer(x=h1, layer_dim=h2_size, scope_name="G_hidden2")
    h2 = tf.nn.softplus(h2)

    h3 = NNbuilds.create_linear_layer(x=h2, layer_dim=h3_size, scope_name="G_hidden3")
    h3 = tf.nn.softplus(h3)

    o = NNbuilds.create_linear_layer(x=h3, layer_dim=output_dim, scope_name="G_output")
    tf.summary.histogram(name="G_output_distribution", values=o)

    return o


def architecture_3(net_input, output_dim):
    """
    Possible architecture for a generator.
    :param net_input: the input to the neural network.
    :param output_dim: the size of the output.
    :return: the last layer (output layer) of the network.
    """
    h1_size = 25
    h2_size = 25
    h3_size = 20
    h4_size = 10
    output_dim = output_dim

    h1 = NNbuilds.create_linear_layer(x=net_input, layer_dim=h1_size, scope_name="G_hidden1")
    h1 = tf.nn.softplus(h1)

    h2 = NNbuilds.create_linear_layer(x=h1, layer_dim=h2_size, scope_name="G_hidden2")
    h2 = tf.nn.softplus(h2)

    h3 = NNbuilds.create_linear_layer(x=h2, layer_dim=h3_size, scope_name="G_hidden3")
    h3 = tf.nn.softplus(h3)

    h4 = NNbuilds.create_linear_layer(x=h3, layer_dim=h4_size, scope_name="G_hidden4")
    h4 = tf.nn.softplus(h4)

    o = NNbuilds.create_linear_layer(x=h4, layer_dim=output_dim, scope_name="G_output")
    tf.summary.histogram(name="G_output_distribution", values=o)

    return o


def _inverse_sigmoid(x):
    """
    simple calculation of the logit function - the inverse for sigmoid
    :param x: variable or value - tensorflow network output.
    :return: the logit value or operation of x
    """
    return tf.log(x / (1 - x))


def _inverse_tanh(x):
    """
    simple calculation of the tanh^-1 function - the inverse for tanh
    :param x: variable or value - tensorflow network output.
    :return: the tanh^-1 value or operation of x
    """
    return 0.5 * tf.log((1 + x) / (1 - x))


class Generator:
    """
    Generator is a neural network that tries to take noise as input pass them through the network and try to output
    samples that looks as if they were sampled from the true distribution.
    """
    _arch_types = {"architecture_1": architecture_1,
                   "architecture_2": architecture_2,
                   "architecture_3": architecture_3}

    def __init__(self, input_dim, minibatch_size, output_dim, arch_num=1, var_scope_name="G"):
        # set object's variables:
        self.var_scope_name = var_scope_name
        self.output_dim = output_dim
        self.minibatch_size = minibatch_size
        self.input_dim = input_dim

        # build the graph according to the given architecture_num.
        with tf.variable_scope(var_scope_name):
            arch_type = Generator._arch_types.get("architecture_{}".format(arch_num))

            self.z = tf.placeholder(tf.float32, shape=[None, input_dim], name=var_scope_name+"_noise_input")
            self.G = arch_type(self.z, output_dim=output_dim)

        # declare but keep empty until the initialize_graph is called. see it's documentation as for why.
        self.loss = None
        self.params = None
        self.opt = None

    def initialize_graph(self, D, loss_type, global_step=None):
        """
        Since the discriminator and generator are interlaced (the generator optimizes the output / loss of D (D2), I
        needed to separate the initialization process into two (so first create a graph and then create the loss and
        optimize it).
        So a proper calling to the network follows along these lines:
        G = Generator(input_dim, minibatch_size, var_scope_name="G")
        D = Discriminator(input_dim, minibatch_size, G.G, var_scope_name="D")
        D.initialize_graph()        # since D.loss and D.D2 must be initialized before G final initialization.
        G.initialize_graph(D=D)     # uses either D.D2 or D.loss.
        :param D: the discriminator adversary to G.
               D's output/loss sets the loss for the generator net in this game.
        :param loss_type: type of cost function to use for the generator.
        :param global_step: A global Tensor object parameter counting the number of operations in the total process.
        :return: None
        """
        with tf.variable_scope(self.var_scope_name):
            if loss_type == "minimax":                          # minimax game
                self.loss = tf.reduce_mean(-D.loss)                 # FIXME: crashes the process: InvalidArgumentError
            elif loss_type == "non_sat_heu":                    # Non-saturating game heuristic
                self.loss = tf.reduce_mean(-tf.log(D.D2))
            elif loss_type == "cross_entropy":
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D.D2_logit,
                                                                                   labels=tf.ones_like(D.D2_logit)))
            elif loss_type == "max_like":                       # Maximum likelihood inverse depends on D cost
                if "sigmoid" in D.loss.name:
                    self.loss = tf.reduce_mean(-tf.exp(_inverse_sigmoid(D.D2)))
                elif "tanh" in D.loss.name:
                    self.loss = tf.reduce_mean(-tf.exp(_inverse_tanh(D.D2)))
                else:
                    raise ValueError("""Make sure the discriminator architecture has name argument in it's final
                                        activation layer that said it is either 'sigmoid_activation' or
                                        'tanh_activation'. You stated {}""".format(D.loss.name))
            else:
                raise ValueError("cost function type {} not supported".format(loss_type))
            self.params = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.var_scope_name)
            global_step = global_step or tf.Variable(0)
            self.opt = NNbuilds.create_optimizer(loss=self.loss, var_list=self.params, learning_rate=None,
                                                 minimize=True, global_step=global_step, opt_type="adam", opt_kwargs={})
            tf.summary.scalar(name="cost_{}".format(loss_type), tensor=self.loss)

    def predict(self, seed_samples, session):
        """
        use the generator to predict the values of a given sample (i.e. apply the generator neural net function on the
        samples). depending on the size of the sample and the size of the minibatch trained, not all samples may be
        processed.
        :param seed_samples: the samples to evaluate
        :param session: the current GAN session used to train the discriminator and the generator.
        :return: prediction vector of size num_samples corresponding to the given samples.
        """
        pred = session.run(fetches=self.G, feed_dict={self.z: seed_samples.reshape(-1, 1)})
        return pred





#
# ##########
#
    #   PREDICT BY BATCH
        # pred = np.zeros(shape=(num_samples, self.output_dim), dtype=np.float32)
        # end_idx = 0
        # for i in range(num_samples // self.minibatch_size):
        #     start_idx = self.minibatch_size * i
        #     end_idx = self.minibatch_size * (i + 1)
        #     batch_samples = seed_samples[start_idx: end_idx].reshape([self.minibatch_size, self.input_dim])
        #     batch_pred = session.run(fetches=self.G, feed_dict={self.z: batch_samples})
        #     pred[start_idx: end_idx] = batch_pred
        # # deal with the leftover examples that didn't fit into the minibatch size:
        # batch_samples = seed_samples[end_idx:].reshape([num_samples - end_idx, self.input_dim])
        # batch_pred = session.run(fetches=self.G, feed_dict={self.z: batch_samples})
        # pred[end_idx:] = batch_pred
        # return pred

