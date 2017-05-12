import tensorflow as tf
# import pandas as pd
# import os
#
# import Metrics


NUM_TEST_SAMPLES = 1000


class GAN:
    """
    Adversarial training of a generative model.

    USAGE NOTE:
    Since the discriminator and generator are interlaced (the generator optimizes the output / loss of D (D2), I needed
    to separate the initialization process into two (so first create a graph and then create the loss and optimize it).
    So a proper calling to the network follows along these lines:
        G = Generator(input_dim, minibatch_size, var_scope_name="G")
        D = Discriminator(input_dim, minibatch_size, G.G, var_scope_name="D")
        D.initialize_graph()        # since D.loss and D.D2 must be initialized before G final initialization.
        G.initialize_graph(D=D)     # uses either D.D2 or D.loss.
    """
    def __init__(self, samples_distribution, generator_distribution,
                 discriminator_nn, generator_nn, graph=None,
                 training_steps=1200, minibatch_size=1, d_step_ratio=1, D_pre=None,
                 id_num=None, global_step=None, tracker=None, animate_dir=None):
        """
        initialize the GAN model.
        :param samples_distribution: Distribution object to sample true data.
        :type samples_distribution: Distribution.
        :param generator_distribution: Distribution object to generate false samples.
        :type generator_distribution: Distribution.
        :param discriminator_nn: A discriminator neural net agent.
        :param generator_nn: A tensorflow object - generator neural net agent.
        :param graph: The tensorflow computational graph object.
        :param training_steps: duration of training in iterations.
        :type training_steps: int
        :param minibatch_size: size of minibatch. None or 1 act as no batch learning. value >= 2 will be the batch size.
        :type minibatch_size: int
        :param id_num: id number of the current game setting (architecture, training_step and other running parameters).
        :param d_step_ratio: how many training iterations of the discriminator for every iteration of generator
        :type d_step_ratio: int
        :param D_pre: Discriminator network to pre-train
        :type D_pre: Discriminator or None
        :param tracker: ProgressTracker object that will accompany the GAN object.
        :type tracker: ProgressTracker
        :param animate_dir: Directory of which to output the plots to.
        """
        self.id_num = id_num
        self.samples_distribution = samples_distribution
        self.generator_distribution = generator_distribution

        self.training_steps = training_steps if training_steps is not None else 1200
        self.minibatch_size = minibatch_size if minibatch_size is not None else 1

        self.d_step_ratio = d_step_ratio
        self.D_pre = D_pre
        self.D = discriminator_nn

        self.G = generator_nn
        self.G_freeze_training = False
        self.global_step = global_step
        self.graph = graph
        self.session = None     # this will hold the Tensorflow session of which the nets will be trained in

        # self.summary_ops = tf.summary.merge_all()   # tensorboard all the summaries in D and G
        self.pg = tracker
        self.pg.init_tf_saver()     # maybe self.pg.init_tf_saver(self)
        # self.animate_dir = animate_dir  # TODO animate plotting the progress

        # self._create_model()

    def _create_model(self):
        # create pre-train D model
        if self.D_pre is not None:
            pass
        # create D net
        # create G net
        raise NotImplementedError

    def train_model(self):
        """
        Train the GAN model
        :return: None. (updates the model within)
        """
        # initialize the current training session:
        self.session = tf.Session(graph=self.graph)

        # initialize the tensorboard writer:
        self.pg.init_tensorboard_writer(self.session)

        # initialize network:
        self.session.run(tf.global_variables_initializer())

        # Pre-train the discriminator
        if self.D_pre is not None:
            # pre_train(D_pre, session, num_pre_train_steps, samples_distribution)
            # initial_D_weights = self.session.run(D_pre.params)
            # for i, v in enumerate(self.D.params):
            #     self.session.run(v.assign(initial_D_weights[i]))
            pass    # TODO: pre-train, assign final weights to d params (self.D.params)

        # Start adversarial training:
        loss_d = None       # just to avoid "before assignment" warning in the logging part
        summ_ops = None
        loss_g = None
        for t in range(self.training_steps):
            # update discriminator:
            for dt in range(self.d_step_ratio):             # train D for d_step_ratio times for every time of G
                x = self.samples_distribution.sample(self.minibatch_size)
                z = self.generator_distribution.sample(self.minibatch_size)
                loss_d, _, summ_ops = self.session.run(fetches=[self.D.loss, self.D.opt, self.pg.summary_ops],
                                                       feed_dict={self.D.x: x.reshape((self.minibatch_size, -1)),
                                                                  self.G.z: z.reshape((self.minibatch_size, -1))})
            # update generator:
            # if not self.G_freeze_training:  # continue to train G
            if True:
                z = self.generator_distribution.sample(self.minibatch_size)
                loss_g, _ = self.session.run(fetches=[self.G.loss, self.G.opt],
                                             feed_dict={self.G.z: z.reshape((self.minibatch_size, -1))})

            # log progress and test current GAN:
            self.pg.track_log(t, loss_d, loss_g, summ_ops, self)

            # update training settings according to the current GAN
            self.G_freeze_training = self.should_freeze_G()

            # animate:

        # logging and model saving after finishing the training:
        self.pg.track_log(self.training_steps, loss_d, loss_g, summ_ops, self)
        self.pg.create_D_df()

    #
    def test_D(self, samples, session=None, pred_type="prob", cutoff=0.5, complement=False):
        return self.D.predict(samples=samples, session=session or self.session, pred_type=pred_type, cutoff=cutoff,
                              complement=complement)

    def test_G(self, samples, session=None):
        if isinstance(samples, int):    # input is int stating the number of samples to generate
            noise_samples = self.generator_distribution.sample(samples)
        else:                           # input is an ndarray containing the random seeds already
            noise_samples = samples
        return self.G.predict(seed_samples=noise_samples, session=session or self.session)

    def should_freeze_G(self):
        # do some calculation using self.pg
        return False


# #     #######     # #
# ###   Metrics   ### #
#     def _calc_Dkl(self, bin_num=100, num_samples=1000):
#         EPSILON = 10e-10
#         generated_samples = self.test_G(num_samples)    # generate samples using the current G
#         true_samples = self.samples_distribution.sample(num_samples)            # get true samples
#         # calc mutual bins for the pdf:
#         min_val = min(generated_samples.min(), true_samples.min())
#         max_val = max(generated_samples.max(), true_samples.max())
#         bins = pd.np.linspace(start=min_val, stop=max_val, num=bin_num, endpoint=True)
#         generated_pdf, _ = pd.np.histogram(generated_samples, bins=bins, density=True)
#         generated_pdf[generated_pdf == 0] = EPSILON     # to avoid division by zero
#         generated_pdf /= generated_pdf.sum()
#         true_pdf, _ = pd.np.histogram(true_samples, bins=bins, density=True)
#         true_pdf /= true_pdf.sum()
#         # Dkl = (true_pdf * pd.np.log2(true_pdf / generated_pdf)).sum()
#         Dkl = kl_entropy(true_pdf, generated_pdf, base=2)
#         return Dkl


