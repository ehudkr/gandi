import MetricsG
import MetricsD

import pandas as pd
import tensorflow as tf
import os
import logging
import time


class ProgressTracker:
    """
    Progress tracker for the GAN model. Can be evoked during training to test the GAN model (either G or D).
    """
    G_metrics_function = {"KL": MetricsG.calc_Dkl,
                          "KS": MetricsG.calc_ks,
                          "CDF_l1": MetricsG.calc_l1_cdf,
                          "Anderson": MetricsG.calc_anderson}

    def __init__(self, gan_id, train_random_seed=None, G_tests_names=None, D_tester=None,
                 n_loss_tracking=None, n_G_tracking=None, n_D_tracking=None, n_logger=None, n_tensorboard=None,
                 n_checkpoint=None,
                 G_test_samples=None, test_true_samples=None, reuse_test_samples=True,
                 log_dir_path=None, tensorboard_dir_path=None, checkpoint_dir_path=None,
                 run_signature=None):
        """
        initializing the tracker object that will accompany the training process
        :param gan_id: The GAN associated with this tracker
        :type gan_id: int or list
        :param train_random_seed: the random seed that will generate the results
        :param G_tests_names: what metrics to test on G's output
        :type G_tests_names: list or set or None
        :param D_tester: # what tests to apply on D (including as AD)
        :type D_tester: MetricD
        :param n_loss_tracking: specific checkpoints or number of steps of which to track loss
        :type n_loss_tracking: int or list or None
        :param n_G_tracking: specific checkpoints or number of steps of which to track G evolvement
        :type n_G_tracking: int or list or None
        :param n_D_tracking: specific checkpoints or number of steps of which to track D evolvement
        :type n_D_tracking: int or list or None
        :param n_logger: specific checkpoints or number of steps of which to manually log progress
        :type n_logger: int or list or None
        :param n_tensorboard: specific checkpoints or number of steps of which to update tensorboard
        :type n_tensorboard: int or list or None
        :param G_test_samples: either number of samples to output from G for test, or the noise seed to input to G
        :param test_true_samples: either number of samples to true-generate for test, or the true samples themselves
        :param reuse_test_samples: should tracker reuse same G seed and true data or generate new ones every time
        :param tensorboard_dir_path: directory to create tensorboard summary to
        :type tensorboard_dir_path: str or None
        """
        self.gan_id = gan_id
        if run_signature is None:
            self.gan_signature = self.create_gan_id_signature(gan_id, train_random_seed)
        else:
            self.gan_signature = run_signature.format(gan_id)

        self.G_tests_names = G_tests_names if not None else []
        self.D_tests_names = D_tester if not None else []

        # tracking the losses over training steps:
        self.loss_tracking = pd.DataFrame(columns=["D_loss", "G_loss"],  dtype=pd.np.float)
        self.n_loss_tracking = n_loss_tracking

        # tracking the metrics tests on G (and another column for collecting the samples generated as we go)
        self.G_tracking = pd.DataFrame(columns=G_tests_names + ["samples"])
        self.n_G_tracking = n_G_tracking

        # tracking the evolvement of D (as AD as well): add anomaly + iteration columns as well. make multi index later
        self.D_tracking = pd.DataFrame(columns=["iter", "anomaly"] + D_tester.metrics_names)
        self.D_tracking_list = []   # append on-the-fly using list of dicts, and then append all at once
        # self.D_tracking = pd.DataFrame(index=pd.MultiIndex(levels=[[]] * 2, labels=[[]] * 2, names=("t", "anomaly")),
        #                                columns=D_tester.metrics_names, dtype=pd.np.float)
        if "AUC" in D_tester.metrics_names:          # if AUC is a metric, calculate FPR and TPR as well.
            self.D_tracking["FPR"] = None
            self.D_tracking["TPR"] = None
        self.D_tester = D_tester
        self.n_D_tracking = n_D_tracking

        self.logger, self.LOG_FILENAME = self.init_logger(log_dir_path)  # create logger object
        self.n_logger = n_logger
        # self.LOG_FILENAME = log_dir_path

        self.TENSORBOARD_DIR = os.path.join(tensorboard_dir_path, self.gan_signature)   # path to use summarizing to
        #                                                                               # tensorboard
        self.n_tensorboard = n_tensorboard           # tensorboard summarize every n steps
        self.tensorboard_writer = None              # a writer object of tensorboard - will be initialized in future
        # self.summary_ops = None                      # tensorboard all the summaries in D and G - will be init in fut
        self.summary_ops = None  # tensorboard all the summaries in D and G - will be init in fut

        self.CHKPT_DIR = os.path.join(checkpoint_dir_path, self.gan_signature)
        self.tf_saver = None                        # a saver object of tensorflow - will be initialized in future
        self.n_tf_saver = n_checkpoint

        self.G_test_samples = G_test_samples
        self.test_true_samples = test_true_samples
        self.reuse_test_samples = reuse_test_samples

    # ############## #
    # ## Helpers: ## #
    # ############## #
    @staticmethod
    def _should_track(cur_iter, step_to_log):       # checkpoints of which to track
        if isinstance(step_to_log, list):
            return cur_iter in step_to_log
        if isinstance(step_to_log, int):            # track every int number of training
            return cur_iter % step_to_log == 0
        # probably given None, so do not track:
        return False

    @staticmethod
    def create_gan_id_signature(gan_id, seed):
        """
        creare a unique signature for the current run
        :param int gan_id: the number of setting configuration of the current model.
        :param int|np.ndarray seed: the rng state used before the initialization
        :return str: A unique signature of the current run based on the time, configuration setting and the random state
        """
        localtime = time.localtime(time.time())
        localtime = "{Y}-{M:02}-{D:02}_{h:02}-{m:02}-{s:02}".format(Y=localtime.tm_year, M=localtime.tm_mon,
                                                                    D=localtime.tm_mday, h=localtime.tm_hour,
                                                                    m=localtime.tm_min, s=localtime.tm_sec)
        seed_str = seed if seed is type(int) else "-".join([str(x) for x in seed])
        return "_".join(["GAN", localtime, seed_str, str(gan_id)])

    # ################# #
    # ## Manual Log: ## #
    # ################# #
    def init_logger(self, log_dir=None):
        logger_file_name = "log" + self.gan_signature
        logger = logging.getLogger(name=logger_file_name)
        logger.addHandler(logging.FileHandler(os.path.join(log_dir, logger_file_name + ".log")))
        return logger, logger_file_name

    def should_log(self, t):
        return self._should_track(t, self.n_logger)

    def log_training(self, t, D_loss, G_loss, gan):
        if not self.should_log(t):
            return
        self.logger.error("GAN {ID} training\t{t:06d}:\t{D_loss}\t{G_loss}\t{time}".format(ID=gan.id_num, t=t,
                                                                                           D_loss=D_loss,
                                                                                           G_loss=G_loss,
                                                                                           time=time.asctime()))

    # ################### #
    # ## Track Losses: ## #
    # ################### #
    def should_track_loss(self, t):
        return self._should_track(t, self.n_loss_tracking)

    def track_losses(self, t, D_loss, G_loss):
        if not self.should_track_loss(t):
            return
        self.loss_tracking.loc[t] = {"D_loss": D_loss, "G_loss": G_loss}

    # ######################## #
    # ## Track Tensorboard: ## #
    # ######################## #
    def init_tensorboard_summary(self):
        self.summary_ops = tf.summary.merge_all()

    def init_tensorboard_writer(self, session):
        if self.n_tensorboard is None:
            return
        # self.tensorboard_writer = tf.summary.FileWriter(os.path.join(self.TENSORBOARD_DIR, "train"), session.graph)
        self.tensorboard_writer = tf.summary.FileWriter(self.TENSORBOARD_DIR, session.graph)

    def should_tensorboard(self, t):
        return self._should_track(t, self.n_tensorboard) and self.tensorboard_writer is not None

    def track_tensorboard(self, t, global_step, summary_ops):
        if not self.should_tensorboard(t):
            return
        self.tensorboard_writer.add_summary(summary=summary_ops, global_step=global_step)

    # ####################################### #
    # ## Track Model Checkpoints Progress: ## #
    # ####################################### #
    def init_tf_saver(self, gan=None):
        if self.n_tf_saver is None:
            self.tf_saver = None
        else:
            self.tf_saver = tf.train.Saver(var_list=None,  # maybe explicitly save: [gan.D.params, gan.G.params]
                                           name=self.gan_signature)

    def should_checkpoint(self, t):
        return self._should_track(t, self.n_tf_saver) and self.tf_saver is not None

    def create_tf_checkpoint(self, t, global_step, gan):
        if not self.should_checkpoint(t):
            return
        self.tf_saver.save(sess=gan.session, save_path=self.CHKPT_DIR + ".ckpt",
                           global_step=global_step, latest_filename=self.gan_signature)

    # ########### ## #
    # ## Track G: ## #
    # ########### ## #
    def should_track_G(self, t):
        return self._should_track(t, self.n_G_tracking) and self.G_tests_names

    def track_G(self, t, gan, test_seed=None, test_true_samples=None):
        if not self.should_track_G(t):
            return
        test_seed = test_seed if test_seed is not None else self.G_test_samples
        # generate samples using current G:
        generated_samples = gan.test_G(test_seed).reshape((-1, gan.G.output_dim))
        # either use the given test_true_samples or generate true samples:
        test_true_samples = test_true_samples if test_true_samples is not None else \
                            gan.samples_distribution.sample(test_seed).reshape(-1, gan.D.input_dim)
        results = {"samples": generated_samples}
        # noinspection PyTypeChecker
        for test_name in self.G_tests_names:
            metric = self.G_metrics_function.get(test_name)
            if "KS" in test_name:
                metric = self.G_metrics_function.get("KS")
                statistic, pval = metric(test_true_samples.flatten(), generated_samples.flatten())
                results["KS_s"] = statistic
                results["KS_p"] = pval
            elif test_name == "Anderson":
                anderson, _, _ = metric(generated_samples.flatten())
                results["Anderson"] = anderson
            else:
                res = metric(test_true_samples, generated_samples)
                results[test_name] = res
        self.G_tracking.loc[t] = results

    # ########### ## #
    # ## Track D: ## #
    # ########### ## #
    def should_track_D(self, t):
        return self._should_track(t, self.n_D_tracking) and self.D_tests_names

    def track_D(self, t, gan, test_seed=None, test_true_samples=None):
        if not self.should_track_D(t):
            return
        ad_over_time = self.D_tester.test_D(t, gan, true_test_samples=test_true_samples)
        # self.D_tracking_list.append(ad_over_time)
        self.D_tracking_list += ad_over_time

    def create_D_df(self):
        """
        This function creates a Dataframe out of the measurements gathered in the list self.D_tracking_list in order to
        ease the manipulations/plotting afterward.
        Measurements are first gathered in list and then combined once needed into df to avoid appending Dataframe rows
        on the fly, which is very inefficient.
        :return:
        """
        self.D_tracking = pd.DataFrame(self.D_tracking_list).set_index(["iter", "anomaly"])

    # ## Main Manager Function: ## #
    def track_log(self, t, D_loss, G_loss, summary_ops, gan, n_test_samples=None):

        # ResStruct = namedtuple('ResStruct', 'global_step G_freeze_training')

        global_step = tf.train.get_global_step(graph=gan.graph).eval(session=gan.session)

        # manual log:
        self.log_training(t, D_loss, G_loss, gan)

        # tensorboard:
        self.track_tensorboard(t, global_step, summary_ops)

        # checkpoint tf saver:
        self.create_tf_checkpoint(t, global_step, gan)

        # track losses:
        self.track_losses(t, D_loss, G_loss)

        G_test_samples, test_true_samples = self.get_samples_for_testing(t, gan, n_test_samples)

        # track G:
        self.track_G(t, gan, test_seed=G_test_samples, test_true_samples=test_true_samples)

        # track D:
        self.track_D(t, gan, test_seed=G_test_samples, test_true_samples=test_true_samples)

        # result = ResStruct(global_step=global_step, G_freeze_training=False)
        # return result
        return

    def get_samples_for_testing(self, t, gan, n_test_samples):
        if not (self.should_track_G(t) or self.should_track_D(t)):
            # for efficiency: if none of G or D needs testing this iteration, there is no reason to generate the samples
            # (in case self.reuse_test_samples is false we want to avoid generating thousands of samples for nothing)
            return None, None

        if self.reuse_test_samples:
            test_true_samples = self.test_true_samples
            G_test_samples = self.G_test_samples
        else:
            n_test_samples = n_test_samples or self.D_tester.G_n_test_samples
            test_true_samples = gan.samples_distribution.sample(n_test_samples).reshape(-1, gan.D.input_dim)
            G_test_samples = gan.generator_distribution.sample(n_test_samples).reshape(-1, gan.G.input_dim)

        return G_test_samples, test_true_samples



