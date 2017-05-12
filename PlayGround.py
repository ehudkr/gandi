# Global imports:
import pandas as pd
import tensorflow as tf
import logging
import os
import time
import pickle
# import dill
from copy import deepcopy
# import random


# Local imports:
import Distributions
import Plots
import BaselineAD
import Tracker
from GAN import GAN
from DiscriminatorNN import Discriminator
from GeneratorNN import Generator
from MetricsD import MetricsD

np = pd.np
LOG_DIR = os.path.join("log_dir", "logs")
TENSBOARD_DIR = os.path.join("log_dir", "tensorboard", "GAN_{}")
PLOT_DIR = os.path.join("log_dir", "plots")
CHKPT_DIR = os.path.join("log_dir", "checkpoints", "GAN_{}")
RSLT_DIR = os.path.join("log_dir", "results")

# TODO: implement simple baseline anomaly detection method
# TODO: use a written GAN module and compare performance
# TODO: convert implementation into keras


def classify_probabilities(prediction, cutoff=0.5):
    pred = np.copy(prediction)
    mask = pred >= cutoff
    pred[mask] = 1
    pred[~mask] = 0
    return pred


# TODO: how to save the models + saving the trained tensforflow networks? save session?


def main(seed=None):
    # network variables:
    input_dim = 1
    # distribution variables:
    true_mu, true_sigma = 4, 1
    # performance measurement variables:
    train_params = {
                    # 0: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 5000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 1: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 20000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 2: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 12000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "non_sat_heu", "D_loss_type": "game"},
                    # 3: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 30000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "non_sat_heu", "D_loss_type": "game"},
                    # 4: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 15000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "non_sat_heu", "D_loss_type": "game"},
                    # 4: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 17000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "non_sat_heu", "D_loss_type": "game"},
                    # 5: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 20000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 6: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 35000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 7: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 20000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 8: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 10000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 5: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 50000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 4: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 75000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 3: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 100000, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 2: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 150001, "D:G_training_steps_ratio": 1,
                    #     "minibatch_size": 12, "D_pre_train": False,
                    #     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    9: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 100001, "D:G_training_steps_ratio": 1,
                        "minibatch_size": 12, "D_pre_train": False,
                        "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 10: {"d_arch_num": 1, "g_arch_num": 2, "training_steps": 100001, "D:G_training_steps_ratio": 1,
                    #      "minibatch_size": 12, "D_pre_train": False,
                    #      "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 100: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 4, "D:G_training_steps_ratio": 1,
                    #       "minibatch_size": 12, "D_pre_train": False,
                    #       "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    # 101: {"d_arch_num": 1, "g_arch_num": 2, "training_steps": 30, "D:G_training_steps_ratio": 1,
                    #       "minibatch_size": 12, "D_pre_train": False,
                    #       "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                    }
    # anomalist is in (mu, std_dev) format
    anomalist = [(0.1, 1), (0.5, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (10, 1),
                 (-0.1, 1), (-0.5, 1), (-1, 1), (-2, 1), (-3, 1), (-4, 1), (-5, 1), (-10, 1)]
    plot_checkpoints = [0, 50, 65, 75, 85, 95, 100, 110, 130, 150, 200, 500, 750, 1000, 1500, 2000, 3500, 5000, 7500,
                        10000, 25000, 50000, 75000, 100000]

    # ### START Playing ### #
    samples_distribution = Distributions.Distribution(dist_type="gaussian", kwargs={"mu": true_mu, "std_dev": true_sigma})
    noise_distribution = Distributions.Distribution(dist_type="uniform", kwargs={"low_bound": -8, "high_bound": 8,
                                                                                 "noise_level": 0.01})
    trackers = {}
    metricsD = {}
    gans = {}
    for p, params in train_params.items():
        # set local variables for this specific run/architecture:
        minibatch_size = params.get("minibatch_size")
        d_arch_num = params.get("d_arch_num")
        g_arch_num = params.get("g_arch_num")
        D_pre_train = params.get("D_pre_train")
        training_steps = params.get("training_steps")
        DG_training_steps_ratio = params.get("D:G_training_steps_ratio")
        G_loss_type = params.get("G_loss_type")
        D_loss_type = params.get("D_loss_type")
        test_size = minibatch_size * 500

        G_tests_names = ["KL", "KS_s", "KS_p", "CDF_l1", "Anderson"]
        D_tests_names = ["AUC"]

        # initialize a ProgressTracker object that will accompany the GAN training:
        metricD = MetricsD(anomalist=anomalist, metrics_names=D_tests_names,
                           anomaly_base_distribution=Distributions.GaussianDistribution,
                           true_loc=true_mu, true_scale=true_sigma,
                           G_n_test_samples=test_size)
        pg = Tracker.ProgressTracker(gan_id=p, G_tests_names=G_tests_names, D_tester=metricD,
                                     n_loss_tracking=10,
                                     n_G_tracking=plot_checkpoints + [i for i in range(0, training_steps, 500)],
                                     n_D_tracking=plot_checkpoints,
                                     n_logger=10000, n_tensorboard=10000, n_checkpoint=50000,
                                     G_test_samples=noise_distribution.sample(test_size).reshape(-1, input_dim),
                                     test_true_samples=samples_distribution.sample(test_size).reshape(-1, input_dim),
                                     reuse_test_samples=False,
                                     log_dir_path=LOG_DIR, tensorboard_dir_path=TENSBOARD_DIR,
                                     checkpoint_dir_path=CHKPT_DIR)

        # initialize the current network:
        #       Note:   in order to make results reproducible,
        #               this is the only way to set a "global" seed per graph every iteration.
        #               can work without, the GAN instance will get default None for graph object.
        gan_graph = tf.Graph()
        #       Use the current gan_graph instance during the initialization and training of the specific GAN setting
        with gan_graph.as_default():
            np.random.seed(seed[0])        # for reproducibility
            tf.set_random_seed(seed[1])    # for reproducibility
            D, D_pre, G, global_step = initialize_GAN(D_loss_type, D_pre_train, G_loss_type, d_arch_num, g_arch_num,
                                                      input_dim, minibatch_size, p, pg)

        # TODO: cross validation add loop for k times. update the performance dict-table as well. save those models too?
        #       maybe just cv the anomaly test batch to get mean_error and std.

            gan = GAN(samples_distribution=samples_distribution, generator_distribution=noise_distribution,
                      discriminator_nn=D, generator_nn=G, D_pre=D_pre,
                      training_steps=training_steps, minibatch_size=minibatch_size,
                      d_step_ratio=DG_training_steps_ratio,
                      graph=gan_graph, global_step=global_step,
                      id_num=p, tracker=pg,
                      # log=(200, logger),
                      # tensorboard_params=(10, TENSBOARD_DIR.format(p)),
                      # checkpoints_params=(2000, CHKPT_DIR.format(p))
                      )
            gan.train_model()       # note that D and G updated as well during training
            # TODO: maybe think of a way to supply the training samples.
            #       so you could have them in hand. How will that incorporate with the number of steps?
            #       maybe supply samples instead of the distribution. Or make Distribution interface so that it will
            #       take train size and keep track of what it generates in some dataframe.

            # ### PLOT RESULTS ### #
            plot_dir_name = os.path.join(PLOT_DIR, pg.LOG_FILENAME + "_{}_{}".format(p, seed))
            os.mkdir(plot_dir_name)
            plot_path_prefix = os.path.join(plot_dir_name, pg.LOG_FILENAME + "_{}_{}".format(p, seed))
            plotter = Plots.Plotter(setting_num=p, plot_path_prefix=plot_path_prefix, progress_tracker=pg,
                                    true_distribution=samples_distribution, anomaly_distribution=None)
            plotter.plot(plot_types=["cdf", "pdf", "qqplot", "roc_setting", "G_tests", "auc_time"],
                         iteration_checkpoints=plot_checkpoints, logx=True)

            # TODO: save gan object, tf session, graph, object internal and everything.   (and restore)
            gans[p] = gan
            trackers[p] = {"D": pg.D_tracking, "G": pg.G_tracking, "loss": pg.loss_tracking}
            metricsD[p] = metricD

        gan.session.close()
        # tf.reset_default_graph()      # cancels the graph accumalation over iterations
    # Plots.plot_auc_by_setting(res["AUC"], train_params,
    #                           save_path=os.path.join(PLOT_DIR, pg.LOG_FILENAME + "_{}".format(seed)))

    # results = {"trackers": trackers, "metricsD": metricsD, "GANs": gans}
    print("Ending time: ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(time.time())))
    pickle.dump(trackers,
                open(os.path.join(RSLT_DIR, pg.LOG_FILENAME + ".pkl"), "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

    # ### General Scheme ### #
    # choose distributions
    # create generator
    # for discriminator going and being more complex:
    # # train generator and discriminator
    # # for anomaly going and increasing:
    # # # calculate the relative detection rate of the discriminator
    # # # put it in a pandas df
    # ### Standardize input? ### #

    return trackers, metricsD, gans


def initialize_GAN(D_loss_type, D_pre_train, G_loss_type, d_arch_num, g_arch_num, input_dim, minibatch_size, p, pg):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    G = Generator(input_dim, minibatch_size, arch_num=g_arch_num, output_dim=1, var_scope_name=str(p) + "G")
    D = Discriminator(input_dim, minibatch_size, G.G, arch_num=d_arch_num, var_scope_name=str(p) + "D")
    D.initialize_graph(loss_type=D_loss_type, global_step=global_step)
    G.initialize_graph(D=D, loss_type=G_loss_type, global_step=global_step)
    if D_pre_train:
        D_pre = Discriminator(input_dim, minibatch_size, G.G, arch_num=d_arch_num, var_scope_name="D_pre")
        D_pre.initialize_graph(loss_type=D_loss_type, global_step=global_step)
    else:
        D_pre = None
    pg.init_tensorboard_summary()
    return D, D_pre, G, global_step


if __name__ == "__main__":
    try:
        tf.gfile.DeleteRecursively(os.path.split(TENSBOARD_DIR)[0])
    except Exception as e:
        print(e)
    # seed = np.random.randint(low=0, high=np.uint32(-1), size=2, dtype=np.uint32)
    seed = np.random.randint(low=0, high=999999, size=2, dtype=np.uint32)
    print(seed)
    # seed = 0
    # seed = 93440      # was a good seed I used to see schizophrenia.
    start_time = time.time()
    print("starting time: ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(start_time)))
    res = main(seed=[753442, 363223])
    print("time elapsed: ", time.time()-start_time)
    # pickle.dump(res, open(os.path.join(RSLT_DIR,
    #                                    res["trackers"][list(res["trackers"].keys())[0]].LOG_FILENAME + ".pkl"), "wb"))
