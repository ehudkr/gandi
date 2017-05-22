
# #################################### #
# Variables to run the Playground with
# #################################### #


# network variables:
G_input_dim = 1
G_output_dim = 1
D_input_dim = G_output_dim
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
                9: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 160001, "D:G_training_steps_ratio": 1,
                    "minibatch_size": 12, "D_pre_train": False,
                    "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                # 10: {"d_arch_num": 1, "g_arch_num": 2, "training_steps": 100001, "D:G_training_steps_ratio": 1,
                #      "minibatch_size": 12, "D_pre_train": False,
                #      "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                10: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 160001, "D:G_training_steps_ratio": 3,
                     "minibatch_size": 12, "D_pre_train": False,
                     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                12: {"d_arch_num": 1, "g_arch_num": 2, "training_steps": 160001, "D:G_training_steps_ratio": 1,
                     "minibatch_size": 12, "D_pre_train": False,
                     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                11: {"d_arch_num": 1, "g_arch_num": 3, "training_steps": 160001, "D:G_training_steps_ratio": 6,
                     "minibatch_size": 12, "D_pre_train": False,
                     "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                # 100: {"d_arch_num": 1, "g_arch_num": 1, "training_steps": 1, "D:G_training_steps_ratio": 1,
                #       "minibatch_size": 12, "D_pre_train": False,
                #       "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                # 101: {"d_arch_num": 1, "g_arch_num": 2, "training_steps": 30, "D:G_training_steps_ratio": 1,
                #       "minibatch_size": 12, "D_pre_train": False,
                #       "G_loss_type": "cross_entropy", "D_loss_type": "cross_entropy"},
                }
# anomalist is in (mu, std_dev) format
anomalist = [(0.1, 1), (0.5, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (10, 1),
             (-0.1, 1), (-0.5, 1), (-1, 1), (-2, 1), (-3, 1), (-4, 1), (-5, 1), (-10, 1)]
plot_checkpoints = [0, 50, 100, 150, 200, 500, 750, 1000, 1500, 2000, 3500, 5000, 7500,
                    10000, 17500, 25000, 37500, 50000, 62500, 75000, 87500, 100000,
                    120000, 140000, 16000]