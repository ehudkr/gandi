from sklearn import metrics
from numpy import concatenate as np_concat


class MetricsD:
    def __init__(self, anomalist, metrics_names, anomaly_base_distribution, G_n_test_samples,
                 true_loc, true_scale):
        self.anomalist = anomalist
        self.anomaly_base_distribution = anomaly_base_distribution
        self.true_loc, self.true_scale = true_loc, true_scale
        self.metrics_names = metrics_names
        self.G_n_test_samples = G_n_test_samples

    def test_D(self, t, gan, test_G_seed=None, true_test_samples=None):
        """
        Test the discriminator performance as an anomaly detector.
        :param int t: The iteration of which the training was paused to perform testing.
        :param GAN.GAN gan: The GAN object currently being trained.
        :param int test_G_seed: a rng state in order to give the same baseline to the adversarial generator producing.
        :param true_test_samples: The true data test samples to which to test the anomaly detector.
        :return List[dict]: dictionary of test name and the outcome of the test - to later be put in pd.DataFrame.
        """
        res = []
        # test real samples:
        # d_pred_complement = gan.bookkeep.iloc[-1]["D_loss"] > gan.bookkeep.iloc[-1]["G_loss"]
        test_size = true_test_samples.shape[0]
        data_pred_prob = gan.test_D(samples=true_test_samples)

        # test anomaly samples:
        for i, anomaly in enumerate(self.anomalist):
            res.append({"iter": t, "anomaly": anomaly})
            # sample anomalies and get D's output on them:
            anomaly_distribution = self.anomaly_base_distribution(mu=self.true_loc+anomaly[0], std_dev=anomaly[1])
            anom_samples = anomaly_distribution.sample(test_size)
            anom_pred_prob = gan.test_D(samples=anom_samples)

            # TODO: tensorboard test subdirectory?

            # evaluate:
            for test_name in self.metrics_names:
                if test_name == "AUC":
                    # the discriminator learned that true data is 1 and generated is 0.
                    # so for the AD 1 is true,0 is anomaly
                    fpr, tpr, _ = metrics.roc_curve(y_true=[1] * test_size + [0] * test_size,
                                                    y_score=np_concat((data_pred_prob, anom_pred_prob), axis=0))
                    auc = metrics.auc(fpr, tpr)
                    res[i].update({"FPR": fpr, "TPR": tpr, "AUC": auc})
                elif test_name == "mean_conf":
                    res[i].update({"mean_conf": anom_pred_prob.mean()})
        return res






