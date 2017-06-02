import matplotlib
# matplotlib.use("Agg")       # in order to generate plots without displaying them, so it could be managed on the cluster
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import os
# from numpy import sort, linspace, cumsum, exp
from statsmodels.distributions.empirical_distribution import ECDF

# from MetricsG import _calc_mutual_pdf


def plot_roc_by_setting(setting_num, data, save_path=None):
    """
    plots curves for different anomalies for one setting, over different iterations
    :param setting_num:
    :param data:
    :param save_path:
    :return:
    """
    figaxes = []

    for t, cur_data in data.groupby(level="iter"):
        cur_data = cur_data.loc[t]
        fig = plt.figure()
        fig.suptitle("ROC curve of different anomalies: setting {0}, t {1}".format(setting_num, t))
        ax = fig.add_subplot(111)
        for anomaly, df in cur_data.iterrows():
            ax.plot(df["FPR"], df["TPR"],
                    label="anomaly: {anomaly}. AUC: {auc:.2%}".format(anomaly=anomaly, auc=df["AUC"]))

        ax.legend(loc="lower right")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        # fig.show()
        # plt.show(block=True)
        figaxes.append((fig, ax))
        if save_path is not None:
            fig.savefig(save_path + "_{iteration}_roc".format(iteration=t))
    return figaxes


def plot_roc_by_anomaly(res, net_params, save_path=None):
    """
    plots curves for different anomalies for one setting
    :param res:
    :param net_params:
    :param save_path:
    :return:
    """
    raise NotImplementedError


def plot_auc_by_setting(auc_df, net_params, space_out=False, save_path=None):
    """
    plots curves for different anomalies for one setting
    :param res:
    :param net_params:
    :param save_path:
    :return:
    """
    raise NotImplementedError
    # auc_df = auc_df.sort_index(ascending=True)      # maybe exp the values first to space them out
    # auc_df = auc_df.astype(float)
    # if space_out:
    #     auc_df = exp(auc_df)
    # ax = auc_df.plot(kind="line", marker=".")
    # ax.set_ylabel("AUC")
    # ax.set_xlabel("Setting")
    # ax.set_xticks(auc_df.index)
    # ax.set_xlim((auc_df.index.min() - 0.05, auc_df.index.max() + 0.05))
    # fig = ax.get_figure()
    # fig.suptitle("AUC over different settings")
    # if save_path is not None:
    #     fig.savefig(save_path + "_auc_by_set")
    # return fig, ax


def plot_anomaly_confidence(res, net_params, save_path=None):
    """
    plot the discriminator confidence as the anomaly grows
    :param res:
    :param net_params:
    :param save_path:
    :return:
    """
    raise NotImplementedError
    # figaxes = []
    # for i, params in net_params.items():
    #     fig = plt.figure()
    #     ax = res["mean_conf"].loc[(i, "anomaly")].plot(kind="line",
    #                                                    title="Discriminator Mean Confidence: setting {}".format(i),
    #                                                    legend=True,
    #                                                    marker=".")
    #     ax.plot([0], [res["mean_conf"].loc[(i, "data"), res["mean_conf"].columns[0]]],
    #             marker="*", color="r", label="({}, data)".format(i))           # real data point at anomaly = 0
    #     ax.legend()
    #     ax.set_xlim(0 - 1, res["mean_conf"].columns.max() + 1)
    #     # fig.show()
    #     # plt.show(block=True)
    #     figaxes.append((fig, ax))
    #     if save_path is not None:
    #         fig.savefig(save_path + "_mean_conf")
    # return figaxes


def QQ_plot(true_distribution, generated_samples, line="45", save_path=None):
    """
    creates Q-Q plot of Generator samples
    :param true_distribution: Distribution object.
    :param generated_samples: 1D array of samples from the generator
    :param line: type of line to draw on the plot diagonal.
    :type line: str {‘45’, ‘s’, ‘r’, q’} or None
    :param save_path:
    :return:
    """
    figs = []
    for t, samples in generated_samples.iteritems():
        fig = sm.qqplot(data=samples, dist=true_distribution.get_type(),
                        loc=true_distribution.get_loc(), scale=true_distribution.get_scale(),
                        fit=False, line=line)
        fig.suptitle("Q-Q Plot of the Generated Samples")
        if save_path is not None:
            fig.savefig(save_path + "_{iteration}_qqplot".format(iteration=t))
        figs.append(fig)

    return figs


def PP_plot(true_distribution, generated_samples, line="45", save_path=None):
    """
    creates P-P plot of Generator samples
    :param true_distribution: Distribution object.
    :param generated_samples: 1D array of samples from the generator
    :param line: type of line to draw on the plot diagonal.
    :type line: str {‘45’, ‘s’, ‘r’, q’} or None
    :param save_path:
    :return:
    """
    figs = []
    for t, samples in generated_samples.iteritems():
        fig = sm.qqplot(data=samples, dist=true_distribution.get_type(),
                        loc=true_distribution.get_loc(), scale=true_distribution.get_scale(),
                        fit=False, line=line)
        fig.suptitle("Q-Q Plot of the Generated Samples")
        if save_path is not None:
            fig.savefig(save_path + "_{iteration}_qqplot".format(iteration=t))
        figs.append(fig)

    return figs


def plot_cdf(true_samples, generated_samples, save_path=None):
    figaxes = []
    for t, samples in generated_samples.iteritems():

        fig = plt.figure()
        ax = sns.distplot(true_samples, hist=False, kde=True, kde_kws={'cumulative': True}, label="True")
        # gen_samples_sort = sort(samples.flatten())
        # gen_samples_summed = cumsum(gen_samples_sort)
        # gen_samples_summed = (gen_samples_summed - min(gen_samples_summed)) / \
        #                      (max(gen_samples_summed) - min(gen_samples_summed))
        ecdf = ECDF(samples.flatten())
        ax.plot(ecdf.x, ecdf.y, label="Generated")
        ax.legend(loc="upper left")
        # ax = sns.distplot(generated_samples, hist=False, kde=True, kde_kws={'cumulative': True}, label="Generated")
        ax.set_ylim(-0.05, 1.05)
        fig.suptitle("True and Generated Empirical CDF")
        if save_path is not None:
            fig.savefig(save_path + "_{iteration}_cdf".format(iteration=t))
        figaxes.append((fig, ax))
    return figaxes


def plot_pdf(true_samples, generated_samples, save_path=None):
    figaxes = []
    for t, samples in generated_samples.iteritems():
        fig = plt.figure()
        # true_pdf, gen_pdf = _calc_mutual_pdf(true_samples, generated_samples, bin_num=100)
        # ax = Series(true_samples).hist(bins=200, label="True", alpha=0.6, histtype="stepfilled")
        # Series(generated_samples.flatten()).hist(bins=200, label="Generated", ax=ax, alpha=0.6, histtype="stepfilled")
        ax = sns.distplot(true_samples, bins=200, hist=True, kde=True, label="True",
                          hist_kws={"histtype": "stepfilled"})
        sns.distplot(samples, bins=200, hist=True, kde=True, label="Generated", ax=ax,
                     hist_kws={"histtype": "stepfilled"})
        # ax = sns.barplot(true_pdf, label="True")
        # sns.barplot(gen_pdf, ax=ax, label="Generated")
        fig.suptitle("True and Generated Empirical PDF")
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        if save_path is not None:
            fig.savefig(save_path + "_{iteration}_pdf".format(iteration=t))
        figaxes.append((fig, ax))
    return figaxes


def QQ_scatter(true_samples, generated_samples, save_path=None):
    raise NotImplementedError
    # fig = plt.figure()
    # ax = sns.regplot(x=sort(true_samples), y=sort(generated_samples), fit_reg=False)
    # # plot = sns.jointplot(x=true_samples, y=generated_samples, kind="reg")
    # ax.set_title("Sorted Scatter Plot of Generated and True Samples")
    # ax.set_xlabel("True Samples")
    # ax.set_ylabel("Generated Samples")
    # ax.set_figure(fig)
    # if save_path is not None:
    #     fig.savefig(save_path + "_qqscatter_g-sampled")
    # return fig, ax


def plot_G_tests(losses_df, G_tracking, metric_names=None, trim_y=False, logx=False, save_path=None):
    if metric_names is not None:
        G_df = G_tracking[metric_names]
    else:
        G_df = G_tracking
    if "samples" in G_df.columns:
        G_df = G_df.drop("samples", axis="columns")

    figaxes = []

    for metric_name, metric_series in G_df.iteritems():
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax_metric, ax_losses = ax[0], ax[1]

        fig.suptitle("Losses and {metric} over steps".format(metric=metric_name))

        losses_df.plot(ax=ax_losses, logx=logx)
        metric_series.plot(ax=ax_metric, logx=logx, colormap="Oranges_r", legend=True)
        if trim_y:
            ax_losses.set_ylim(bottom=0, top=3)
        if save_path is not None:
            if logx:
                fig.savefig(save_path + "_loss_{metric}_over_steps_logx".format(metric=metric_name))
            else:
                fig.savefig(save_path + "_loss_{metric}_over_steps".format(metric=metric_name))

        figaxes.append((fig, ax))
    return figaxes


def plot_auc_over_time(auc_series, logx=False, save_path=None):
    auc_series = auc_series.unstack(level="anomaly")
    ax = auc_series.plot(kind="line", marker=".",
                         title="AD Performance Over Time",
                         logx=logx)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Iteration")
    ax.legend(loc="lower left")
    ax.set_xlim((auc_series.index.min() - 0.05, auc_series.index.max() + 0.05))
    fig = ax.get_figure()
    if save_path is not None:
        if logx:
            fig.savefig(save_path + "_auc_over_steps_logx")
        else:
            fig.savefig(save_path + "_auc_over_steps")
    return fig, ax


def plot_auc_anomaly_fit_heatmap(D, G, gan_id, fit_test="KL", vmin=0, cmap="GnBu", save_path=None):
    mutual_iter = D.index.get_level_values("iter").unique().intersection(G.index)

    G_test = G.loc[mutual_iter, fit_test]
    D_auc = D.sort_index(axis=0).loc[pd.IndexSlice[mutual_iter, :], "AUC"]
    D_auc = D_auc.unstack()     # create a table of iteration-over-anomaly instead of multi-index series.

    D_auc.reindex_axis(sorted(D_auc.columns, key=lambda x: x[0]), axis=1)   # sort columns by anomaly (mu)

    G_test = G_test.sort_values()
    D_auc = D_auc.reindex(G_test.index)       # sort the AUC values (rows) by the fit

    D_auc = D_auc.set_index(G_test.round(5))                     # set the fit values to be the df indices

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)

    ax = sns.heatmap(D_auc, vmin=vmin, vmax=1, cmap=cmap, ax=ax)

    ax.set_title("AUC over anomaly and {test}. Setting: {setting}".format(test=fit_test, setting=gan_id))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # fig = ax.get_figure()
    if save_path is not None:
        fig.savefig(save_path + "_auc_anomaly_fit_heatmap_{test}".format(test=fit_test))

    return fig, ax


class Plotter:
    plot_funcs = {"cdf": plot_cdf,
                  "pdf": plot_pdf,
                  "qqplot": QQ_plot,
                  "qqscatter": QQ_scatter,
                  "roc_setting": plot_roc_by_setting,
                  "auc_time": plot_auc_over_time,
                  "G_tests": plot_G_tests}

    def __init__(self, setting_num, plot_dir, progress_tracker, true_distribution=None, anomaly_distribution=None):
        self.setting_num = setting_num
        self.pg = progress_tracker
        self.true_distribution = true_distribution
        self.anomaly_distribution = anomaly_distribution
        plot_dir_name = os.path.join(plot_dir, self.pg.gan_signature)
        os.mkdir(plot_dir_name)
        self.plot_path_prefix = os.path.join(plot_dir_name, self.pg.gan_signature)

    def plot(self, plot_types, iteration_checkpoints, trim_y=False, logx=False):
        """
        Main plotting function managing the plots
        :param trim_y: should trim the y limits or not.
        :param plot_types: what types of plots to generate. Should correspond to the Plotter.plot_funcs dictionary
        :param iteration_checkpoints: at what iterations to generate plots. For plots that need an entire figure for a
                                      given iteration (e.g. pdf, cdf, qq plot, roc, etc...)
        :return:
        """
        # get only the relevant generated samples over time:
        generated_samples = self.pg.G_tracking.loc[iteration_checkpoints, "samples"].dropna()
        for plot_type in plot_types:
            plot_func = self.plot_funcs.get(plot_type)
            if plot_type == "cdf":
                plot_cdf(true_samples=self.pg.test_true_samples, generated_samples=generated_samples,
                         save_path=self.plot_path_prefix)
            elif plot_type == "pdf":
                plot_pdf(true_samples=self.pg.test_true_samples, generated_samples=generated_samples,
                         save_path=self.plot_path_prefix)
            elif plot_type == "qqplot":
                QQ_plot(true_distribution=self.true_distribution, generated_samples=generated_samples,
                        save_path=self.plot_path_prefix, line="s")
            elif plot_type == "qqscatter":
                raise NotImplementedError
            elif plot_type == "roc_setting":
                # filter only the relevant iterations for plotting:
                df = self.pg.D_tracking.loc[iteration_checkpoints, ["FPR", "TPR", "AUC"]].dropna()
                # df_pos = df.select(lambda x: x[1][0] >= 0, axis="rows")     # filter in positive anomalies
                df_pos = df.loc[[x[1][0] >= 0 for x in df.index.get_values()]]
                if not df_pos.empty:
                    plot_roc_by_setting(self.setting_num, df_pos, save_path=self.plot_path_prefix + "_pos-anom")
                # df_neg = df.select(lambda x: x[1][0] <= 0, axis="rows")  # filter in negative anomalies
                df_neg = df.loc[[x[1][0] <= 0 for x in df.index.get_values()]]
                if not df_neg.empty:
                    plot_roc_by_setting(self.setting_num, df_neg, save_path=self.plot_path_prefix + "_neg-anom")
            elif plot_type == "auc_time":
                auc_series = self.pg.D_tracking["AUC"]
                # auc_series_pos = auc_series.select(lambda x: x[1][0] >= 0)
                auc_series_pos = auc_series[[x[1][0] >= 0 for x in auc_series.index.get_values()]]
                if not auc_series_pos.empty:
                    plot_auc_over_time(auc_series_pos, save_path=self.plot_path_prefix + "_pos-anom", logx=logx)
                # auc_series_neg = auc_series.select(lambda x: x[1][0] <= 0)
                auc_series_neg = auc_series[[x[1][0] <= 0 for x in auc_series.index.get_values()]]
                if not auc_series_neg.empty:
                    plot_auc_over_time(auc_series_neg, save_path=self.plot_path_prefix + "_neg-anom", logx=logx)
            elif plot_type == "G_tests":
                plot_G_tests(self.pg.loss_tracking, self.pg.G_tracking, metric_names=None, trim_y=trim_y, logx=logx,
                             save_path=self.plot_path_prefix)
            elif plot_type == "auc_fit_anomaly_heatmap":
                for fit_test in self.pg.G_tracking.columns.drop("samples"):     # do so for every fit test done on G
                    plot_auc_anomaly_fit_heatmap(self.pg.D_tracking, self.pg.G_tracking, self.pg.gan_id,
                                                 fit_test=fit_test, save_path=self.plot_path_prefix)


#

#

# ######################### #
# ## Plots for after run ## #
# ######################### #

def aplot_auc_anomaly_fit_heatmap(results, fit_test="KL", vmin=0, cmap="GnBu", save_path=None):
    figaxes = {}
    for p, tracks in results.items():
        test_figaxes = {}
        D = tracks.get("D")
        G = tracks.get("G")
        for test_fit in G.columns.drop("samples"):
            fig, ax = plot_auc_anomaly_fit_heatmap(D, G, gan_id=p, fit_test=test_fit,
                                                   vmin=vmin, cmap=cmap, save_path=save_path+"_"+str(p))
            test_figaxes[test_fit] = (fig, ax)

        figaxes[p] = test_figaxes
    return figaxes

