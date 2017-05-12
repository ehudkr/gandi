from scipy.stats import entropy, ks_2samp, kstest, anderson
import numpy as np

EPSILON = 10e-10


# ############################################### #
# ##### Distribution Metrics for Testing G: ##### #
# ############################################### #


def calc_Dkl(true_samples, generated_samples, bin_num=100):

    # calc mutual bins for the pdf:
    true_pdf, generated_pdf = _calc_mutual_pdf(true_samples, generated_samples, bin_num)
    # Dkl = (true_pdf * pd.np.log2(true_pdf / generated_pdf)).sum()
    Dkl = entropy(true_pdf, generated_pdf, base=2)
    return Dkl


def calc_ks(true_samples, generated_samples):
    if isinstance(true_samples, np.ndarray):    # apply ks test for samples
        return ks_2samp(true_samples, generated_samples)
    else:                                       # true_samples is a cdf callable
        return kstest(rvs=generated_samples, cdf=true_samples, alternative="two-sided")


def calc_l1_cdf(true_samples, generated_samples, bin_num=100):
    """
    Like total variation distance but based on the cdf rather the pdf
    :param true_samples:
    :param generated_samples:
    :param bin_num:
    :return:
    """
    true_pdf, generated_pdf = _calc_mutual_pdf(true_samples, generated_samples, bin_num)
    true_cdf = np.cumsum(true_pdf)
    generated_cdf = np.cumsum(generated_pdf)
    return np.sum(np.abs(true_cdf - generated_cdf))     # l_1 distance


def calc_anderson(generated_samples):
    return anderson(generated_samples, dist="norm")


def _calc_mutual_pdf(true_samples, generated_samples, bin_num):
    """
    calculates pdfs over mutual range
    :param true_samples:
    :param generated_samples:
    :param num_bins:
    :return:
    """
    min_val = min(generated_samples.min(), true_samples.min())
    max_val = max(generated_samples.max(), true_samples.max())
    bins = np.linspace(start=min_val, stop=max_val, num=bin_num, endpoint=True)
    generated_pdf, _ = np.histogram(generated_samples, bins=bins, density=True)
    generated_pdf[generated_pdf == 0] = EPSILON  # to avoid division by zero
    generated_pdf /= generated_pdf.sum()
    true_pdf, _ = np.histogram(true_samples, bins=bins, density=True)
    true_pdf /= true_pdf.sum()
    return true_pdf, generated_pdf


# ###################### #
# ##### Testing D: ##### #
# ###################### #



