import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Kriging:
    def __init__(self, c_v, c_v_bounds, rbf_ls, rbf_ls_bounds, n_rest, a, norm_y):
        """

        :param float c_v: Initial constant value for Kriging model.
        :param tuple c_v_bounds: Bounds of values for constant value of Kriging model (Low, High).
        :param np.ndarray rbf_ls: Initial values of Squared Exponential Kernel of Kriging Model.
        :param list rbf_ls_bounds: Bounds of values for Squared Exponential kernel of Kriging model [(Low, High),
                                                                                                     (Low, High).
                                                                                                     (Low, High)].
        :param int n_rest: Number of times the optimizer can reset.
        :param float a: Noise added at the diagonal.
        :param bool norm_y: Whether to normalize predicted values or not.
        """
        self.rbf_ls = rbf_ls
        self.rbf_ls_bounds = rbf_ls_bounds
        self.c_v = c_v
        self.c_v_bounds = c_v_bounds
        self.n_rest = n_rest
        self.a = a
        self.norm_y = norm_y

        self.kernel = C(c_v, c_v_bounds) * RBF(length_scale=self.rbf_ls,
                                               length_scale_bounds=self.rbf_ls_bounds)
        self.model = GPR(kernel=self.kernel,
                         n_restarts_optimizer=n_rest,
                         alpha=a,
                         normalize_y=norm_y)

    def train(self, samples, real_values):
        """
        Train Kriging model.

        :param np.ndarray samples: BSA parameters [[Size, Cutoff, Threshold]]
        :param np.ndarray real_values: Real aptitudes.
        :return: None.
        """
        # Re-Shape the values.
        rs_realValues = real_values.reshape(-1, 1)

        self.model.fit(samples, rs_realValues)

    def predict(self, sample):
        """
        Make predictions using Kriging model.

        :param np.ndarray or list sample: BSA parameters [[Size, Cutoff, Threshold]]
        :return: prediction: SNR predicted value [[Float]].
                 sigma: Sigma value of prediction [Float].
        """
        prediction, sigma = self.model.predict(sample, return_std=True)
        return prediction, sigma

    def report_kernel(self):
        """
        Report kernel of Kirging model.

        :return: Kernel [String]
        """
        return f'{self.model.kernel_}'

    def report_lml(self):
        """
        Report Log-Marginal Likelihood.

        :return: Log-Marginal Likelihood [Float & String].
        """
        lml = self.model.log_marginal_likelihood(self.model.kernel_.theta)
        return lml, f'{lml}'

    def report_score(self, x, y):
        """
        R2-Score.
        :param np.ndarray x: Samples
        :param np.ndarray y: Real values.
        :return: R2-Score [Float]
        """
        return self.model.score(x, y)


if __name__ == '__main__':
    # Temporal
    from Implementation.lhs import LHS
    from Implementation.bsa import BSA
    import scipy.io as scio


    def get_real(samples):
        snrs = []
        for sample in samples:
            spiker = BSA(int(sample[0]), sample[1], sample[2], scale, 1000)

            encoded = spiker.encode(signal)
            decoded = spiker.decode(encoded)

            snrs.append(spiker.SNR(signal, decoded))

        return np.array(snrs)


    def plot_comparison(name, real, predicted):
        plt.figure(name)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)

        plt.scatter(real, predicted, label='Samples')
        plt.plot([real.min, real.max], [real.min, real.max],
                 'k--', lw=3, label='Perfect prediction')

        plt.xlabel('True value (SNR)', fontsize=16)
        plt.ylabel('Predicted value (SNR)', fontsize=16)
        plt.legend()


    # Signal
    fs = 1000
    signal = scio.loadmat('../Signals/Tests/Signal_S_E2_1.mat')['signal'].T[0]
    shift = signal.min()
    signal = signal - shift
    scale = signal.max() * 2

    c_v_ = 1.0
    c_v_bouds_ = (1e-5, 1e5)
    rbf_ls_ = np.ones(3)
    rbf_ls_bounds_ = [(1e-2, 1e6), (1e-2, 1e8), (1e-5, 1e2)]
    n_rest_ = 50
    a_ = 1e-3

    # Training Sampling
    samples_train = LHS(32,
                        rangeSize=[16, 80],
                        rangeCut=[20, 80],
                        rangeThreshold=[0.8, 1.1])

    snrs_train = get_real(samples_train)
    kriging = Kriging(c_v=c_v_, c_v_bounds=c_v_bouds_, rbf_ls=rbf_ls_, rbf_ls_bounds=rbf_ls_bounds_,
                      a=a_, n_rest=n_rest_, norm_y=True)
    print(kriging.kernel)
    kriging.train(samples_train, snrs_train)
    kriging.report_kernel()
    kriging.report_lml()

    samples_test = LHS(32,
                       rangeSize=[16, 80],
                       rangeCut=[20, 80],
                       rangeThreshold=[0.8, 1.1])
    snrs_test = get_real(samples_test)

    pred, sig = kriging.predict(samples_test)
    plot_comparison('Test', snrs_test, pred)
    print(pred)
    print(sig)
