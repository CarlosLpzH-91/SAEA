import scipy.io as scio
from scipy.signal import firwin
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor as sklK
from sklearn.gaussian_process.kernels import RBF as sklRBF, ConstantKernel as sklC
# from skopt.acquisition import gaussian_ei as EI

from Implementation.lhs import LHS
from Implementation.bsa import BSA

plt.style.use('seaborn')


def plot(name, real_y, predicted_y):
    plt.figure(name)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    plt.scatter(real_y, predicted_y, label='Samples')
    plt.plot([real_y.min(), real_y.max()], [real_y.min(), real_y.max()],
             'k--', lw=3, label='Perfect prediction')
    plt.xlabel('True value (SNR)', fontsize=16)
    plt.ylabel('Predicted value (SNR)', fontsize=16)
    plt.legend()


def get_real(samples):
    snrs = []
    for sample in samples:
        spiker = BSA(int(sample[0]), sample[1], sample[2], scale, 1000)

        encoded = spiker.encode(signal)
        decoded = spiker.decode(encoded)

        snrs.append(spiker.SNR(signal, decoded))

    return np.array(snrs)


def EI(y, std, y_opt=0.0, xi=0.01, return_grad=False):
    """
    Use the expected improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    ----------- Modified version of SciKit-Optimize:
    https://scikit-optimize.github.io/stable/_modules/skopt/acquisition.html#gaussian_ei

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Values where the acquisition function should be computed.

    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    y_opt : float, default 0
        Previous minimum value which we would like to improve upon.

    xi : float, default=0.01
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    values : array-like, shape=(X.shape[0],)
        Acquisition function values computed at X.
    """
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #
    #     if return_grad:
    #         mu, std, mu_grad, std_grad = model.predict(
    #             X, return_std=True, return_mean_grad=True,
    #             return_std_grad=True)
    #
    #     else:
    #         mu, std = model.predict(X, return_std=True)

    # check dimensionality of mu, std so we can divide them below
    # if (mu.ndim != 1) or (std.ndim != 1):
    #     raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
    #                      "however both must be 1-dimensional. Did you train "
    #                      "your model with an (N, 1) vector instead of an "
    #                      "(N,) vector?"
    #                      .format(mu.ndim, std.ndim))

    ####### Modification
    mu = y.reshape(1, -1)[0]

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    # if return_grad:
    #     if not np.all(mask):
    #         return values, np.zeros_like(std_grad)
    #
    #     # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
    #     # improve_grad is the gradient of t wrt x.
    #     improve_grad = -mu_grad * std - std_grad * improve
    #     improve_grad /= std ** 2
    #     cdf_grad = improve_grad * pdf
    #     pdf_grad = -improve * cdf_grad
    #     exploit_grad = -mu_grad * cdf - pdf_grad
    #     explore_grad = std_grad * pdf + pdf_grad
    #
    #     grad = exploit_grad + explore_grad
    #     return values, grad

    return values


# Signal
print('Acquiring signal')
fs = 1000
signal = scio.loadmat('../Signals/Tests/Signal_S_E2_1.mat')['signal'].T[0]
shift = signal.min()
signal = signal - shift
scale = max(signal) * 2

# Training Sampling
print('Sampling for Training')
samples_train = LHS(32,
                    rangeSize=[16, 80],
                    rangeCut=[20, 80],
                    rangeThreshold=[0.8, 1.1])

snrs_train = get_real(samples_train)

# Sklearn Implementation
print('GP Fitting - SkLearn')
kernel = sklC(1.0, (1e-5, 1e5)) * sklRBF(length_scale=np.ones(3),
                                         length_scale_bounds=[(1e-5, 1e5)] * 3)
# kernel = sklC(1.0, (1e-5, 1e5)) * sklRBF(length_scale=np.ones(3),
#                                          length_scale_bounds=[(1e-5, 1e5),
#                                                               (1e-5, 1e5),
#                                                               (1e-5, 1e5)])
# kernel = 1.0 * sklRBF(length_scale=np.ones(3), length_scale_bounds=[(1e-5, 1e5)] * 3)

sklGP = sklK(kernel=kernel,
             n_restarts_optimizer=10,
             normalize_y=True,
             alpha=1e-6)
print(f'Initial: {kernel}')
sklGP.fit(samples_train, snrs_train.reshape(-1, 1))
print(f'Final: {sklGP.kernel_}')
print(f'LML: {sklGP.log_marginal_likelihood(sklGP.kernel_.theta)}')

# predictions_train, sigma_train = sklGP.predict(samples_train, return_std=True)
#
# plot('Train', snrs_train, predictions_train)

# Prediction
# Over 3 values
# print('Sampling for Testing')
# samples_test = LHS(16,
#                    rangeSize=[16, 80],
#                    rangeCut=[20, 80],
#                    rangeThreshold=[0.8, 1.1])
#
# snrs_test = get_real(samples_test)
#
# predictions_test, sigma_test = sklGP.predict(samples_test, return_std=True)
#
# plot('Test', snrs_test, np.ravel(predictions_test))

# Over one value
# Size
print('Prediction: Size')
x = np.zeros((16, 3))
x[:, 0] = np.arange(16, 80, 4) + 1
x[:, 1] = 46
x[:, 2] = 0.95

ySiz_snr = get_real(x)

ypredSiz, sigmaSiz = sklGP.predict(x, return_std=True)
# EI
sizEI = EI(ypredSiz, sigmaSiz)
print(sizEI)

# plot('Size-True', ySiz_snr, np.ravel(ypredSiz))

plt.figure('Kall-Size')
plt.plot(x[:, 0], ySiz_snr, 'k--', label='True')
plt.plot(x[:, 0], ypredSiz, label='Prediction', color='green')
plt.fill_between(np.ravel(x[:, 0]),
                 np.ravel(ypredSiz - 1.96 * sigmaSiz.reshape(-1, 1)),
                 np.ravel(ypredSiz + 1.96 * sigmaSiz.reshape(-1, 1)),
                 color='lightgreen',
                 label='Confidence'
                 )
# plt.scatter(x[:, 0], ySiz_snr, label='Real Data', s=orderEI*10, cmap='YlOrBr', c=orderEI)
plt.scatter(x[:, 0], ySiz_snr, label='Observations', cmap='YlOrBr', c=sizEI, zorder=4)
plt.colorbar()
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Size')

# Cutoff
print('Prediction: Cutoff')
x = np.zeros((16, 3))
x[:, 0] = 69
x[:, 1] = np.linspace(20, 80, num=16)
x[:, 2] = 0.95

yCut_snr = get_real(x)

ypredCut, sigmaCut = sklGP.predict(x, return_std=True)
# EI
cutEI = EI(ypredCut, sigmaCut)
print(cutEI)

# plot('Cut-True', yCut_snr, np.ravel(ypredCut))

plt.figure('Kall-Cut')
plt.plot(x[:, 1], yCut_snr, 'k--', label='True')
plt.plot(x[:, 1], ypredCut, label='Prediction', color='green')
plt.fill_between(np.ravel(x[:, 1]),
                 np.ravel(ypredCut - 1.96 * sigmaCut.reshape(-1, 1)),
                 np.ravel(ypredCut + 1.96 * sigmaCut.reshape(-1, 1)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.scatter(x[:, 1], yCut_snr, label='Observations', cmap='YlOrBr', c=cutEI, zorder=4)
plt.colorbar()
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Cutoff Frequency')

# Threshold
print('Prediction: Threshold')
x = np.zeros((16, 3))
x[:, 0] = 69
x[:, 1] = 46
x[:, 2] = np.linspace(0.8, 1.1, num=16)

yThr_snr = get_real(x)

ypredThr, sigmaThr = sklGP.predict(x, return_std=True)
# EI
thrEI = EI(ypredThr, sigmaThr)
print(thrEI)

# plot('Thr-True', yThr_snr, np.ravel(ypredThr))

plt.figure('Kall-Thr')
plt.plot(x[:, 2], yThr_snr, 'k--', label='True')
plt.plot(x[:, 2], ypredThr, label='Prediction', color='green')
plt.fill_between(np.ravel(x[:, 2]),
                 np.ravel(ypredThr - 1.96 * sigmaThr.reshape(-1, 1)),
                 np.ravel(ypredThr + 1.96 * sigmaThr.reshape(-1, 1)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.scatter(x[:, 2], yThr_snr, label='Observations', cmap='YlOrBr', c=thrEI, zorder=4)
plt.colorbar()
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Threshold')



if __name__ == '__main__':
    # Variables
    verbose = True
    maximization = True
    fs = 1000
    size = 50
    num_gen = 200
    cr = 0.8
    fx = 0.5
    ranges = [[16, 80],
              [20, 80],
              [0.8, 1.1]]
    update = 10
    selection = 5
    type_criteria = 'EI - First'
    re_evaluate_km = False

    # Signal
    signal = scio.loadmat('../Signals/Tests/Signal_S_E2_1.mat')['signal'].T[0]
    shift = signal.min()
    signal = signal - shift
    scale = signal.max() * 2

    # Values
    bests = []
    current_gen = 0

    # Analysis
    differences = []
    sigmas = []

    # Initial time
    stime = time.time()

    # Initial Sampling: LHS
    samples = LHS(numPoints=size,
                  rangeSize=ranges[0],
                  rangeCut=ranges[1],
                  rangeThreshold=ranges[2])

    # Initial Evaluations: Expensive
    aptitudes = np.array([get_real(sample, signal, scale, fs) for sample in samples])

    # Initial Kriging model
    km = Kriging()
    km.train(samples, aptitudes)
    if verbose:
        print(km.report_kernel())
        print(km.report_lml())

    # Initial population
    population = [Individual(km, vector, aptitude, 0, ranges, source='KM')
                  for vector, aptitude in zip(samples, aptitudes)]

    if maximization:
        population.sort(key=lambda x: x.aptitude, reverse=True)
    else:
        population.sort(key=lambda x: x.aptitude)

    # Initial best
    bests.append(population[0])
    print(f'Best: {population[0].report()}')

    # DE: Main loop

    while current_gen != num_gen:
        # Increase generation counter
        current_gen += 1
        # DE: Mutation-Crossover
        all_targets = [target.give_values() for target in population]
        for i in range(size):
            # Possible guest vectors
            candidates = [candidate for j, candidate in enumerate(all_targets)]
            # Select guest vectors
            choices = random.sample(candidates, 3)
            # Do operation
            population[i].operator_model(cr, fx, choices[0], choices[1], choices[2], False)

        # KM: Update model
        if not current_gen % update:
            # print(f'Pop: {[x.give_values() for x in population]}')
            # Select most uncertain points
            criteria = np.array([i.sigma for i in population])
            # print(f'Criteria: {criteria}')
            max_indices = criteria.argsort()[-selection:]
            # print(f'Max: {max_indices}')
            new_samples = np.array([population[k].vector
                                    for k in max_indices])
            # print(f'New Samples: {new_samples}')
            old_aptitudes = [population[i].aptitude for i in max_indices]
            selected_sigmas = [population[i].sigma for i in max_indices]

            # Expensive evaluation with new samples
            new_aptitudes = np.array([get_real(sample, signal, scale, fs) for sample in new_samples])
            # print(f'New Aptitudes: {new_aptitudes}')

            # Update aptitude of selected samples
            for i, update_aptitude in zip(max_indices, new_aptitudes):
                population[i].aptitude = update_aptitude
                population[i].sigma = 0
                population[i].source = 'KM'

            # print(f'Pop Updated: {[x.give_values() for x in population]}')

            # Append new samples and aptitudes
            samples = np.append(samples, new_samples, axis=0)
            aptitudes = np.append(aptitudes, new_aptitudes, axis=0)

            if verbose:
                print(f'New samples: {new_samples}')
                print(f'Old aptitudes: {old_aptitudes}')
                print(f'New aptitudes: {new_aptitudes}')

            # Re-train model
            km.train(samples, aptitudes)
            if verbose:
                print(km.report_kernel())
                print(km.report_lml())

            if re_evaluate_km:
                # Re-evaluated population in KM
                for i, individual in enumerate(population):
                    if i not in max_indices:
                        prediction, sigma = km.predict([individual.vector])
                        individual.aptitude = prediction[0][0]
                        individual.sigma = sigma[0]
                        individual.source = 'KM'

            differences.append(np.mean(old_aptitudes - new_aptitudes))
            sigmas.append(np.mean(selected_sigmas))

        if maximization:
            population.sort(key=lambda x: x.aptitude, reverse=True)
        else:
            population.sort(key=lambda x: x.aptitude)

        # Best
        bests.append(copy.deepcopy(population[0]))
        print(f'Gen {current_gen}: {population[0].report()}')

    best_aptitudes = [best.aptitude for best in bests]
    # Re-evaluated population in expensive
    real_aptitudes = [get_real(i.vector, signal, scale, fs) for i in population]
    if maximization:
        real_best = population[np.argsort(real_aptitudes)[-1]]
    else:
        real_best = population[np.argsort(real_aptitudes)[0]]
    print(f'Real Best: {real_best.vector} f(x): {np.max(real_aptitudes)}')





    print(f'Total time: {time.time() - stime}')

    de_aptitudes = [i.aptitude for i in population]
    # Show analysis
    plt.figure('MD_RE-P_M')
    plt.plot(range(1, 21), differences)
    plt.xticks(range(1, 21))
    plt.axhline(0, c='k', ls='--', zorder=-1)
    plt.xlabel('Re-evaluations')
    plt.ylabel('Mean Differences')
    plt.subplots_adjust(left=0.04, bottom=0.067, right=0.97, top=0.97)

    plt.figure('Sig_RE_M')
    plt.plot(range(1, 21), sigmas)
    plt.xticks(range(1, 21))
    plt.axhline(0, c='k', ls='--', zorder=-1)
    plt.xlabel('Re-evaluations')
    plt.ylabel('Mean Sigma')
    plt.subplots_adjust(left=0.04, bottom=0.067, right=0.97, top=0.97)

    plt.figure('Conv_M')
    plt.plot(range(201), best_aptitudes, zorder=2)
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.subplots_adjust(left=0.04, bottom=0.067, right=0.98, top=0.99)

    plt.figure('Final_Population_M')
    plt.scatter(range(size), de_aptitudes, label='Kriging Values', s=10)
    plt.scatter(range(size), real_aptitudes, label='Real Values', zorder=-1)
    plt.xlabel('Individual')
    plt.ylabel('f(x)')
    plt.legend()
    plt.subplots_adjust(left=0.04, bottom=0.067, right=0.98, top=0.99)

    km.plot_comparison('Comp_M', np.array(real_aptitudes), np.array(de_aptitudes))
    print(f'Mean Differences: {np.mean(differences)}')
    print(f'Mean Sigmas: {np.mean(sigmas)}')
