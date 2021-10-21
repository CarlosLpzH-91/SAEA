import scipy.io as scio
from scipy.signal import firwin
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.gaussian_process import GaussianProcessRegressor as sklK
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from smt.surrogate_models import KRG as smtK

from Implementation.lhs import LHS, uniqueVals
from Implementation.bsa import BSA

# Signal
print('Acquiring signal')
fs = 1000
signal = scio.loadmat('../Signals/Tests/Signal_S_E2_1.mat')['signal'].T[0]
shift = signal.min()
signal = signal - shift
scale = max(signal) * 2

# Sampling
print('Sampling')
samples = LHS(32,
              rangeSize=[16, 80],
              rangeCut=[20, 80],
              rangeThreshold=[0.8, 1.1])

sizSamples_fit = np.array(samples[:, 0])
cutSamples_fit = np.array(sorted(samples[:, 1]))
thrSamples_fit = np.array(sorted(samples[:, 2]))
sizSamples = np.array(sorted(sizSamples_fit))
cutSamples = np.array(sorted(cutSamples_fit))
thrSamples = np.array(sorted(thrSamples_fit))

# testRange = np.arange(0.8, 1.1, 0.01)

# Evaluation
print('Evaluating')

# Size
# print('Size')
# sizSnrs = []
# for s in sizSamples_fit:
#     spiker = BSA(int(s), 46, 0.95, scale, 1000)
#
#     encoded = spiker.encode(signal)
#     decoded = spiker.decode(encoded)
#
#     sizSnrs.append(spiker.SNR(signal, decoded))
#
# sizSnrs = np.array(sizSnrs)

# Cutoff Frequency
# print('Cutoff frequency')
# cutSnrs = []
# for c in cutSamples_fit:
#     spiker = BSA(69, c, 0.95, scale, 1000)
#
#     encoded = spiker.encode(signal)
#     decoded = spiker.decode(encoded)
#
#     cutSnrs.append(spiker.SNR(signal, decoded))
#
# cutSnrs = np.array(cutSnrs)

# Threshold
print('Threshold')
trhSnrs = []
for t in thrSamples_fit:
    spiker = BSA(69, 46, t, scale, 1000)

    encoded = spiker.encode(signal)
    decoded = spiker.decode(encoded)

    trhSnrs.append(spiker.SNR(signal, decoded))

thrSnrs = np.array(trhSnrs)

# Sklearn Implementation
print('GP Fitting')

# # Size
# print('Size')
sizX = np.arange(16, 80, 4) + 1
# # kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-6, 20.0))
# sizKernel = 1.0 * RBF(1.9, (1e-6, 20))
# sizGp = sklK(kernel=sizKernel, n_restarts_optimizer=20)
# print(f'Initial: {sizKernel}')
# sizGp.fit(sizSamples_fit.reshape(-1, 1), sizSnrs.reshape(-1, 1))
# print(f'Final: {sizGp.kernel_}')
# print(f'LML: {sizGp.log_marginal_likelihood(sizGp.kernel_.theta)}')
# sizYpred, sizSigma = sizGp.predict(sizX.reshape(-1, 1), return_std=True)

# # Cutoff frequency
# print('Cutoff frequency')
cutX = np.linspace(20, 80, num=16)
# # kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-6, 20.0))
# cutKernel = 1.0 * RBF(1.9, (1e-6, 20))
# cutGp = sklK(kernel=cutKernel, n_restarts_optimizer=20)
# print(f'Initial: {cutKernel}')
# cutGp.fit(cutSamples_fit.reshape(-1, 1), cutSnrs.reshape(-1, 1))
# print(f'Final: {cutGp.kernel_}')
# print(f'LML: {cutGp.log_marginal_likelihood(cutGp.kernel_.theta)}')
# cutYpred, cutSigma = cutGp.predict(cutX.reshape(-1, 1), return_std=True)

# Threshold
# print('Threshold')
thrX = np.linspace(0.8, 1.1, num=16)
# # kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-6, 20.0))
# thrKernel = 1.0 * RBF(10, (1e-6, 20))
# thrGp = sklK(kernel=thrKernel, n_restarts_optimizer=20)
# print(f'Initial: {thrKernel}')
# thrGp.fit(thrSamples_fit.reshape(-1, 1), thrSnrs.reshape(-1, 1))
# print(f'Final: {thrGp.kernel_}')
# print(f'LML: {thrGp.log_marginal_likelihood(thrGp.kernel_.theta)}')
# thrYpred, thrSigma = thrGp.predict(thrX.reshape(-1, 1), return_std=True)


# Originals SNRs
print('Acquiring originals SNRs')
sizOrigianlSnrs = scio.loadmat('SNRsSiz_test.mat')['data'].T
cutOrigianlSnrs = scio.loadmat('SNRsCut_test.mat')['data'].T
thrOrigianlSnrs = scio.loadmat('SNRsThr_test.mat')['data'].T

sizRange = np.arange(16, 80, 2) + 1
cutRange = np.arange(20, 80, 1)
thrRange = np.arange(0.8, 1.1, 0.01)

# Plots
print('Plotting')

# plt.figure('K-Size')
# plt.plot(sizRange, sizOrigianlSnrs, '--', label='Original')
# plt.plot(sizSamples_fit, sizSnrs, 'o', label='Training data')
# plt.plot(sizX, sizYpred, label='Prediction', color='green')
# plt.fill_between(np.ravel(sizX),
#                  np.ravel(sizYpred - 3.0 * sizSigma.reshape(-1, 1)),
#                  np.ravel(sizYpred + 3.0 * sizSigma.reshape(-1, 1)),
#                  color='lightgreen',
#                  label='Confidence'
#                  )
# plt.scatter(69, 8.711, 100, color='r', marker='*', label='Optimum')
# plt.legend()
# plt.ylabel('SNR')
# plt.xlabel('Size')
#
# plt.figure('K-Cutoff')
# plt.plot(cutRange, cutOrigianlSnrs, '--', label='Original')
# plt.plot(cutSamples_fit, cutSnrs, 'o', label='Training data')
# plt.plot(cutX, cutYpred, label='Prediction', color='green')
# plt.fill_between(np.ravel(cutX),
#                  np.ravel(cutYpred - 3.0 * cutSigma.reshape(-1, 1)),
#                  np.ravel(cutYpred + 3.0 * cutSigma.reshape(-1, 1)),
#                  color='lightgreen',
#                  label='Confidence'
#                  )
# plt.scatter(46, 8.711, 100, color='r', marker='*', label='Optimum')
# plt.legend()
# plt.ylabel('SNR')
# plt.xlabel('Cutoff')

# plt.figure('K-Threshold')
# plt.plot(thrRange, thrOrigianlSnrs, '--', label='Original')
# plt.plot(thrSamples_fit, thrSnrs, 'o', label='Training data')
# plt.plot(thrX, thrYpred, label='Prediction', color='green')
# plt.fill_between(np.ravel(thrX),
#                  np.ravel(thrYpred - 3.0 * thrSigma.reshape(-1, 1)),
#                  np.ravel(thrYpred + 3.0 * thrSigma.reshape(-1, 1)),
#                  color='lightgreen',
#                  label='Confidence'
#                  )
# plt.scatter(0.95, 8.711, 100, color='r', marker='*', label='Optimum')
# plt.legend()
# plt.ylabel('SNR')
# plt.xlabel('Threshold')

# # Overall
# print('Overall')
#
# # Evaluation
# print('Evaluating')
#
snrs = []
for sample in samples:
    spiker = BSA(int(sample[0]), sample[1], sample[2], scale, 1000)

    encoded = spiker.encode(signal)
    decoded = spiker.decode(encoded)

    snrs.append(spiker.SNR(signal, decoded))

snrs = np.array(snrs)

# Sklearn Implementation
print('GP Fitting - SkLearn')

# kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-6, 20.0))
kernel = 1.0 * RBF(1.9, (2e-2, 2e2))
gp = sklK(kernel=kernel, alpha=1e-2, n_restarts_optimizer=9)
print(f'Initial: {kernel}')
gp.fit(samples, snrs.reshape(-1, 1))
print(f'Final: {gp.kernel_}')
print(f'LML: {gp.log_marginal_likelihood(gp.kernel_.theta)}')

# Prediction
x = np.zeros((16, 3))

print('Size Predicting - SkLearn')
x[:, 0] = sizX
x[:, 1] = 46
x[:, 2] = 0.95
ypredSiz, sigmaSiz = gp.predict(x, return_std=True)

plt.figure('Kall-Size- SkLearn')
plt.plot(samples[:, 0], snrs, 'o', label='Training data')
plt.plot(sizX, ypredSiz, label='Prediction', color='green')
plt.fill_between(np.ravel(sizX),
                 np.ravel(ypredSiz - 1.0 * sigmaSiz.reshape(-1, 1)),
                 np.ravel(ypredSiz + 1.0 * sigmaSiz.reshape(-1, 1)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Size')

print('Cutoff Predicting - SkLearn')
x[:, 0] = 69
x[:, 1] = cutX
x[:, 2] = 0.95
ypredCut, sigmaCut = gp.predict(x, return_std=True)

plt.figure('Kall-Cutoff-SkLearn')
plt.plot(samples[:, 1], snrs, 'o', label='Training data')
plt.plot(cutX, ypredCut, label='Predictions', color='green')
plt.fill_between(np.ravel(cutX),
                 np.ravel(ypredCut - 1.0 * sigmaCut.reshape(-1, 1)),
                 np.ravel(ypredCut + 1.0 * sigmaCut.reshape(-1, 1)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Cutoff')

print('Threshold Predicting - SkLearn')
x = np.zeros((16, 3))
x[:, 0] = 69
x[:, 1] = 46
x[:, 2] = thrX

ypredThr, sigmaThr = gp.predict(x, return_std=True)

plt.figure('Kall-Threshold-SkLearn')
plt.plot(samples[:, 2], snrs, 'o', label='Training data')
plt.plot(thrX, ypredThr, label='Prediction', color='green')
plt.fill_between(np.ravel(thrX),
                 np.ravel(ypredThr - 1.0 * sigmaThr.reshape(-1, 1)),
                 np.ravel(ypredThr + 1.0 * sigmaThr.reshape(-1, 1)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Threshold')

# SMT Implementation
print('GP Fitting - SMT')

gp2 = smtK(theta0=[1e-2])
gp2.set_training_values(samples, snrs)
gp2.train()

# Prediction
x = np.zeros((16, 3))

print('Size Predicting - SMT')
x[:, 0] = sizX
x[:, 1] = 46
x[:, 2] = 0.95
ypredSiz, sigmaSiz = gp2.predict_values(x), gp2.predict_variances(x)

plt.figure('Kall-Size-SMT')
plt.plot(samples[:, 0], snrs, 'o', label='Training data')
plt.plot(sizX, ypredSiz, label='Prediction', color='green')
plt.fill_between(np.ravel(sizX),
                 np.ravel(ypredSiz - 1.0 * np.sqrt(sigmaSiz)),
                 np.ravel(ypredSiz + 1.0 * np.sqrt(sigmaSiz)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Size')

print('Cutoff Predicting - SMT')
x[:, 0] = 69
x[:, 1] = cutX
x[:, 2] = 0.95
ypredCut, sigmaCut = gp2.predict_values(x), gp2.predict_variances(x)

plt.figure('Kall-Cutoff-SMT')
plt.plot(samples[:, 1], snrs, 'o', label='Training data')
plt.plot(cutX, ypredCut, label='Predictions', color='green')
plt.fill_between(np.ravel(cutX),
                 np.ravel(ypredCut - 1.0 * np.sqrt(sigmaCut)),
                 np.ravel(ypredCut + 1.0 * np.sqrt(sigmaCut)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Cutoff')

print('Threshold Predicting - SMT')
x = np.zeros((16, 3))
x[:, 0] = 69
x[:, 1] = 46
x[:, 2] = thrX

ypredThr, sigmaThr = gp2.predict_values(x), gp2.predict_variances(x)

plt.figure('Kall-Threshold-SMT')
plt.plot(samples[:, 2], snrs, 'o', label='Training data')
plt.plot(thrX, ypredThr, label='Prediction', color='green')
plt.fill_between(np.ravel(thrX),
                 np.ravel(ypredThr - 1.0 * np.sqrt(sigmaThr)),
                 np.ravel(ypredThr + 1.0 * np.sqrt(sigmaThr)),
                 color='lightgreen',
                 label='Confidence'
                 )
plt.legend()
plt.ylabel('SNR')
plt.xlabel('Threshold')

#-------------------------------------------------------------------------------
# plt.plot(x, y_pred, label='Prediction SkLearn', color='green')
# # plt.fill(np.concatenate([x.reshape(-1, 1), x.reshape(-1, 1)[::-1]]),
# #          np.concatenate([y_pred - 1.96 * sigma,
# #                          (y_pred + 1.96 * sigma)[::-1]]),
# #          label='Confidence SKL')
# plt.fill_between(
#     np.ravel(x),
#     np.ravel(y_pred - 1.0 * sigma.reshape(-1, 1)),
#     np.ravel(y_pred + 1.0 * sigma.reshape(-1, 1)),
#     color='lightgreen',
#     label='Confidence SKL'
# )
# plt.legend()

# plt.plot(testRange, snrs)
# scio.savemat('SNRsThr_test.mat', {'data': snrs})

# SMT Implementation
# sm = smtK(theta0=[1e-2], print_global=False)
# sm.set_training_values(cutSamples, snrs)
# sm.train()
#
# num = 1000
# x = np.linspace(0.001, 100, num=num)
# y = sm.predict_values(x)
# s2 = sm.predict_variances(x)
# dydx = sm.predict_derivatives(cutSamples, 0)
#
# oriSignal = scio.loadmat('SNRsCut_test.mat')['snr'].T
# ori = oriSignal[:100]
# plt.plot(ori, label='Original', color='red')
# plt.plot(cutSamples, snrs, 'o', label='Training data', color='orange')
# plt.plot(x, y, label='Prediction SMT', color='blue')
# # plt.fill_between(
# #     np.ravel(x),
# #     np.ravel(y - 3 * np.sqrt(s2)),
# #     np.ravel(y + 3 * np.sqrt(s2)),
# #     color='lightblue',
# #     label='Confidence SMT'
# # )
#
# # Sklearn Implementation
# # kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-6, 20.0))
# kernel = 1 * RBF(1.9, (1e-6, 20))
# gp = sklK(kernel=kernel, n_restarts_optimizer=9, alpha=1e-1)
# gp.fit(cutSamples.reshape(-1, 1), snrs.reshape(-1, 1))
# y_pred, sigma = gp.predict(x.reshape(-1, 1), return_std=True)
#
# plt.plot(x, y_pred, label='Prediction SkLearn', color='green')
# # plt.fill(np.concatenate([x.reshape(-1, 1), x.reshape(-1, 1)[::-1]]),
# #          np.concatenate([y_pred - 1.96 * sigma,
# #                          (y_pred + 1.96 * sigma)[::-1]]),
# #          label='Confidence SKL')
# plt.fill_between(
#     np.ravel(x),
#     np.ravel(y_pred - 1.0 * sigma.reshape(-1, 1)),
#     np.ravel(y_pred + 1.0 * sigma.reshape(-1, 1)),
#     color='lightgreen',
#     label='Confidence SKL'
# )
# plt.legend()
