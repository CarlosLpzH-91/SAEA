import statistics
import scipy.stats as scsts
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as scio
import time
from Implementation.kade import KADE
from Implementation.de import DE
from Implementation.lhs import LHS

# Number of executions
execs = 30
test = 'EEG'

# Variables
name1 = 'DE_PT'
# name2 = 'KADE_2_5%_PT_U'
name2 = 'KADE2_5%_PT_EI'
name3 = 'KADE_5%_PT_U'
hist1 = []
hist2 = []
hist3 = []
# hist4 = []
times1 = []
times2 = []
times3 = []
# times4 = []

# Signal
# original_signal = scio.loadmat('../Signals/Tests/Signal_S_E2_2.mat')['signal'].T[0]
original_signal = scio.loadmat('../Signals/chb01_09_C1.mat')['c1'][0][:10000]
shift = original_signal.min()
original_signal = original_signal - shift
scale = original_signal.max() * 2
# freq = 1000
freq = 256

# Constants
# General
maximization = True
ranges = [[16, 80],
          [20, 80],
          [0.8, 1.1]]
# DE related
size = 50
generations = [200, 100, 50]  # [name1, name2 - name3] - [200, 100, 300]
cr = [2.86, 3.49, 2.43]
fx = [0.65, 0.39, 0.20]
# KADE related
update = [2, 1]
prc_select = 0.05
c_v = 1.0
c_v_bouds = (1e-5, 1e5)
rbf_ls = np.ones(3)
rbf_ls_bounds_1 = [(1e-2, 5e9), (1e-2, 5e9), (1e-5, 3e9)]
rbf_ls_bounds_2 = [(1e-2, 5e9), (1e-2, 6e9), (1e-5, 4e8)]
# n_rest = 50
# a = 1e-3
n_rest = [40, 20]
a = [1e-5, 1e-6]

# Initial sampling
# init_samples = LHS(numPoints=size,
#                    rangeSize=ranges[0],
#                    rangeCut=ranges[1],
#                    rangeThreshold=ranges[2])
global_stime = time.time()
for e in range(execs):
    # Initial sampling
    init_samples = LHS(numPoints=size,
                       rangeSize=ranges[0],
                       rangeCut=ranges[1],
                       rangeThreshold=ranges[2])

    print(f'------------ Execution {e} --------------')
    print(f'-------- Doing {name2} --------')
    stime = time.time()
    res2, _, _, _, _, _, _, _ = KADE(signal=original_signal, scale=scale, fs=freq,
                                     ranges=ranges, samples=init_samples, tot_evals=0, size=size, select_option='E',
                                     num_gen=generations[1], cr=cr[1], fx=fx[1], n_update=update[0],
                                     prc_selection=prc_select, c_v=c_v, c_v_bounds=c_v_bouds, stop='Gen',
                                     rbf_ls=rbf_ls, rbf_ls_bounds=rbf_ls_bounds_1, n_rest=n_rest[0],
                                     a=a[0], re_evaluate_km=False, plot_vars=False, verbose=False)
    hist2.append(res2)
    times2.append(time.time() - stime)

    # print(f'-------- Doing {name3} --------')
    # stime = time.time()
    # res3, _, _, _, _, _, _, _ = KADE(signal=original_signal, scale=scale, fs=freq,
    #                                  ranges=ranges, samples=init_samples, tot_evals=0, size=size, select_option='E',
    #                                  num_gen=generations[1], cr=cr[1], fx=fx[1], n_update=update[0],
    #                                  prc_selection=prc_select, c_v=c_v, c_v_bounds=c_v_bouds, stop='Gen',
    #                                  rbf_ls=rbf_ls, rbf_ls_bounds=rbf_ls_bounds_1, n_rest=n_rest[0],
    #                                  a=a[0], re_evaluate_km=False, plot_vars=False, verbose=False)
    # hist3.append(res3)
    # times3.append(time.time() - stime)
    print(f'-------- Doing {name3} --------')
    stime = time.time()
    res3, _, _, _, _, _, _, _ = KADE(signal=original_signal, scale=scale, fs=freq,
                                     ranges=ranges, samples=init_samples, tot_evals=0, size=size, select_option='U',
                                     num_gen=generations[2], cr=cr[2], fx=fx[2], n_update=update[1],
                                     prc_selection=prc_select, c_v=c_v, c_v_bounds=c_v_bouds, stop='Gen',
                                     rbf_ls=rbf_ls, rbf_ls_bounds=rbf_ls_bounds_2, n_rest=n_rest[1],
                                     a=a[1], re_evaluate_km=False, plot_vars=False, verbose=False)
    hist3.append(res3)
    times3.append(time.time() - stime)
    print(f'-------- Doing {name1} --------')
    stime = time.time()
    res1, _, _ = DE(samples=init_samples, num_gen=generations[0], size=size, cr=cr[0], fx=fx[0],
                    ranges=ranges, signal=original_signal, scale=scale, fs=freq, total_eval=0, stop='Gen')
    hist1.append(res1)
    times1.append(time.time() - stime)

    print('\n------- Times Resume -------')
    print(f'{name1}: {times1[e]} - {res1[-1].report()}')
    print(f'{name2}: {times2[e]} - {res2[-1].report()}')
    print(f'{name3}: {times3[e]} - {res3[-1].report()}')
    # print(f'{name4}: {times3[e]} - {res4[-1].report()}')
    print(f' Global: {time.time() - global_stime}\n')

print(f'Total time: {time.time() - global_stime}')

print('\n---Analysis---')
print('--Executions')

with open(f'Evidence/Executions/Execs_{name1}_{test}.txt', 'w', newline='') as file:
    for n, exe in enumerate(hist1):
        file.write(f'Exec {n + 1}: {exe[-1].report()}\n')

with open(f'Evidence/Executions/Execs_{name2}_{test}.txt', 'w', newline='') as file:
    for n, exe in enumerate(hist2):
        file.write(f'Exec {n + 1}: {exe[-1].report()}\n')

with open(f'Evidence/Executions/Execs_{name3}_{test}.txt', 'w', newline='') as file:
    for n, exe in enumerate(hist3):
        file.write(f'Exec {n + 1}: {exe[-1].report()}\n')

# with open(f'Evidence/Executions/Execs_{name4}_{test}.txt', 'w', newline='') as file:
#     for n, exe in enumerate(hist4):
#         file.write(f'Exec {n + 1}: {exe[-1].report()}\n')


print('--Statistics')
hist1.sort(key=lambda x: x[-1].aptitude, reverse=maximization)
hist2.sort(key=lambda x: x[-1].aptitude, reverse=maximization)
hist3.sort(key=lambda x: x[-1].aptitude, reverse=maximization)
# hist4.sort(key=lambda x: x[-1].aptitude, reverse=maximization)

apts1 = [e[-1].aptitude for e in hist1]
apts2 = [e[-1].aptitude for e in hist2]
apts3 = [e[-1].aptitude for e in hist3]
# apts4 = [e[-1].aptitude for e in hist4]

alpha = 0.05

with open(f'Evidence/Executions/Stcs_{test}.txt', 'w', newline='') as file:
    file.write(f'-----{name1}\n')
    file.write(f'Best: {apts1[0]}\n')
    file.write(f'Mean: {statistics.mean(apts1)}\n')
    file.write(f'Median: {statistics.median_low(apts1)}\n')
    file.write(f'Worst: {apts1[-1]}\n')
    file.write(f'StDev: {statistics.stdev(apts1)}\n')

    file.write(f'-----{name2}\n')
    file.write(f'Best: {apts2[0]}\n')
    file.write(f'Mean: {statistics.mean(apts2)}\n')
    file.write(f'Median: {statistics.median_low(apts2)}\n')
    file.write(f'Worst: {apts2[-1]}\n')
    file.write(f'StDev: {statistics.stdev(apts2)}\n')

    file.write(f'-----{name3}\n')
    file.write(f'Best: {apts3[0]}\n')
    file.write(f'Mean: {statistics.mean(apts3)}\n')
    file.write(f'Median: {statistics.median_low(apts3)}\n')
    file.write(f'Worst: {apts3[-1]}\n')
    file.write(f'StDev: {statistics.stdev(apts3)}\n')

    # file.write(f'-----{name4}\n')
    # file.write(f'Best: {apts4[0]}\n')
    # file.write(f'Mean: {statistics.mean(apts4)}\n')
    # file.write(f'Median: {statistics.median_low(apts4)}\n')
    # file.write(f'Worst: {apts4[-1]}\n')
    # file.write(f'StDev: {statistics.stdev(apts4)}\n')

    file.write('Shapiro-Wilk test\n')
    stat, pval = scsts.shapiro(apts1)
    file.write(f'{name1} = Stat: {stat} - p-value: {pval}\n')
    stat, pval = scsts.shapiro(apts2)
    file.write(f'{name2} = Stat: {stat} - p-value: {pval}\n')
    stat, pval = scsts.shapiro(apts3)
    file.write(f'{name3} = Stat: {stat} - p-value: {pval}\n')
    # stat, pval = scsts.shapiro(apts4)
    # file.write(f'{name4} = Stat: {stat} - p-value: {pval}\n')

    file.write(f'Wilcoxon rank-sum: {name1} vs {name2}\n')
    stat_u, p_value = scsts.mannwhitneyu(apts1, apts2)
    file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    if p_value < alpha:
        file.write(f'Significant difference between {name1} and {name2} (Reject H0)\n')
        print(f'Significant difference between {name1} and {name2} (Reject H0)')
    else:
        file.write(f'No significant difference between {name1} and {name2} (Accept H0)\n')
        print(f'No significant difference between {name1} and {name2} (Accept H0)')

    file.write(f'Wilcoxon rank-sum: {name1} vs {name3}\n')
    stat_u, p_value = scsts.mannwhitneyu(apts1, apts3)
    file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    if p_value < alpha:
        file.write(f'Significant difference between {name1} and {name3} (Reject H0)\n')
        print(f'Significant difference between {name1} and {name3} (Reject H0)')
    else:
        file.write(f'No significant difference between {name1} and {name3} (Accept H0)\n')
        print(f'No significant difference between {name1} and {name3} (Accept H0)')

    # file.write(f'Wilcoxon rank-sum: {name1} vs {name4}\n')
    # stat_u, p_value = scsts.mannwhitneyu(apts1, apts4)
    # file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    # if p_value < alpha:
    #     file.write(f'Significant difference between {name1} and {name4} (Reject H0)\n')
    #     print(f'Significant difference between {name1} and {name4} (Reject H0)')
    # else:
    #     file.write(f'No significant difference between {name1} and {name4} (Accept H0)\n')
    #     print(f'No significant difference between {name1} and {name4} (Accept H0)')

    file.write(f'Wilcoxon rank-sum: {name2} vs {name3}\n')
    stat_u, p_value = scsts.mannwhitneyu(apts2, apts3)
    file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    if p_value < alpha:
        file.write(f'Significant difference between {name2} and {name3} (Reject H0)\n')
        print(f'Significant difference between {name2} and {name3} (Reject H0)')
    else:
        file.write(f'No significant difference between {name2} and {name3} (Accept H0)\n')
        print(f'No significant difference between {name2} and {name3} (Accept H0)')

    # file.write(f'Wilcoxon rank-sum: {name2} vs {name4}\n')
    # stat_u, p_value = scsts.mannwhitneyu(apts2, apts4)
    # file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    # if p_value < alpha:
    #     file.write(f'Significant difference between {name2} and {name4} (Reject H0)\n')
    #     print(f'Significant difference between {name2} and {name4} (Reject H0)')
    # else:
    #     file.write(f'No significant difference between {name2} and {name4} (Accept H0)\n')
    #     print(f'No significant difference between {name2} and {name4} (Accept H0)')

    # file.write(f'Wilcoxon rank-sum: {name3} vs {name4}\n')
    # stat_u, p_value = scsts.mannwhitneyu(apts3, apts4)
    # file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    # if p_value < alpha:
    #     file.write(f'Significant difference between {name3} and {name4} (Reject H0)\n')
    #     print(f'Significant difference between {name3} and {name4} (Reject H0)')
    # else:
    #     file.write(f'No significant difference between {name3} and {name4} (Accept H0)\n')
    #     print(f'No significant difference between {name3} and {name4} (Accept H0)')

    file.write('Friedman test\n')
    stat_u, p_value = scsts.friedmanchisquare(apts1, apts2, apts3)
    file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    if p_value < alpha:
        file.write('Significant difference (Reject H0)\n')
        print('Significant difference')
    else:
        file.write('No significant difference (Accept H0)\n')
        print('No significant difference')

    file.write('Times\n')
    file.write(f'{name1}: {np.mean(times1)}\n')
    file.write(f'{name2}: {np.mean(times2)}\n')
    file.write(f'{name3}: {np.mean(times3)}\n')
    # file.write(f'{name4}: {np.mean(times4)}')

# Best Results
with open(f'Evidence/Executions/Bests_{test}.txt', 'w', newline='') as file:
    file.write(f'{name1}\n')
    file.write(hist1[0][-1].report())
    file.write('\n')

    file.write(f'{name2}\n')
    file.write(hist2[0][-1].report())
    file.write('\n')

    file.write(f'{name3}\n')
    file.write(hist3[0][-1].report())
    file.write('\n')

    # file.write(f'{name4}\n')
    # file.write(hist4[0][-1].report())
    # file.write('\n')

print('Analysis over the best results')
with open(f'Evidence/Executions/Best_{name1}_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist1[0]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Best_{name2}_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist2[0]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Best_{name3}_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist3[0]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

# with open(f'Evidence/Executions/Best_{name4}_{test}.txt', 'w', newline='') as file:
#     for indx, gen_r in enumerate(hist4[0]):
#         file.write(f'Gen {indx}:\n {gen_r.report()}\n')

print('Analysis over the median')
pos_median = math.ceil(execs / 2) - 1
with open(f'Evidence/Executions/Median_{name1}_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist1[pos_median]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Median_{name2}_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist2[pos_median]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Median_{name3}_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist3[pos_median]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

# with open(f'Evidence/Executions/Median_{name4}_{test}.txt', 'w', newline='') as file:
#     for indx, gen_r in enumerate(hist4[pos_median]):
#         file.write(f'Gen {indx}:\n {gen_r.report()}\n')

# Figures
best_apts1 = [g.aptitude for g in hist1[0]]
best_apts2 = [g.aptitude for g in hist2[0]]
best_apts3 = [g.aptitude for g in hist3[0]]
# best_apts4 = [g.aptitude for g in hist4[0]]

med_apts1 = [g.aptitude for g in hist1[pos_median]]
med_apts2 = [g.aptitude for g in hist2[pos_median]]
med_apts3 = [g.aptitude for g in hist3[pos_median]]
# med_apts4 = [g.aptitude for g in hist4[pos_median]]


plt.figure(f'Aptitudes-Best_{test}', figsize=[12, 6])
plt.plot(best_apts1, "-*", markersize=2.5, linewidth=1, label=f'{name1}')
plt.plot(best_apts2, "-*", markersize=2.5, linewidth=1, label=f'{name2}')
plt.plot(best_apts3, "-*", markersize=2.5, linewidth=1, label=f'{name3}')
# plt.plot(best_apts4, "-*", markersize=2.5, linewidth=1, label=f'{name4}')

plt.xlabel('Generations')
plt.ylabel('f(x)')
plt.legend()

plt.figure(f'Aptitudes-Median_{test}', figsize=[12, 6])
plt.plot(med_apts1, "-*", markersize=2.5, linewidth=1, label=f'{name1}')
plt.plot(med_apts2, "-*", markersize=2.5, linewidth=1, label=f'{name2}')
plt.plot(med_apts3, "-*", markersize=2.5, linewidth=1, label=f'{name3}')
# plt.plot(med_apts4, "-*", markersize=2.5, linewidth=1, label=f'{name4}')

plt.xlabel('Generations')
plt.ylabel('f(x)')
plt.legend()

plt.show()


# Save all best aptitudes
scio.savemat(f'Evidence/Executions/best_{test}.mat', {name1: apts1, name2: apts2, name3: apts3})

# Save best execution generations
scio.savemat(f'Evidence/Executions/best_gen_{test}.mat', {name1: best_apts1, name2: best_apts2, name3: best_apts3})

# Save med execution generations
scio.savemat(f'Evidence/Executions/med_gen_{test}.mat', {name1: med_apts1, name2: med_apts2, name3: med_apts3})
