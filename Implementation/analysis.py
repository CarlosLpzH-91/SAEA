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
test = 2

# Variables
name1 = 'KADE-10%'
name2 = 'KADE-5%'
name3 = 'KADE-2%'
hist1 = []
hist2 = []
hist3 = []
times1 = []
times2 = []
times3 = []

# Signal
original_signal = scio.loadmat('../Signals/Tests/Signal_S_E2_2.mat')['signal'].T[0]
shift = original_signal.min()
original_signal = original_signal - shift
scale = original_signal.max() * 2
freq = 1000

# Constants
# General
maximization = True
ranges = [[16, 80],
          [20, 80],
          [0.8, 1.1]]
# DE related
size = 50
generations = [50, 100, 100]  # [name1, name2 - name3] - [200, 100, 300]
cr = 0.8
fx = 0.5
# KADE related
update = [1, 1, 1]
prc_select = [0.1, 0.05, 0.02]
c_v = 1.0
c_v_bouds = (1e-5, 1e5)
rbf_ls = np.ones(3)
rbf_ls_bounds = [(1e-2, 1e6), (1e-2, 1e8), (1e-5, 1e2)]
n_rest = 50
a = 1e-3

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
    print(f'-------- Doing {name1} --------')
    stime = time.time()
    # res1, _, _ = DE(samples=init_samples, num_gen=generations[0], size=size, cr=cr, fx=fx,
    #                 ranges=ranges, signal=original_signal, scale=scale, fs=freq)
    res1, _, _, _, _, _, _, _ = KADE(signal=original_signal, scale=scale, fs=freq,
                                     ranges=ranges, samples=init_samples, size=size,
                                     num_gen=generations[0], cr=cr, fx=fx, n_update=update[0],
                                     prc_selection=prc_select[0], c_v=c_v, c_v_bounds=c_v_bouds,
                                     rbf_ls=rbf_ls, rbf_ls_bounds=rbf_ls_bounds, n_rest=n_rest,
                                     a=a, re_evaluate_km=False, plot_vars=False, verbose=False)
    hist1.append(res1)
    times1.append(time.time() - stime)
    print(f'-------- Doing {name2} --------')
    stime = time.time()
    res2, _, _, _, _, _, _, _ = KADE(signal=original_signal, scale=scale, fs=freq,
                                     ranges=ranges, samples=init_samples, size=size,
                                     num_gen=generations[1], cr=cr, fx=fx, n_update=update[1],
                                     prc_selection=prc_select[1], c_v=c_v, c_v_bounds=c_v_bouds,
                                     rbf_ls=rbf_ls, rbf_ls_bounds=rbf_ls_bounds, n_rest=n_rest,
                                     a=a, re_evaluate_km=False, plot_vars=False, verbose=False)
    hist2.append(res2)
    times2.append(time.time() - stime)
    print(f'-------- Doing {name3} --------')
    stime = time.time()
    res3, _, _, _, _, _, _, _ = KADE(signal=original_signal, scale=scale, fs=freq,
                                     ranges=ranges, samples=init_samples, size=size,
                                     num_gen=generations[2], cr=cr, fx=fx, n_update=update[2],
                                     prc_selection=prc_select[2], c_v=c_v, c_v_bounds=c_v_bouds,
                                     rbf_ls=rbf_ls, rbf_ls_bounds=rbf_ls_bounds, n_rest=n_rest,
                                     a=a, re_evaluate_km=False, plot_vars=False, verbose=False)
    hist3.append(res3)
    times3.append(time.time() - stime)

    print('\n------- Times Resume -------')
    print(f'{name1}: {times1[e]}')
    print(f'{name2}: {times2[e]}')
    print(f'{name3}: {times3[e]}')
    print(f' Global: {time.time() - global_stime}\n')

print(f'Total time: {time.time() - global_stime}')

print('\n---Analysis---')
print('--Executions')
# DE
with open(f'Evidence/Executions/Execs_{name1}_{test}.txt', 'w', newline='') as file:
    for n, exe in enumerate(hist1):
        file.write(f'Exec {n + 1}: {exe[-1].report()}\n')
# KADE-2%
with open(f'Evidence/Executions/Execs_{name2}_{test}.txt', 'w', newline='') as file:
    for n, exe in enumerate(hist2):
        file.write(f'Exec {n + 1}: {exe[-1].report()}\n')
# KADE-5-10%
with open(f'Evidence/Executions/Execs_{name3}_{test}.txt', 'w', newline='') as file:
    for n, exe in enumerate(hist3):
        file.write(f'Exec {n + 1}: {exe[-1].report()}\n')


print('--Statistics')
hist1.sort(key=lambda x: x[-1].aptitude, reverse=maximization)
hist2.sort(key=lambda x: x[-1].aptitude, reverse=maximization)
hist3.sort(key=lambda x: x[-1].aptitude, reverse=maximization)

apts1 = [e[-1].aptitude for e in hist1]
apts2 = [e[-1].aptitude for e in hist2]
apts3 = [e[-1].aptitude for e in hist3]

alpha = 0.05

with open(f'Evidence/Executions/Stcs_{test}.txt', 'w', newline='') as file:
    file.write('-----DE\n')
    file.write(f'Best: {apts1[0]}\n')
    file.write(f'Mean: {statistics.mean(apts1)}\n')
    file.write(f'Median: {statistics.median_low(apts1)}\n')
    file.write(f'Worst: {apts1[-1]}\n')
    file.write(f'StDev: {statistics.stdev(apts1)}\n')

    file.write('-----KADE-2%\n')
    file.write(f'Best: {apts2[0]}\n')
    file.write(f'Mean: {statistics.mean(apts2)}\n')
    file.write(f'Median: {statistics.median_low(apts2)}\n')
    file.write(f'Worst: {apts2[-1]}\n')
    file.write(f'StDev: {statistics.stdev(apts2)}\n')

    file.write('-----KADE-5-10%\n')
    file.write(f'Best: {apts3[0]}\n')
    file.write(f'Mean: {statistics.mean(apts3)}\n')
    file.write(f'Median: {statistics.median_low(apts3)}\n')
    file.write(f'Worst: {apts3[-1]}\n')
    file.write(f'StDev: {statistics.stdev(apts3)}\n')

    file.write('Shapiro-Wilk test\n')
    stat, pval = scsts.shapiro(apts2)
    file.write(f'DE = Stat: {stat} - p-value: {pval}\n')
    stat, pval = scsts.shapiro(apts2)
    file.write(f'KADE-2% = Stat: {stat} - p-value: {pval}\n')
    stat, pval = scsts.shapiro(apts3)
    file.write(f'KADE-5-10% = Stat: {stat} - p-value: {pval}\n')

    file.write('Wilcoxon rank-sum: DE vs KADE-2%\n')
    stat_u, p_value = scsts.mannwhitneyu(apts1, apts2)
    file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    if p_value < alpha:
        file.write('Significant difference between DE and KADE-2% (Reject H0)\n')
        print('Significant difference between DE and KADE-2% (Reject H0)')
    else:
        file.write('No significant difference between DE and KADE-2% (Accept H0)\n')
        print('No significant difference between DE and KADE-2% (Accept H0)')

    file.write('Wilcoxon rank-sum: DE vs KADE-5-10%\n')
    stat_u, p_value = scsts.mannwhitneyu(apts1, apts3)
    file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    if p_value < alpha:
        file.write('Significant difference between DE and KADE-5-10% (Reject H0)\n')
        print('Significant difference between DE and KADE-5-10% (Reject H0)')
    else:
        file.write('No significant difference between DE and KADE-5-10% (Accept H0)\n')
        print('No significant difference between DE and KADE-5-10% (Accept H0)')

    file.write('Wilcoxon rank-sum: KADE-2% vs KADE-5-10%\n')
    stat_u, p_value = scsts.mannwhitneyu(apts2, apts3)
    file.write(f'Stat: {stat_u} - p-value: {p_value}\n')
    if p_value < alpha:
        file.write('Significant difference between KADE-2% and KADE-5-10% (Reject H0)\n')
        print('Significant difference between KADE-2% and KADE-5-10% (Reject H0)')
    else:
        file.write('No significant difference between KADE-2% and KADE-5-10% (Accept H0)\n')
        print('No significant difference between KADE-2% and KADE-5-10% (Accept H0)')

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
    file.write(f'DE: {np.mean(times1)}\n')
    file.write(f'KADE-2%: {np.mean(times2)}\n')
    file.write(f'KADE-5-10%: {np.mean(times3)}')

# Best Results
with open(f'Evidence/Executions/Bests_{test}.txt', 'w', newline='') as file:
    file.write('DE\n')
    file.write(hist1[0][-1].report())
    file.write('\n')

    file.write('KADE-2%\n')
    file.write(hist2[0][-1].report())
    file.write('\n')

    file.write('KADE-5-10%\n')
    file.write(hist3[0][-1].report())
    file.write('\n')

print('Analysis over the best results')
with open(f'Evidence/Executions/Best_DE_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist1[0]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Best_KADE-2%_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist2[0]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Best_KADE-5-10%_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist3[0]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

print('Analysis over the median')
pos_median = math.ceil(execs / 2) - 1
with open(f'Evidence/Executions/Median_DE_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist1[pos_median]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Median_KADE-2%_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist2[pos_median]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

with open(f'Evidence/Executions/Median_KADE-5-10%_{test}.txt', 'w', newline='') as file:
    for indx, gen_r in enumerate(hist3[pos_median]):
        file.write(f'Gen {indx}:\n {gen_r.report()}\n')

# Figures
best_apts1 = [g.aptitude for g in hist1[0]]
best_apts2 = [g.aptitude for g in hist2[0]]
best_apts3 = [g.aptitude for g in hist3[0]]

med_apts1 = [g.aptitude for g in hist1[pos_median]]
med_apts2 = [g.aptitude for g in hist2[pos_median]]
med_apts3 = [g.aptitude for g in hist3[pos_median]]


plt.figure(f'Aptitudes-Best_{test}', figsize=[12, 6])
plt.plot(best_apts1, "-*", markersize=2.5, linewidth=1, label='DE/rand/1/bin')
plt.plot(best_apts2, "-*", markersize=2.5, linewidth=1, label='KADE-2%')
plt.plot(best_apts3, "-*", markersize=2.5, linewidth=1, label='KADE-5-10%')

plt.xlabel('Generations')
plt.ylabel('f(x)')
plt.legend()

plt.figure(f'Aptitudes-Median_{test}', figsize=[12, 6])
plt.plot(med_apts1, "-*", markersize=2.5, linewidth=1, label='DE/rand/1/bin')
plt.plot(med_apts2, "-*", markersize=2.5, linewidth=1, label='KADE-2%')
plt.plot(med_apts3, "-*", markersize=2.5, linewidth=1, label='KADE-5-10%')

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
