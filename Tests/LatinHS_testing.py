from pyDOE2 import lhs
from scipy.stats import qmc, shapiro, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np


def vis2Methods():
    numPoints = 20
    # Via PyDOE
    samples1 = lhs(2, samples=numPoints, criterion='center', random_state=1234)
    x1 = [s[0] for s in samples1]
    y1 = [s[1] for s in samples1]

    # Via Scipy
    sampler = qmc.LatinHypercube(2, centered=True, seed=1234)
    samples2 = sampler.random(numPoints)
    x2 = [s[0] for s in samples2]
    y2 = [s[1] for s in samples2]

    plt.scatter(x1, y1, color='r', label='pyDOE')
    plt.scatter(x2, y2, label='SciPy')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('Variable Y')
    plt.xlabel('Variable X')
    plt.subplots_adjust(left=0.05, bottom=0.067, right=0.986, top=0.986)

    intervals = [inter / numPoints for inter in range(1, numPoints)]
    for interval in intervals:
        plt.axhline(interval, color='k')
        plt.axvline(interval, color='k')


def anal2Methods(vars=2, num=30):
    numPoints = 50
    metSci = []
    metDoe = []
    for i in range(num):
        # Sci
        sampler = qmc.LatinHypercube(vars, centered=True)
        s1 = sampler.random(numPoints)
        metSci.append(qmc.discrepancy(s1))

        # Doe
        s2 = lhs(vars, numPoints, criterion='center')
        metDoe.append(qmc.discrepancy(s2))

    metSci = np.array(metSci)
    metDoe = np.array(metDoe)

    print('Stats Sci:')
    print(f'Best: {metSci.min()}')
    print(f'Mean: {metSci.mean()}')
    print(f'Median: {metSci[int(num / 2)]}')
    print(f'Worst: {metSci.max()}')
    print(f'Std: {metSci.std()}')

    print('Stats Doe:')
    print(f'Best: {metDoe.min()}')
    print(f'Mean: {metDoe.mean()}')
    print(f'Median: {metDoe[int(num / 2)]}')
    print(f'Worst: {metDoe.max()}')
    print(f'Std: {metDoe.std()}')

    shapiro_statSci, shapiro_pSci = shapiro(metSci)
    print(f'Sci: {shapiro_statSci} - {shapiro_pSci}')
    shapiro_statDoe, shapiro_pDoe = shapiro(metDoe)
    print(f'Sci: {shapiro_statDoe} - {shapiro_pDoe}')

    if shapiro_pSci < 0.05 or shapiro_pDoe < 0.05:
        print('Wilcoxon')
        stats, p_value = mannwhitneyu(metSci, metDoe)
    else:
        print('T-student')
        stats, p_value = ttest_ind(metSci, metDoe, equal_var=False)

    print(f'Diff: {stats} - {p_value}')

    if p_value < 0.05:
        print('Significant differences')
    else:
        print('No significant differences')

    return metSci, metDoe


def scaleMethod(vars=2, numPoints=20):
    ranges = np.array([[1, 100],
                       [0.001, 128]])

    sampler = qmc.LatinHypercube(vars, centered=True)
    s1 = sampler.random(numPoints)
    scale_sample = qmc.scale(s1, ranges[:, 0], ranges[:, 1])

    plt.figure('Not Scaled')
    plt.scatter(s1[:, 0], s1[:, 1])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('Variable Y')
    plt.xlabel('Variable X')
    plt.subplots_adjust(left=0.05, bottom=0.067, right=0.986, top=0.986)

    incrV1 = (ranges[0, 1] - ranges[0, 0]) / numPoints
    incrV2 = (ranges[1, 1] - ranges[1, 0]) / numPoints

    intX, intY = ranges[0, 0], ranges[1, 0]

    intervals = [inter / numPoints for inter in range(1, numPoints)]
    for interval in intervals:
        plt.axhline(interval, color='k')
        plt.axvline(interval, color='k')

    plt.figure('Scaled')
    plt.scatter(scale_sample[:, 0], scale_sample[:, 1])
    plt.xlim(ranges[0, 0], ranges[0, 1])
    plt.ylim(ranges[1, 0], ranges[1, 1])
    plt.ylabel('Variable Y')
    plt.xlabel('Variable X')
    plt.subplots_adjust(left=0.05, bottom=0.067, right=0.986, top=0.986)

    incrV1 = (ranges[0, 1] - ranges[0, 0]) / numPoints
    incrV2 = (ranges[1, 1] - ranges[1, 0]) / numPoints

    intX, intY = ranges[0, 0], ranges[1, 0]

    for _ in range(numPoints - 1):
        intX += incrV1
        intY += incrV2
        plt.axhline(intY, color='k')
        plt.axvline(intX, color='k')


def scaleMethod_Cor(vars=2, numPoints=10):
    ranges = np.array([[1, 102],
                       [0.001, 128]])

    scale_sample = []
    valid = False
    while not valid:
        sampler = qmc.LatinHypercube(vars, centered=True)
        samples = sampler.random(numPoints)
        scale_sample = qmc.scale(samples, ranges[:, 0], ranges[:, 1])
        saved_samples = np.array(scale_sample)
        # Corrections
        correctionV1 = [int(val) if int(val) % 2 else int(val) + 1 for val in scale_sample[:, 0]]
        # Checking Corrections:
        valid = uniqueVals(correctionV1)

        scale_sample[:, 0] = correctionV1

    plt.scatter(saved_samples[:, 0], saved_samples[:, 1], label='Original')
    plt.scatter(scale_sample[:, 0], scale_sample[:, 1], label='Corrected')

    plt.xlim(ranges[0, 0], ranges[0, 1])
    plt.ylim(ranges[1, 0], ranges[1, 1])
    plt.ylabel('Cutoff Frequency')
    plt.xlabel('Filter size')
    plt.subplots_adjust(left=0.05, bottom=0.067, right=0.986, top=0.986)
    plt.legend()

    incrV1 = (ranges[0, 1] - ranges[0, 0]) / numPoints
    incrV2 = (ranges[1, 1] - ranges[1, 0]) / numPoints

    intX, intY = ranges[0, 0], ranges[1, 0]

    for _ in range(numPoints - 1):
        intX += incrV1
        intY += incrV2
        plt.axhline(intY, color='k')
        plt.axvline(intX, color='k')


def uniqueVals(listVar):
    seen = set()
    for x in listVar:
        if x in seen: return False
        seen.add(x)
    return True


def finalTest(vars=3, numPoints=50):
    ranges = np.array([[1, 102],
                       [0.001, 128],
                       [0.1, 2]])

    scale_sample = []
    valid = False
    while not valid:
        sampler = qmc.LatinHypercube(vars, centered=True)
        samples = sampler.random(numPoints)
        scale_sample = qmc.scale(samples, ranges[:, 0], ranges[:, 1])
        # Corrections
        correctionV1 = [int(val) if int(val) % 2 else int(val) + 1 for val in scale_sample[:, 0]]
        # Checking Corrections:
        valid = uniqueVals(correctionV1)

        scale_sample[:, 0] = correctionV1
    plt.figure('3D')
    ax = plt.axes(projection='3d')
    ax.scatter3D(scale_sample[:, 0], scale_sample[:, 1], scale_sample[:, 2], depthshade=False)
    ax.set_xlim(ranges[0, 0], ranges[0, 1])
    ax.set_xlabel('Size')
    ax.set_ylim(ranges[1, 0], ranges[1, 1])
    ax.set_ylabel('Cutoff')
    ax.set_zlim(ranges[2, 0], ranges[2, 1])
    ax.set_zlabel('Threshold')

    plt.figure('S-C')
    plt.scatter(scale_sample[:, 0], scale_sample[:, 1])
    plt.xlim(ranges[0, 0], ranges[0, 1])
    plt.ylim(ranges[1, 0], ranges[1, 1])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.986, top=0.986)

    plt.xlabel('Size')
    plt.ylabel('Cutoff')

    incrV1 = (ranges[0, 1] - ranges[0, 0]) / numPoints
    incrV2 = (ranges[1, 1] - ranges[1, 0]) / numPoints

    intX, intY = ranges[0, 0], ranges[1, 0]

    for _ in range(numPoints - 1):
        intX += incrV1
        intY += incrV2
        plt.axhline(intY, linestyle='--', color='k', linewidth=1)
        plt.axvline(intX, linestyle='--', color='k', linewidth=1)

    plt.figure('S-T')
    plt.scatter(scale_sample[:, 0], scale_sample[:, 2])
    plt.xlim(ranges[0, 0], ranges[0, 1])
    plt.ylim(ranges[2, 0], ranges[2, 1])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.986, top=0.986)

    plt.xlabel('Size')
    plt.ylabel('Threshold')

    incrV1 = (ranges[0, 1] - ranges[0, 0]) / numPoints
    incrV2 = (ranges[2, 1] - ranges[2, 0]) / numPoints

    intX, intY = ranges[0, 0], ranges[2, 0]

    for _ in range(numPoints - 1):
        intX += incrV1
        intY += incrV2
        plt.axhline(intY, linestyle='--', color='k', linewidth=1)
        plt.axvline(intX, linestyle='--', color='k', linewidth=1)

    plt.figure('C-T')
    plt.scatter(scale_sample[:, 1], scale_sample[:, 2])
    plt.xlim(ranges[1, 0], ranges[1, 1])
    plt.ylim(ranges[2, 0], ranges[2, 1])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.986, top=0.986)

    plt.xlabel('Cutoff')
    plt.ylabel('Threshold')

    incrV1 = (ranges[1, 1] - ranges[1, 0]) / numPoints
    incrV2 = (ranges[2, 1] - ranges[2, 0]) / numPoints

    intX, intY = ranges[1, 0], ranges[2, 0]

    for _ in range(numPoints - 1):
        intX += incrV1
        intY += incrV2
        plt.axhline(intY, linestyle='--', color='k', linewidth=1)
        plt.axvline(intX, linestyle='--', color='k', linewidth=1)

scaleMethod()