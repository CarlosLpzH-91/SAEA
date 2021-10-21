import random
import scipy.io as scio
import math
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy import signal
from BaseDatos.Manipulacion.BSA_Algorithm import Function, corrections, BSA


# # BCHM Rand (Position)
# def rand(low, high):
#     return low + (random.random() * (high - low))


# def corrections(representation, ranges):
#     """
#
#     :param list representation: [0] = size
#                                 [1] = cutoff
#                                 [2] = beta
#                                 [3] = threshold
#     :return:
#     """
#     for indx, (value, v_range) in enumerate(zip(representation, ranges)):
#         if value < v_range[0] or value > v_range[1]:
#             representation[indx] = rand(v_range[0], v_range[1])
#
#     size_ceil = math.ceil(representation[0])
#
#     size = size_ceil if size_ceil % 2 else size_ceil + 1
#     cutoff = representation[1]
#     beta = math.ceil(representation[2])
#     threshold = representation[3]
#
#     return [size, cutoff, beta, threshold]


# class Function:
#     def __init__(self, fs=256.0):
#         self.ranges = [[1, 31],
#                        [0.000001, fs / 2],
#                        [1, 40],
#                        [0, 1]]
#
#         self.signal = scio.loadmat('SEIZURES/TRAIN/PR/PR_chb01_04.mat')['data'][0][:1280]
#         self.scale = max(self.signal) * 2
#         self.fs = fs
#
#     def valid_representation(self):
#         x1 = random.randint(self.ranges[0][0], self.ranges[0][1])
#         x2 = random.uniform(self.ranges[1][0], self.ranges[1][1])
#         x3 = random.randint(self.ranges[2][0], self.ranges[2][1])
#         x4 = random.random()
#
#         representation = corrections([x1, x2, x3, x4], self.ranges)
#
#         return representation
#
#     def get_value(self, values):
#         """
#
#         :param list values: values[0] = size
#                             values[1] = cutoff
#                             values[2] = beta
#                             values[3] = threshold
#         :return:
#         """
#         size = values[0]
#         cutoff = values[1]
#         beta = values[2]
#         threshold = values[3]
#         # fir_filter = firwin(size, 60, window=('kaiser', beta), fs=256.0)
#         fir_filter = signal.firwin(size, cutoff, window=('kaiser', beta), fs=self.fs)
#         spiker = BSA(fir_filter, threshold, self.scale)
#
#         encoded, shift = spiker.encode(self.signal)
#         decoded = spiker.decode(encoded, shift)
#
#         snr = spiker.snr(self.signal, decoded, fs=self.fs)
#         # print(f'Val: {values} - {snr}')
#         return snr, self.signal, decoded, fir_filter, spiker, encoded


class Individual:
    def __init__(self, function, rep=None):
        self.function = function

        if rep is None:
            self.representation = self.function.valid_representation()
        else:
            self.representation = corrections(rep, self.function.ranges_)

        # self.aptitude, self.original, self.decoded, self.filter, self.spiker, self.encoded, self.metrics = \
        #     self.function.get_value(self.representation)
        self.aptitude, self.original, self.filter, \
            self.spiker, self.metrics = self.function.get_value(self.representation)

    # def show(self, gen, exec, save=True):
    #     plt.figure(figsize=[10, 6])
    #     plt.plot(self.original, '-', linewidth=1, label='Original')
    #     # plt.plot(self.decoded, '--', linewidth=1, label='DE/rand/1/bin')
    #     plt.plot(self.decoded, '--', linewidth=1, label='Decoded')
    #
    #     size = math.ceil(self.representation[0])
    #     cutoff = self.representation[1]
    #     beta = math.ceil(self.representation[2])
    #     threshold = self.representation[3]
    #     objective = self.aptitude
    #
    #     plt.legend()
    #     # plt.suptitle(f'Best of Generation: {gen}', weight='bold')
    #     # plt.title(f'\n Values: [Size: {size}, Cutoff: {cutoff}, Beta: {beta}, Threshold: {threshold}]'
    #     #           f' \n RMSE: {objective}', size='medium')
    #     # plt.xlabel(f'Time[s]')
    #     # plt.ylabel('Voltage [uV]')
    #     plt.xlabel(f'Sample Time')
    #     plt.ylabel('Signal Value')
    #
    #     if save:
    #         plt.savefig(f'Evidence/DE/Exec{exec}/Gen{gen}.png')
    #         plt.close()
    #     else:
    #         plt.show()

    def report(self):
        return f'{self.representation} f(x) = {self.aptitude} - {[round(np.mean(metric), 4) for metric in self.metrics]}'


def operator(cr, fx, target, v1, v2, v3):
    # print(f'Target: {target}')
    # print(f'V1: {v1}')
    # print(f'V2: {v2}')
    # print(f'V3: {v3}')

    # Generate Jrand value
    j_rand = random.randint(0, len(target) - 1)
    # print(f'Jrand: {j_rand}')

    # Initialize empty trial vector
    trial = []

    # Iterate over every value of vectors
    for j, (i, r1, r2, r3) in enumerate(zip(target, v1, v2, v3)):
        # print(f'J: {j}')
        # Generate rand_j value
        rand_j = random.random()
        # print(f'Randj: {rand_j}')

        # Verify conditions and form trial vector
        if rand_j < cr or j == j_rand:
            # print('Mutation')
            trial.append(r1 + fx * (r2 - r3))
        else:
            trial.append(i)

    # Return trial vector
    # print(f'Trial: {trial}')
    return trial


def DE(function, size, num_gen, cr, fx, show_last=True, exec=0, maximization=True):
    # plt.ioff()
    # Generation counter
    gen = 0
    # Evaluation counter
    evals = 0
    # Best's solutions
    bests = []

    # ---------------------- Initial population (Rand)
    population = [Individual(function) for _ in range(size)]

    # ---------------------- Eval population
    # Increase evaluation counter
    evals += size

    # Sort population by feasible and aptitude
    if maximization:
        population.sort(key=lambda x: x.aptitude, reverse=True)
    else:
        population.sort(key=lambda x: x.aptitude)
    # Max
    # population.sort(key=lambda x: x.aptitude, reverse=True)
    # Min
    # population.sort(key=lambda x: x.aptitude)

    # Print all initial results
    # print(f'Initial Representations:\t{[ind.representation for ind in population]}')
    # print(f'Initial Aptitudes:\t{[ind.aptitude for ind in population]}')
    # print(f'Initial Restrictions:\t{[ind.restrictions for ind in population]}')
    # print(f'Initial Validations:\t{[ind.valid for ind in population]}')

    # Report the best solution in initial population.
    print(f'Gen {gen}:\n {population[0].report()}')

    # Save best of Generation 0
    bests.append(population[0])

    while gen != num_gen:
        # if not gen % 10:
        #     population[0].show(gen, exec)
        # Iterate over population
        for i in range(size):
            # Get target vector
            target = population[i]

            # Narrow choices for guests vectors
            pool_choices = [vector.representation for indx, vector in enumerate(population) if indx != i]

            # Select guest vectors
            choices = random.sample(pool_choices, 3)

            # Create trial
            rep_trial = operator(cr, fx, target.representation, choices[0], choices[1], choices[2])
            trial = Individual(function, rep=rep_trial)

            # ---------------------- Selection
            if maximization:
                if trial.aptitude >= target.aptitude:
                    population[i] = trial
            else:
                if trial.aptitude <= target.aptitude:
                    population[i] = trial
            # If Trial is better: Trial wins
            # If Target is better: Target wins (Population is not changed)
            # Max
            # if trial.aptitude >= target.aptitude:
            #     # print('Trial wins')
            #     population[i] = trial
            # Min
            # if trial.aptitude <= target.aptitude:
            #     # print('Trial wins')
            #     population[i] = trial

        # Increase evaluation counter
        evals += size

        # Sort new population by feasible and aptitude
        if maximization:
            population.sort(key=lambda x: x.aptitude, reverse=True)
        else:
            population.sort(key=lambda x: x.aptitude)
        # Max
        # population.sort(key=lambda x: x.aptitude, reverse=True)
        # Min
        # population.sort(key=lambda x: x.aptitude)

        # Increase generation counter
        gen += 1

        # ---------------------- Report solutions
        # Print all new results
        # print(f'New Representations:\t{[ind.representation for ind in population]}')
        # print(f'New Aptitudes:\t{[ind.aptitude for ind in population]}')
        # print(f'New Restrictions:\t{[ind.restrictions for ind in population]}')
        # print(f'New Validations:\t{[ind.valid for ind in population]}')

        # Report the best solution in generation.
        print(f'Gen {gen}:\n {population[0].report()}')

        # Save best of Generation
        bests.append(population[0])

    last_best = bests[-1]
    # last_best.show(gen, exec)
    all_apt = [i.aptitude for i in bests]
    # plt.plot(all_snr)
    # plt.xlabel('Generations')
    # plt.ylabel('SNR')
    # plt.savefig(f'Evidence/DE/Exec{exec}/SNR.png')

    # if show_last:
    #     last_best.show(gen, exec, save=False)
    #
    # plt.ion()
    # Return the best of every generation
    # Return the best of every generation
    return bests, gen, all_apt, evals


if __name__ == '__main__':
    for i in range(8, 16):
        test_signal = scio.loadmat(f'Experiments-Syn/Signals/Signal_L_E2_{i}.mat')['signal'].T[0]
        results, _, histApts, _ = DE(Function(signal=test_signal, fs=1000), 50, 200, 0.8, 0.5)
        # # trialSignal += np.random.normal(0, 8, trialSignal.shape)
        # trialSignal = scio.loadmat('syntheticSignal-Noise_T2.mat')['data'][0]
        # # trialSignal = scio.loadmat('SEIZURES/TRAIN/PR/PR_chb01_04.mat')['data'][0]
        # # plt.plot(trialSignal)
        # scale = max(trialSignal)
        # #
        # results, gen, snrs, evaluations = DE(Function(signal=trialSignal[:52], fs=256), 50, 200, 0.8, 0.5)
        # #
        best = results[-1]
        scio.savemat(f'Experiments-Syn/Signals/Deco/E2-Deco-DE-L{i}.mat', {'data': best.decoded})
        print(f'Signal {i}:\n')
        print(best.report())
        print(f'SNR: {best.metrics[0]}')
        print(f'RMSE: {best.metrics[1]}')
        print(f'MAPE: {best.metrics[2]}')
        print(f'SMAPE: {best.metrics[3]}')
        print(f'AFR: {best.metrics[4]}')
        # plt.figure(f'E2_DE_S{i}')
        # plt.plot(best.original, label='Original')
        # plt.plot(best.decoded, '--', label='Decoded')
        # plt.legend()
        # plt.xlabel('Sample time')
        # plt.ylabel('Value signal')

        print(f'FIR:\n{best.filter}')
        scio.savemat(f'Experiments-Syn/Signals/Deco/E2-histConv-DE_L{i}.mat', {'data': histApts})

    # plt.subplot(2, 1, 2)
    # plt.plot(encoded, '.-')
    # plt.title("Train Signal")

    # #
    # # # Segment Results:
    # segmentSNR = bestSpiker.snr(trialSignal[:52], best.decoded, fs=256)
    # segmentRMSE = bestSpiker.rmse(trialSignal[:52], best.decoded)
    # segmentMAPE = bestSpiker.mape(trialSignal[:52], best.decoded)
    # segmentAFR = bestSpiker.afr(best.encoded)
    # #
    # print('Segmented')
    # print(f'SNR: {segmentSNR}')
    # print(f'RMSE: {segmentRMSE}')
    # print(f'MAPE: {segmentMAPE}')
    # print(f'AFR: {segmentAFR}')
    # # #
    # bestSpiker.scale = scale * 2
    # encoded, shift = bestSpiker.encode(trialSignal)
    # decoded = bestSpiker.decode(encoded, shift)
    #
    # # Complete Results:
    # completeSNR = bestSpiker.snr(trialSignal, decoded, fs=256)
    # completeRMSE = bestSpiker.rmse(trialSignal, decoded)
    # completeMAPE = bestSpiker.mape(trialSignal, decoded)
    # completeAFR = bestSpiker.afr(encoded)
    #
    # print('Complete')
    # print(f'SNR: {completeSNR}')
    # print(f'RMSE: {completeRMSE}')
    # print(f'MAPE: {completeMAPE}')
    # print(f'AFR: {completeAFR}')

