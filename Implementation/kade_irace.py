import argparse
import logging
import sys
import numpy as np
import scipy.io as scio
from kade import KADE
from lhs import LHS

# Signal
original = scio.loadmat(
    'C:/Users/Alberto/Documents/MIA/Periodos/3_1-Semestral-Ago-Ene/ComputacionEvolutiva2/SAEA/Implementation/IRACE/Signal/Signal_S_E2_2.mat')[
    'signal'].T[0]
shift = original.min()
original = original - shift
scale = original.max() * 2

gen = 200  # Default
evals = 2500
freq = 1000
ranges = [[16, 80],
          [20, 80],
          [0.8, 1.1]]

# KADE 5%
update = 1
prc_select = 0.05
rbf_ls = np.ones(3)
rbf_ls_bounds = [(1e-2, 1e6), (1e-2, 1e8), (1e-5, 1e2)]
# n_rest = 50
# a = 1e-3
pop = 50


def main(cr, fx, a, n_rest, datafile):
    init_samples = LHS(numPoints=pop,
                       rangeSize=ranges[0],
                       rangeCut=ranges[1],
                       rangeThreshold=ranges[2])

    _, _, aptitudes, _, _, _, _, _ = KADE(signal=original, scale=scale, fs=freq, ranges=ranges, samples=init_samples,
                                          tot_evals=evals, size=pop, num_gen=gen, cr=cr, fx=fx, n_update=update,
                                          prc_selection=prc_select, select_option='U', stop='Eval', rbf_ls=rbf_ls,
                                          rbf_ls_bounds=rbf_ls_bounds, n_rest=int(n_rest), a=a, re_evaluate_km=False,
                                          plot_vars=False, norm_y=True, verbose=False)
    score = -aptitudes[-1]
    print(score)

    with open(datafile, 'w') as f:
        f.write(str(score))


if __name__ == '__main__':
    # Checking if args are ok
    with open('args.txt', 'w') as file:
        file.write(str(sys.argv))

    # Loading example arguments
    arg_parse = argparse.ArgumentParser(description='Feature Selection using DE')
    arg_parse.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true')
    # 3 args to test
    # arg_parse.add_argument('--pop', dest='pop', type=int, required=True, help='Population size')
    arg_parse.add_argument('--cr', dest='cr', type=float, required=True, help='CR value')
    arg_parse.add_argument('--fx', dest='fx', type=float, required=True, help='FX value')
    arg_parse.add_argument('--a', dest='a', type=float, required=True, help='Alpha')
    arg_parse.add_argument('--n_rest', dest='n_rest', type=float, required=True, help='#N Restarts')
    # 1 arg file name to save and load fo value
    arg_parse.add_argument('--datafile', dest='datafile', type=str, required=True, help='File to print results')

    args = arg_parse.parse_args()
    logging.debug(args)
    # Call main function passing args
    main(args.cr, args.fx, args.a, args.n_rest, args.datafile)
