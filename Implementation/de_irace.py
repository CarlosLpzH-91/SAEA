import argparse
import logging
import sys
import scipy.io as scio
from de import DE
from lhs import LHS


def main(pop, cr, fx, datafile):
    gen = 200  # Defaul
    evals = 8000
    freq = 1000
    ranges = [[16, 80],
              [20, 80],
              [0.8, 1.1]]

    # Signal
    original = scio.loadmat('C:/Users/Alberto/Documents/MIA/Periodos/3_1-Semestral-Ago-Ene/ComputacionEvolutiva2/SAEA/Implementation/IRACE/Signal/Signal_S_E2_2.mat')['signal'].T[0]
    shift = original.min()
    original = original - shift
    scale = original.max() * 2

    init_samples = LHS(numPoints=pop,
                       rangeSize=ranges[0],
                       rangeCut=ranges[1],
                       rangeThreshold=ranges[2])

    _, aptitudes, _ = DE(init_samples, gen, pop, cr, fx, ranges, original, scale, freq, evals, 'Eval')
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
    # 4 args to test
    arg_parse.add_argument('--pop', dest='pop', type=int, required=True, help='Population size')
    arg_parse.add_argument('--cr', dest='cr', type=float, required=True, help='CR value')
    arg_parse.add_argument('--fx', dest='fx', type=float, required=True, help='FX value')
    # 1 arg file name to save and load fo value
    arg_parse.add_argument('--datafile', dest='datafile', type=str, required=True, help='File to print results')

    args = arg_parse.parse_args()
    logging.debug(args)
    # Call main function passing args
    main(args.pop, args.cr, args.fx, args.datafile)
