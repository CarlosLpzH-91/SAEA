import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from Implementation.lhs import LHS
from Implementation.bsa import BSA

fs = 1000
signal = scio.loadmat('../Signals/Tests/Signal_S_E2_1.mat')['signal'].T[0]
shift = signal.min()
signal = signal - shift
scale = max(signal) * 2

sizRange =