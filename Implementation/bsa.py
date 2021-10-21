import numpy as np
from scipy.signal import firwin
import copy


def get_real(sample, original, signal_scale, frequency):
    """
    Returns the real value of given BSA parameters.

    :param np.ndarray sample: BSA parameters in the form of [Size, Cutoff, Threshold]
    :param np.ndarray original: Original signal.
    :param float signal_scale: Signal scale.
    :param float frequency: Sampling frequency.
    :return: SNR value [Float]
    """
    spiker = BSA(int(sample[0]), sample[1], sample[2], signal_scale, frequency)

    encoded = spiker.encode(original)
    decoded = spiker.decode(encoded)

    snr = spiker.SNR(original, decoded)

    return snr


class BSA:
    def __init__(self, filterSize, cutFreq, threshold, scale, fs=256.0):
        """
        BSA algorithm implementation.
        :param int filterSize: Filter size (Must be odd).
        :param float cutFreq: Cutoff frequency
        :param float threshold: Threshold.
        :param float scale: Scale factor to Fir filter.
        :param float fs: Sampling frequency.
        """

        # Simple variables
        self.sizeFilter = filterSize
        self.cutoff = cutFreq
        self.threshold = threshold
        self.scale = scale
        self.fs = fs

        # Construct FIR filter
        self.FIR = firwin(self.sizeFilter, self.cutoff, fs=self.fs) * self.scale
        self.lenFIR = self.FIR.size

    def encode(self, inputSignal):
        """
        BSA encoding.
        :param np.ndarray inputSignal: Input signal to be encoded.
        :return: Spike train signal
        """
        inputSignal_copy = copy.deepcopy(inputSignal)
        lenSignal = inputSignal_copy.size
        inputSignal_copy -= min(inputSignal_copy)
        outputSignal = np.zeros(lenSignal)

        for t in range(lenSignal - self.lenFIR):
            error1 = 0
            error2 = 0
            for k in range(self.lenFIR):
                error1 += np.abs(inputSignal_copy[t + k + 1] - self.FIR[k])
                error2 += np.abs(inputSignal_copy[t + k])
            if error1 <= (error2 * self.threshold):
                outputSignal[t] = 1
                for k in range(self.lenFIR):
                    inputSignal_copy[t + k] -= self.FIR[k]

        return outputSignal

    def decode(self, inputSignal):
        """
        BSA decoding.
        :param np.ndarray inputSignal: Spike train signal to be decoded.
        :return: Reconstructed signal.
        """

        signalConvoluted = np.convolve(inputSignal, self.FIR)
        outputSignal = signalConvoluted[:(len(signalConvoluted) - self.lenFIR + 1)]

        return outputSignal

    @classmethod
    def SNR(cls, originalSignal, decodedSignal):
        """
        SNR calculation
        :param np.ndarray originalSignal: Original Signal
        :param np.ndarray decodedSignal: Reconstructed Signal
        :return: SNR value [Float]
        """
        powS = np.mean(np.power(originalSignal, 2))
        powN = np.mean(np.power(decodedSignal - originalSignal, 2))
        snr = 10 * np.log10(powS / powN)

        return snr


if __name__ == '__main__':
    import scipy.io as scio
    import matplotlib.pyplot as plt

    siz = 61  # 41, 57
    cut = 20.00003094  # 52, 42,
    thr = 0.94415867  # 0.96, 0.94

    ori_signal = scio.loadmat('../Signals/Tests/Signal_S_E2_2.mat')['signal'].T[0]
    shift = ori_signal.min()
    ori_signal = ori_signal - shift
    scale_ = ori_signal.max() * 2
    fs = 1000

    val_snr = get_real(np.array([siz, cut, thr]), ori_signal, scale_, fs)

    spiker = BSA(siz, cut, thr, scale_, fs)

    encoded = spiker.encode(ori_signal)
    decoded = spiker.decode(encoded)

    snr = spiker.SNR(ori_signal, decoded)


