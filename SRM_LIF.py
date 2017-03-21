#########
# IMPORTS
#########
import numpy as np

class LIF():
    """Create a Spiking Neural Network (SNN) using the model Leak Integrate-and-fire Neuron
        Assumptions:
            - One layer of postsynaptic neurons
            - Fully connected neurons (all the inputs are connected to each of the postsynaptic neurons
            - Lateral inhibition in each layer of the network """

    def __init__(self, numNeurons, uRest, preWeights):
        """Create the network"""

        self.numNeurons = numNeurons
        self.u          = uRest
        self.preWeights = preWeights

    def membranePotential(self, si, sj, sk):
        return self.kernelEta(si) + 1 * self.kernelEpsilon(sj) + self.kernelMu(sk)

    def kernelEpsilon(self, sj):

        K = 2.11654611985
        tauM = 0.01
        tauS = 0.0025

        if sj >= 0:
            return K * (np.exp(-sj/tauM) - np.exp(-sj/tauS)) * self.Heaviside(sj)
        else:
            return 0


    def kernelEta(self, si):

        T  = 550
        K1 = 2
        K2 = 4
        tauM = 0.01
        tauS = 0.0025

        if si >= 0:
            return T * (K1 * np.exp(-si/tauM) - K2 * (np.exp(-si/tauM) - np.exp(-si/tauS)))
        else:
            return 0


    def kernelMu(self, sk):

        alpha = 0.25
        T     = 550

        if sk >= 0:
            return - alpha * T * self.kernelEpsilon(sk)
        else:
            return 0


    def Heaviside(self, s):
        if s >= 0:
            return 1
        else:
            return 0