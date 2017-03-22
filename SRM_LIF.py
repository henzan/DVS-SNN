#########
# IMPORTS
#########
import numpy as np
from math import *


class LIF():
    """Create a Spiking Neural Network (SNN) using the model Leak Integrate-and-fire Neuron
        Assumptions:
            - One layer of postsynaptic neurons
            - Fully connected neurons (all the inputs are connected to each of the postsynaptic neurons
            - Lateral inhibition in each layer of the network """

    def __init__(self, numPreNeurons, numPostNeurons, preWeights, time, counters):
        """Create the network"""

        self.numPreNeurons  = numPreNeurons
        self.numPostNeurons = numPostNeurons
        self.preWeights     = preWeights
        self.timeSpikes     = time
        self.counterSpike   = counters
        self.potential      = np.zeros(numPostNeurons)
        self.threshold      = 550
        self.tj3d           = np.zeros((numPreNeurons, numPostNeurons, 100))
        self.ti             = np.zeros(numPostNeurons)
        self.tk             = np.zeros(numPostNeurons)
        self.tj3d.fill(10**6)
        self.ti.fill(10**6)
        self.tk.fill(10**6)

    def simulation(self, tSim, dt):

        """ Assumption: the presynaptic spikes arrive at the same time at all the postsynaptic neurons"""

        tauM = 0.01

        self.potArray = np.zeros((self.numPostNeurons, int(floor(tSim / dt))))
        self.timePlot = np.zeros(int(floor(tSim / dt)))

        counters = np.zeros(self.numPreNeurons)
        for t in xrange(0, int(floor(tSim / dt))):

            # store the time vector
            self.timePlot[t] = t*dt

            flags = np.zeros(self.numPreNeurons)
            for x in xrange(0, self.numPostNeurons):

                # incoming presynaptic spike
                for j in xrange(0, self.numPreNeurons):
                    if t*dt >= self.timeSpikes[j][int(counters[j])] and counters[j] <= self.counterSpike[j] - 1:
                        flags[j] = 1
                        self.tj3d[j][x][:] = np.roll(self.tj3d[j][x], 1, axis=0)
                        self.tj3d[j][x][0] = self.timeSpikes[j][int(counters[j])]

                # check if above threshold
                if self.potential[x] > self.threshold:
                    self.ti[x] = t*dt
                    self.tk.fill(10**6)
                    self.tk[x] = t*dt

                # initilize the membrane potential
                self.potential[x] = 0

                # update the potential of this neuron
                self.potential[x] += self.kernelEta(t*dt - self.ti[x])
                for j in xrange(0, self.numPreNeurons):
                    for w in xrange(0, 100):
                        if self.tj3d[j][x][w] != 0:
                            if (t * dt - self.tj3d[j][x][w]) < 7*tauM:
                                self.potential[x] += self.preWeights[j][x] * self.kernelEpsilon(t * dt - self.tj3d[j][x][w])
                        else:
                            break
                self.potential[x] += self.kernelMu(t*dt - min(self.tk))

                # update the matrix
                self.potArray[x][t] = self.potential[x]

            # this spike has already been fired
            for j in xrange(0, self.numPreNeurons):
                if flags[j] == 1:
                    counters[j] += 1


    def kernelEpsilon(self, sj):

        K = 2.11654611985
        tauM = 0.01
        tauS = 0.0025

        if sj >= 0:
            return K * (np.exp(-sj/tauM) - np.exp(-sj/tauS))
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