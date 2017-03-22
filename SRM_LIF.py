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
        self.threshold      = 550
        self.tj3d           = np.zeros((numPreNeurons, numPostNeurons, 100))
        self.ti             = np.zeros(numPostNeurons)
        self.tk             = np.zeros(numPostNeurons)
        self.tj3d.fill(10**6)
        self.ti.fill(10**6)
        self.tk.fill(10**6)

        self.refraction = 5

    def simulation(self, tSim, dt):

        """ Assumption: the presynaptic spikes arrive at the same time at all the postsynaptic neurons"""

        tauM = 0.01

        self.potArray = np.zeros((self.numPostNeurons, int(floor(tSim / dt))))
        self.timePlot = np.zeros(int(floor(tSim / dt)))

        counters = np.zeros(self.numPreNeurons)
        flagsSpikes = np.zeros(self.numPostNeurons)
        for t in xrange(0, int(floor(tSim / dt))-1):

            # store the time vector
            self.timePlot[t] = t*dt

            flags = np.zeros(self.numPreNeurons)
            flagsSpikesInstant = np.zeros(self.numPostNeurons)
            for x in xrange(0, self.numPostNeurons):

                # incoming presynaptic spike
                for j in xrange(0, self.numPreNeurons):
                    if t*dt >= self.timeSpikes[j][int(counters[j])] and counters[j] <= self.counterSpike[j] - 1:
                        flags[j] = 1
                        self.tj3d[j][x][:] = np.roll(self.tj3d[j][x], 1, axis=0)
                        self.tj3d[j][x][0] = self.timeSpikes[j][int(counters[j])]

                # Refractory period (no new spike)
                if self.ti[x] < 10**6 and (t*dt - self.ti[x]) >= self.refraction*dt:
                    flagsSpikes[x] = 0

                # check if above threshold
                if self.potArray[x][t] > self.threshold and flagsSpikes[x] == 0:
                    self.ti[x] = t*dt
                    self.tk.fill(10**6)
                    self.tk[x] = t*dt
                    flagsSpikes[x] = 1
                    flagsSpikesInstant[x] = 1

                # initilize the membrane potential
                self.potArray[x][t] = 0

                # update the potential of this neuron
                if (t*dt - self.ti[x]) < 7*tauM:
                    self.potArray[x][t] += self.kernelEta(t*dt - self.ti[x])

                for j in xrange(0, self.numPreNeurons):
                    for w in xrange(0, 100):
                        if self.tj3d[j][x][w] != 0:
                            if (t * dt - self.tj3d[j][x][w]) < 7*tauM:
                                self.potArray[x][t] += self.preWeights[j][x] * self.kernelEpsilon(t * dt - self.tj3d[j][x][w])
                        else:
                            break

                if (t*dt - min(self.tk)) < 7 * tauM:
                    self.potArray[x][t] += self.kernelMu(t*dt - min(self.tk))

                # spike-time-dependent plasticity (STDP)
                if flagsSpikesInstant[x] == 1:
                    tauPlus  = 16.8*dt
                    tauMinus = 33.7*dt
                    aPlus    = 0.03125
                    aMinus   = 0.85*aPlus
                    for j in xrange(0, self.numPreNeurons):
                        if self.tj3d[j][x][-1] <= self.ti[x]:
                            self.preWeights[j][x] += aPlus * np.exp((self.tj3d[j][x][-1] - self.ti[x])/tauPlus)
                        else:
                            self.preWeights[j][x] += - aMinus * np.exp(-(self.tj3d[j][x][-1] - self.ti[x])/tauMinus)

                        # bound the synaptic weights
                        if self.preWeights[j][x] > 1:
                            self.preWeights[j][x] = 1
                        elif self.preWeights[j][x] < -1:
                            self.preWeights[j][x] = -1

                # update the membrane potential
                self.potArray[x][t+1] = self.potArray[x][t]
                self.timePlot[t+1] = (t+1) * dt

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

        K1 = 2
        K2 = 4
        tauM = 0.01
        tauS = 0.0025

        if si >= 0:
            return self.threshold * (K1 * np.exp(-si/tauM) - K2 * (np.exp(-si/tauM) - np.exp(-si/tauS)))
        else:
            return 0


    def kernelMu(self, sk):

        alpha = 0.25
        T     = 550

        if sk >= 0:
            return - alpha * T * self.kernelEpsilon(sk)
        else:
            return 0