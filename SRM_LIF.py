#########
# IMPORTS
#########
import time
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

        self.tauM = 0.01
        self.tauS = 0.0025
        self.K = 2.11654611985
        self.K1 = 2
        self.K2 = 4
        self.alpha = 0.25


    def simulation(self, tSim, dt, timers=False):

        """ Assumption: the presynaptic spikes arrive at the same time at all the postsynaptic neurons"""

        self.potArray = np.zeros((self.numPostNeurons, int(floor(tSim / dt))))
        self.timePlot = np.linspace(0, tSim, int(floor(tSim / dt)))

        counters = np.zeros(self.numPreNeurons)
        flagsSpikes = np.zeros(self.numPostNeurons)
        for t in xrange(0, int(floor(tSim / dt))-1):

            if timers == True:
                print 'New timestep------------------------------------------------'

            # timers
            time1 = time2 = time3 = time4 = time5 = 0

            flags = np.zeros(self.numPreNeurons)
            flagsSpikesInstant = np.zeros(self.numPostNeurons)
            for x in xrange(0, self.numPostNeurons):

                start = time.clock()

                # incoming presynaptic spike
                for j in xrange(0, self.numPreNeurons):
                    if t*dt >= self.timeSpikes[j][int(counters[j])] and counters[j] <= self.counterSpike[j] - 1:
                        flags[j] = 1
                        self.tj3d[j][x][:] = np.roll(self.tj3d[j][x], 1, axis=0)
                        self.tj3d[j][x][0] = self.timeSpikes[j][int(counters[j])]

                stop = time.clock()
                time1 += stop - start
                start = time.clock()

                # Refractory period (no new spike)
                if self.ti[x] < 10**6 and (t*dt - self.ti[x]) >= self.refraction*dt:
                    flagsSpikes[x] = 0

                stop = time.clock()
                time2 += stop - start
                start = time.clock()

                # check if above threshold
                if self.potArray[x][t] > self.threshold and flagsSpikes[x] == 0:
                    self.ti[x] = t*dt
                    self.tk.fill(10**6)
                    self.tk[x] = t*dt
                    flagsSpikes[x] = 1
                    flagsSpikesInstant[x] = 1

                stop = time.clock()
                time3 += stop - start
                start = time.clock()

                # initilize the membrane potential
                self.potArray[x][t] = 0

                # update the potential of this neuron
                if (t*dt - self.ti[x]) < 7*self.tauM:
                    self.potArray[x][t] += self.kernelEta(t*dt - self.ti[x])

                for j in xrange(0, self.numPreNeurons):
                    for w in xrange(0, 100):
                        if self.tj3d[j][x][w] != 0 and (t * dt - self.tj3d[j][x][w]) < 7*self.tauM:
                            self.potArray[x][t] += self.preWeights[j][x] * self.kernelEpsilon(t * dt - self.tj3d[j][x][w])
                        else:
                            break

                if (t*dt - min(self.tk)) < 7 * self.tauM:
                    self.potArray[x][t] += self.kernelMu(t*dt - min(self.tk))

                stop = time.clock()
                time4 += stop - start
                start = time.clock()

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

                stop = time.clock()
                time5 += stop - start

                # update the membrane potential
                self.potArray[x][t+1] = self.potArray[x][t]

            # this spike has already been fired
            for j in xrange(0, self.numPreNeurons):
                if flags[j] == 1:
                    counters[j] += 1

            # timers
            if timers == True:
                print 'Presynaptic spike: %s' % (time1)
                print 'Refractory period: %s' % (time2)
                print 'Check if above the threshold: %s' % (time3)
                print 'Update potential: %s' % (time4)
                print 'STDP: %s' % (time5)

    def kernelEpsilon(self, sj):
        if sj >= 0:
            return self.K * (np.exp(-sj/self.tauM) - np.exp(-sj/self.tauS))
        else:
            return 0


    def kernelEta(self, si):
        if si >= 0:
            return self.threshold * (self.K1 * np.exp(-si/self.tauM) - self.K2 * (np.exp(-si/self.tauM) - np.exp(-si/self.tauS)))
        else:
            return 0


    def kernelMu(self, sk):
        if sk >= 0:
            return - self.alpha * self.threshold * self.kernelEpsilon(sk)
        else:
            return 0