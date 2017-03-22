#########
# IMPORTS
#########
from SRM_LIF import *
from math import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


###########
# FUNCTIONS
###########

def poissonSpikeGen(dt, tSim, numPreNeurons, plot=False):

    nBins = int(floor(tSim / dt))
    spikeMat = np.zeros((numPreNeurons, nBins))

    tVec = np.linspace(0, tSim, nBins)
    r = np.random.uniform(0, 90,(numPreNeurons, nBins))
    s = np.random.uniform(-1800, 1800,(numPreNeurons, nBins))

    if numPreNeurons <= 0:
        print 'poissonSpikeGen: error'
    else:
        for t in xrange(0, nBins-1):
            for j in xrange(0, numPreNeurons):
                spikeMat[j][t] = np.random.uniform(0, 1, 1) < r[j][t] * dt
                r[j][t+1] += s[j][t] * dt
                if r[j][t+1] > 90:
                    r[j][t + 1] = 90
                elif r[j][t+1] < 0:
                    r[j][t + 1] = 0
                s[j][t + 1] += np.random.uniform(-360, 360, 1)
                if s[j][t+1] > 1800:
                    s[j][t + 1] = 1800
                elif s[j][t+1] < -1800:
                    s[j][t + 1] = -1800

    # post-processing
    time       = np.zeros((numPreNeurons, nBins))
    spikeTrain = np.zeros((numPreNeurons, nBins))
    counters   = np.zeros(numPreNeurons)

    for t in xrange(0, nBins):
        for j in xrange(0, numPreNeurons):
            if spikeMat[j][t] == True:
                spikeTrain[j][int(counters[j])] = j
                time[j][int(counters[j])] = tVec[t]
                counters[j] += 1

    if plot == True:
        # initialize figure
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure(2)
        ax = fig.gca()
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('v')
        for j in xrange(0, numPreNeurons):
            auxSpikes = spikeTrain[j][0:int(counters[j])]
            auxTime   = time[j][0:int(counters[j])]
            ax.scatter(auxTime, auxSpikes)

    return time, counters

######
# MAIN
######

# variables
dt   = 0.001
tSim = 1

# number of fully-connected neurons
numPreNeurons  = 10
numPostNeurons = 3

# define input spike train
time, counters = poissonSpikeGen(dt, tSim, numPreNeurons, False)

# weights of all the synapses
preWeights = np.random.uniform(0, 1, (numPreNeurons, numPostNeurons))

# create the first layer of neurons
firstLayer = LIF(numPreNeurons, numPostNeurons, preWeights, time, counters)

# execute the simulation
firstLayer.simulation(tSim, dt)

# initialize figure
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(1)
ax  = fig.gca()
ax.set_xlabel('Time (s)')
ax.set_ylabel('v')
ax.plot(firstLayer.timePlot, firstLayer.potArray[0][:])

# initialize figure
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(2)
ax  = fig.gca()
ax.set_xlabel('Time (s)')
ax.set_ylabel('v')
ax.plot(firstLayer.timePlot, firstLayer.potArray[1][:])

plt.show()