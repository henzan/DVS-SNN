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

def poissonSpikeGen(dt, fr, tSim, numPreNeurons):
    nBins    = int(floor(tSim / dt))
    spikeMat = np.zeros((numPreNeurons, nBins))

    if numPreNeurons <= 0:
        print 'poissonSpikeGen: error'
    else:
        spikeMat = np.random.uniform(0, 1,(numPreNeurons, nBins)) < fr*dt

    tVec = np.linspace(0, tSim, tSim/dt)
    return tVec, spikeMat

######
# MAIN
######

# variables
dt = 0.001
fr = 30
tSim = 2

# number of fully-connected neurons
numPreNeurons  = 10
numPostNeurons = 3

# define input spike train
tVec, spikeMat = poissonSpikeGen(dt, fr, tSim, 1)

# weights of all the synapses
preWeights = np.random.uniform(0, 1,(numPreNeurons, numPostNeurons))

# create the first layer of neurons
firstLayer = LIF(numPostNeurons, 0, preWeights)

fede = np.linspace(0, 2, num=2000)
u    = np.linspace(0, 2, num=2000)
for x in xrange(0, len(u)):
    u[x] = firstLayer.membranePotential(fede[x] - 0.75, fede[x] - 0.25, fede[x] - 1)

# initialize figure
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(1)
ax  = fig.gca()
ax.plot(fede, u)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('v')
plt.show()