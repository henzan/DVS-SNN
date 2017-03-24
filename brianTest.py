#########
# IMPORTS
#########
from brian2 import *
from math import *
import pickle
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

    # encode the information in an understandable structure for Brian
    indexes = []
    times   = []
    for i in xrange(0, len(time)):
        for j in xrange(0, len(time[0])):
            if time[i][j] == 0:
                break
            else:
                indexes.append(i)
                times.append(time[i][j]*10**3)

    if plot == True:
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure(1)
        ax = fig.gca()
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('v')
        for j in xrange(0, numPreNeurons):
            auxSpikes = spikeTrain[j][0:int(counters[j])]
            auxTime   = time[j][0:int(counters[j])]
            ax.scatter(auxTime, auxSpikes)

    return indexes, times

######
# MAIN
######

import os.path
start_scope()

# variables
seconds = 10
threshold = 400

# number of fully-connected neurons
numPreNeurons  = 2000
numPostNeurons = 3

# create or read the input file
filename = str(int(seconds*1000)) + '_' + str(numPreNeurons) + '_' + str(numPostNeurons) + '.dat'
if os.path.isfile('inputs/' + filename) == False:
    indexes, times = poissonSpikeGen(0.001, seconds, numPreNeurons, plot=False)
    with open('inputs/' + filename, "wb") as f:
        pickle.dump([indexes, times], f)
        print 'Input file "%s" created' % filename
else:
    with open('inputs/' + filename, "rb") as f:
        data = pickle.load(f)
        indexes = data[0]
        times   = data[1]
        print 'Input file "%s" used' % filename

# poisson inputs
I = SpikeGeneratorGroup(numPreNeurons, indexes, times*ms)

# first layer of neurons
tau = 10*ms
eqs = '''
dv/dt = (-v)/tau : 1 (unless refractory)
'''
G = NeuronGroup(numPostNeurons, eqs, threshold='v>threshold', reset='v = 0', refractory='5*ms', method='linear')
M = StateMonitor(G, 'v', record=True)
spikemon = SpikeMonitor(G)

# excitatory synapses (input-neurons)
taupre  = 16.8*ms
taupost = 33.7*ms
wmax = 1
Apre = 0.03125
Apost = -0.85*Apre

S = Synapses(I, G,
             '''
             w : 1
             dapre/dt = -apre/taupre : 1 (event-driven)
             dapost/dt = -apost/taupost : 1 (event-driven)
             ''',
             on_pre='''
             v_post += w
             apre += Apre
             w = clip(w+apost, 0, wmax)
             ''',
             on_post='''
             apost += Apost
             w = clip(w+apre, 0, wmax)
             ''', method='linear')

S.connect(j='k for k in range(numPreNeurons)', skip_if_invalid=True)
S.w = 'rand()'

# lateral inhibition
LI = Synapses(G, G,'', on_pre='v_post = 0', method='linear')
LI.connect(condition='i!=j')

# run the simulation
run((seconds / 10**(-3))*ms, report='text')

# plots
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(2)
ax = fig.gca()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('v')
ax.plot(M.t/ms, M.v[0])

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(3)
ax = fig.gca()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('v')
ax.plot(M.t/ms, M.v[1])

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(4)
ax = fig.gca()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('v')
ax.plot(M.t/ms, M.v[2])

plt.show()