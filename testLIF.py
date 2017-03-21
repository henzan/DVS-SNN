#########
# IMPORTS
#########
from SRM_LIF import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


###########
# FUNCTIONS
###########


######
# MAIN
######

# number of fully-connected presynaptic neurons
numPreNeurons  = 10

# number of fully-connected postsynaptic neurons
numPostNeurons = 3

# weights of all the synapses
preWeights = np.random.uniform(0,1,(numPreNeurons, numPostNeurons))

# create the first layer of neurons
firstLayer = LIF(numPostNeurons, 0, preWeights)

fede = np.linspace(0, 2, num=2000)
u    = np.linspace(0, 2, num=2000)
for x in xrange(0, len(u)):
    u[x] = firstLayer.membranePotential(fede[x] - 0, fede[x] - 0.5, fede[x] - 2)

# initialize figure
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(1)
ax  = fig.gca()
ax.plot(fede, u)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('v')
plt.show()