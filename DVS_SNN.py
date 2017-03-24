#########
# IMPORTS
#########
import paer
import pickle
import warnings
import numpy as np
from math import *
from brian2 import *
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

###########
# FUNCTIONS
###########
def get_data(file):
    aefile = paer.aefile(file)
    aedata = paer.aedata(aefile)
    print 'Points: %i, Time: %0.2f' % (len(aefile.data), (aefile.timestamp[-1]-aefile.timestamp[0])/(10**6))
    return aedata

######
# MAIN
######

# start the scope for Brian
start_scope()

# read the DVS dile
filename = 'mnist_0_scale04_0216.aedat'
data = get_data('DVS-datasets/' + filename)

# number of input neurons (2*128*128)
DVSpx   = 128
indices = []
spikes  = []
limit   = DVSpx * DVSpx - 1
for x in xrange(0, len(data.y)):

    # inhibitory connections
    if data.t[x] == 0:
        indices.append(((data.y[x] - 1)*DVSpx + data.x[x]) -1)
        spikes.append(data.ts[x])

    # excitatory connections
    else:
        indices.append(((data.y[x] - 1) * DVSpx + data.x[x]) - 1 + limit)
        spikes.append(data.ts[x])

# create Brian group for the inputs
I = SpikeGeneratorGroup(2*128*128, indices, spikes*us)

# first layer of neurons
tau = 10*ms
eqs = '''
dv/dt = (-v)/tau : 1 (unless refractory)
'''
G = NeuronGroup(128*128, eqs, threshold='v>100000', reset='v = 0', refractory='5*ms', method='linear')
M = StateMonitor(G, 'v', record=8192)
spikemon = SpikeMonitor(G)

# synapses (input-neurons)
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

S.connect(condition='j==i and j==(limit+i)')
S.w = 'rand()'

# lateral inhibition
LI = Synapses(G, G,'', on_pre='v_post = 0', method='linear')
LI.connect(condition='i!=j')

# run the simulation
run(100*ms, report='text')

# plots
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(2)
ax = fig.gca()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('v')
ax.plot(M.t/ms, M.v[0])

plt.show()