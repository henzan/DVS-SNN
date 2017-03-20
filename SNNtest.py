#########
# IMPORTS
#########
from brian2 import *
import matplotlib as mpl
import matplotlib.pyplot as plt


###########
# FUNCTIONS
###########


######
# MAIN
######

start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''

G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', refractory=5*ms, method='linear')

statemon = StateMonitor(G, 'v', record=0)
spikemon = SpikeMonitor(G)

run(50*ms)

# initialize figure
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(1)
ax  = fig.gca()
ax.plot(statemon.t/ms, statemon.v[0])
ax.grid(grid)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('v')
plt.show()