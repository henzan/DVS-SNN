#########
# IMPORTS
#########
import paer
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
    aetime = (aefile.timestamp[-1]-aefile.timestamp[0])
    print 'Points: %i, Time: %0.2f us' % (len(aefile.data), aetime)
    return aedata, aetime

######
# MAIN
######

# start the scope for Brian
start_scope()
defaultclock.dt = 1*us

# INPUTS #################################################################################################

# read the DVS dile
filename = 'mnist_0_scale16_0287.aedat'
data, aetime = get_data('DVS-datasets/' + filename)

# number of input neurons (2*128*128)
DVSpx   = 128
indices = []
spikes  = []
limit   = DVSpx * DVSpx - 1
for x in xrange(0, len(data.y)):

    # P = 0
    if data.t[x] == 0:
        indices.append(((data.y[x] - 1)*DVSpx + data.x[x]) -1)
        spikes.append(data.ts[x])

    # P = 1
    else:
        indices.append(((data.y[x] - 1) * DVSpx + data.x[x]) - 1 + limit)
        spikes.append(data.ts[x])

# correct the data file for possible errors (repetitions)
indices2 = []
spikes2  = []
cnt = 0
for x in xrange(0, len(spikes)-1):
    if abs(spikes[x] - spikes[x+1]) == 0 and indices[x] == indices[x+1]:
        cnt += 1
    else:
        indices2.append(indices[x])
        spikes2.append(spikes[x])
print "Errors:", cnt

lastSpike = max(spikes2)
spikes3  = []
for x in xrange(0, len(spikes2)):
    spikes3.append(spikes2[x] + lastSpike + 50)

# create Brian group for the inputs
I = SpikeGeneratorGroup(2*128*128, indices2+indices2, (spikes2 + spikes3)*us)
Minput = SpikeMonitor(I)

# FIRST LAYER #############################################################################################
'''Each region of 4x4 pixels in the input layer is connected to 8 neurons
   representing the directions of the motion. Because of the bipolarity of the DVS,
   we need 8*2 neurons for each 4x4 pixel region.
   128*128        = 16384
   128*128/16     = 1024
   128*128/16*8   = 8192
   128*128/16*8*2 = 16384
   The sensor data is divided in regions of 16 pixels, with eight directions in each 4x4 region,
   times 2 polarities of the sensor. In total, 16384 neurons in the first layer.'''

sizeDVSpx  = DVSpx**2
sidediv    = 4
divDVSpx   = sidediv**2
directions = 8
nG1        = sizeDVSpx / divDVSpx * directions * 2

# first layer of neurons
tau = 10*ms
eqs = '''
dv/dt = (-v)/tau : 1 (unless refractory)
'''
G1 = NeuronGroup(nG1, eqs, threshold='v>5', reset='v = 0', refractory='5*ms', method='linear')
spikemon = SpikeMonitor(G1)
M = StateMonitor(G1, 'v', record=12400)

# synapses between the input and first layers (direction not included)
cntRow = 0
cntCol = 0
connectIG1 = [0 for k in xrange(0, 2*sizeDVSpx)]
for k in xrange(0, 2*sizeDVSpx):

    # update the connection array
    xIdx  = cntCol / sidediv + 1
    yIdx  = cntRow / sidediv + 1
    G1Idx = ((yIdx - 1) * DVSpx/4 + xIdx) - 1
    connectIG1[k] = G1Idx

    # update counters
    if cntCol == DVSpx-1:
        cntCol  = 0
        cntRow += 1
    else:
        cntCol += 1

# synapses with 8 directions
connectIG1dir = []
connectIG1inp = []
for width in xrange(0, len(connectIG1)):
    for height in xrange(0, directions):
        connectIG1dir.append(int(connectIG1[width] * directions + height))
        connectIG1inp.append(width)
npconnectIG1dir = np.asarray(connectIG1dir)
npconnectIG1inp = np.asarray(connectIG1inp)

# synapses (input-neurons)
taupre  = 16.8*ms
taupost = 33.7*ms
wmax = 1
Apre = 0.03125
Apost = -0.85*Apre

S_IG1 = Synapses(I, G1,
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

S_IG1.connect(i = npconnectIG1inp, j = npconnectIG1dir)
S_IG1.w = 'rand()'

# lateral inhibition between the set of neurons that refer to the same input region
LIinx = []
LIconnections = []
for neuron in xrange(0, nG1):
    for connection in xrange(neuron-neuron%directions, neuron+(directions-neuron%directions)):
        if neuron != connection:
            LIinx.append(neuron)
            LIconnections.append(connection)

LI_G1 = Synapses(G1, G1,'', on_pre='v_post = 0', method='linear')
LI_G1.connect(i=LIinx, j=LIconnections)

# run the simulation
run((max(spikes3))*us, report='text')

# plot
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(1)
ax = fig.gca()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
ax.scatter(Minput.t/ms, Minput.i)

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(2)
ax = fig.gca()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
ax.scatter(spikemon.t/ms, spikemon.i)
plt.show()