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
def get_data(file, max=10**60):
    aefile = paer.aefile(file, max_events=max+1)
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


# INPUTS ########################################################################################
'''Assumptions: - Polarity not taken into account'''

# do we consider the polarity of the data?
polarity = False

# read the DVS dile
filename = 'mnist_0_scale16_0001.aedat'
data, aetime = get_data('DVS-datasets/' + filename)

# number of input neurons (2*128*128)
DVSlen  = 128
DVSsize = DVSlen**2
indices = []
spikes  = []
limit   = DVSsize - 1
minSpikes = min(data.ts)
if polarity:
    for x in xrange(0, len(data.y)):
        if data.t[x] == 0: # P = 0
            indices.append(int(((data.y[x] - 1) * DVSlen + data.x[x]) - 1))
            spikes.append(int(data.ts[x] - minSpikes + 1000))

        else: # P = 1
            indices.append(int(((data.y[x] - 1) * DVSlen + data.x[x]) - 1 + limit))
            spikes.append(int(data.ts[x] - minSpikes + 1000))

else:
    for x in xrange(0, len(data.y)):
        indices.append(int(((data.y[x] - 1) * DVSlen + data.x[x]) - 1))
        spikes.append(int(data.ts[x] - minSpikes + 1000))


# correct the data file for possible errors (repetitions)
indicesSameTime = []
indices2 = []
spikes2  = []
cntError = 0
timeDt   = spikes[0]
for x in xrange(0, len(spikes)-1):

    # extend the vector with the neurons firing in this dt
    if spikes[x] == timeDt:
        indicesSameTime.append(indices[x])

    # check for repetitions in the current dt
    if spikes[x+1] != timeDt:

        # vector of unique neurons
        uniqueIndices = []
        for z in xrange(0, len(indicesSameTime)):
            flag = 0
            for zz in xrange(0, len(uniqueIndices)):
                if indicesSameTime[z] == uniqueIndices[zz]:
                    flag = 1
                    break
            if flag == 0:
                uniqueIndices.append(indicesSameTime[z])

        indices2.extend(uniqueIndices)
        spikes2.extend([timeDt for z in xrange(0, len(uniqueIndices))])
        if len(uniqueIndices) != len(indicesSameTime):
            cntError +=1
        indicesSameTime = []
        timeDt = spikes[x+1]
print "Errors:", cntError

# create Brian group for the inputs
if polarity: I = SpikeGeneratorGroup(2*DVSsize, indices2, spikes2*us)
else:        I = SpikeGeneratorGroup(DVSsize, indices2, spikes2*us)

# monitor the input spike train
Minput = SpikeMonitor(I)


# FIRST CONVOLUTIONAL LAYER #####################################################################
'''Receptive fields: 4x4 with overlap 1
   Neuronal maps: 8'''

RFc1len  = 4
RFc1size = RFc1len**2
nMapsc1  = 8
overlap  = 1
nC1      = DVSsize / RFc1size * nMapsc1

# first layer of neurons
tau = 10*ms
eqs = '''
dv/dt = (-v)/tau : 1 (unless refractory)
'''
C1 = NeuronGroup(nC1, eqs, threshold='v>5', reset='v = 0', refractory='5*ms', method='linear')
spikemon = SpikeMonitor(C1)
M = StateMonitor(C1, 'v', record=False)

# synapses between the input and first convolutional (neural maps not included)
idxRF  = 0
connectIC1     = np.zeros((RFc1size * nMapsc1, DVSsize))
cntConnections = np.zeros(DVSsize)
connectIC1.fill(-1)

for nIdx in xrange(0, DVSsize):

    # get the location on the image
    DVSrow = nIdx / DVSlen
    DVScol = nIdx - DVSrow * DVSlen

    # check if we can fit a new RF at this location
    if DVScol + RFc1len <= DVSlen and DVSrow + RFc1len <= DVSlen:

        # check the indices included in this RF
        for rRF in xrange(0, RFc1len):
            for cRF in xrange(0, RFc1len):
                auxIdx = (DVSrow + rRF) * DVSlen + (DVScol + cRF)
                connectIC1[cntConnections[auxIdx]][auxIdx] = idxRF
                cntConnections[auxIdx] += 1

        # update the RF counter
        idxRF += 1

# maximum number of RF
idxRFmax = idxRF + 1

print connectIC1[:, 0]

# expand the network to other maps
for nIdx in xrange(0, DVSsize):
    for mIdx in xrange(1, nMapsc1):
        for l in xrange(0, int(cntConnections[nIdx])):
            connectIC1[cntConnections[nIdx]][nIdx] = idxRFmax * mIdx + connectIC1[l][nIdx]
            cntConnections[nIdx] += 1
            # print connectIC1[:, 0]

print connectIC1[:,130]

# RUN THE SIMULATION & PLOTS ####################################################################
# run((max(spikes2) + 1000)*us, report='text')