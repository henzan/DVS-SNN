#########
# IMPORTS
#########

import paer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###########
# FUNCTIONS
###########

# read the file
def get_data(file):
    aefile = paer.aefile(file)
    aedata = paer.aedata(aefile)
    print 'Points: %i, Time: %0.2f' % (len(aefile.data), (aefile.timestamp[-1]-aefile.timestamp[0])/(10**6))
    return aedata

######
# MAIN
######

# options
save = False
plot = False

base_dir = '/home/fedepare/DVS-SNN/DVS-datasets'
filename = base_dir + '/' + 'mnist_0_scale04_0216.aedat'

# store the data -> class with [x (px),y (px),t (0-1),ts (micro seconds)]
d1 = get_data(filename)

# 3D plot
if plot:

    # divide the vector based on the polarity of the change
    polZero = np.zeros((3,len(d1.x)))
    polOne  = np.zeros((3,len(d1.x)))
    cntZeros = 0
    cntOnes  = 0

    for x in xrange(0, len(d1.x)):
        if d1.t[x] == 0:
            polZero[0][cntZeros] = d1.x[x]
            polZero[1][cntZeros] = d1.y[x]
            polZero[2][cntZeros] = d1.ts[x] / 10**6
            cntZeros += 1
        else:
            polOne[0][cntOnes] = d1.x[x]
            polOne[1][cntOnes] = d1.y[x]
            polOne[2][cntOnes] = d1.ts[x] / 10**6
            cntOnes += 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(polZero[0][0:cntZeros], polZero[1][0:cntZeros], polZero[2][0:cntZeros], c='r', label='P=0')
    ax.scatter(polOne[0][0:cntOnes], polOne[1][0:cntOnes], polOne[2][0:cntOnes], c='b', label='P=1')
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    ax.set_zlabel('z [s]')
    ax.set_xlim([0, 128])
    ax.set_ylim([0, 128])
    ax.set_zlim([0, max(d1.ts)/10**6])
    ax.legend()
    plt.show()

# save the data into a new .aedat file
if save:
    lib = paer.aefile(filename, max_events=1)
    lib.save(d1, 'test.aedat')