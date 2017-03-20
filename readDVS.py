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
def get_data(file, min, max):
    aefile = paer.aefile(file, max_events=max+1)
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
filename = base_dir + '/' + 'mnist_0_scale04_0001.aedat'

# store the data -> class with [x (px),y (px),t (?),ts (micro seconds)]
d1 = get_data(filename, 0, 15744)

# 3D plot
if plot:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(d1.x, d1.y, d1.ts, label='parametric curve')
    ax.legend()
    plt.show()

# save the data into a new .aedat file
if save:
    lib = paer.aefile(filename, max_events=1)
    lib.save(d1, 'test.aedat')