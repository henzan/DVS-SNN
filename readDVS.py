#########
# IMPORTS
#########

import paer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

###########
# FUNCTIONS
###########

# Helper function to read a file. Given (min,max) which are data ranges for extraction, this will return a cropped and
#  suitably sparse output.
def get_data(file, min, max):
    aefile = paer.aefile(file, max_events=max+1)
    aedata = paer.aedata(aefile)
    print 'Points: %i, Time: %0.2f' % (len(aefile.data), (aefile.timestamp[-1]-aefile.timestamp[0])/(10**6))

    return aedata

######
# MAIN
######

base_dir = '/home/fedepare/DVS-SNN/DVS-datasets'
filename = base_dir + '/' + 'mnist_0_scale04_0001.aedat'

# Loop through all files - indexes are extrapolated.
d1 = get_data(filename, 0, 15744)

# Need to pre-load a file, to get the correct headers when writing!
lib = paer.aefile(filename, max_events=1)
lib.save(d1, 'test.aedat')