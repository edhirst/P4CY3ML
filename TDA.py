'''TDA: Persistent Homology'''
'''Uses GitHub library 'ripser' for computation: https://github.com/scikit-tda/ripser.py.git '''
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from ripser import ripser
from persim import plot_diagrams #...additional library for persistence diagram plotting

#%% #Import data
with open('./Data/CY.txt','r') as f0:
    CY = np.array(literal_eval(f0.read()))
    
#Import Hodge numbers
with open('./Data/Hodge.txt','r') as f_hodge:
    Hodge = np.array(literal_eval(f_hodge.read()))
    
del(f0,f_hodge)

#%% #Compute persistent homology
persistence_diagrams = ripser(CY,maxdim=1)['dgms'] #...maxdim p => compute cohomology for classes up to H_p

#%% #Plot persistence diagram
fig = plt.figure()
axes = plt.axes()
plot_diagrams(persistence_diagrams, lifetime=False, show=True, ax=axes) 
fig.tight_layout()
#fig.savefig('./CYPersistenceLifetime.pdf')
