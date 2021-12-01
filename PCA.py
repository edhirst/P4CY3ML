'''Data PCA'''
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.preprocessing import StandardScaler #...converts each vector entry to its standard score (entry - entry's mean) / entry's std
from sklearn.decomposition import PCA            #...rotates the input vector onto basis which diagonalises the database's covariance matrix (sorted st first entry has largest variance etc)
from sklearn.decomposition import KernelPCA      #...maps the data to a higher-dimensional space, and uses a non-linear metric (of choice) to separate the data, note a kernel-trick is used so as not to require computation of the high-dimensional embedding

#%% #Import data
with open('./Data/CY.txt','r') as f0:
    CY = np.array(literal_eval(f0.read()))

with open('./Data/Random.txt','r') as f1:
    Random = np.array(literal_eval(f1.read()))

with open('./Data/Coprime.txt','r') as f2:
    Coprime = np.array(literal_eval(f2.read()))

with open('./Data/Transverse.txt','r') as f3:
    Transverse = np.array(literal_eval(f3.read()))
del(f0,f1,f2,f3)

#%% #Perform 2d PCA on selected dataset
#Choose PCA options
scale_check = False #...select to prescale the data st all weights take their standardised scores --> better not to do as lose some structure in the vectors
kpca_check = False  #...select to use kernel-PCA with some non-linear kernel ('rbf' gaussian kernel given but can change inline) --> using kernel did not improve structure extraction

#Choose dataset to PCA
PCAData = Transverse    #...choose from: CY, Random, Coprime, Transverse

#PCA on data
if kpca_check: pca = KernelPCA(n_components=5,kernel='rbf')
else:          pca = PCA(n_components=5)
if scale_check:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(PCAData)
    pcad_data = pca.fit_transform(scaled_data)
else: pcad_data = pca.fit_transform(PCAData)

#Output PCA information
covar_matrix = pca.get_covariance()
print('Covariance Matrix: '+str(covar_matrix)+'\n\nEigenvalues: '+str(pca.explained_variance_)+'\nExplained Variance ratio: '+str(pca.explained_variance_ratio_)+' (i.e. normalised eigenvalues)\nEigenvectors: '+str(pca.components_)) #...note components gives rows as eigenvectors

#Plot 2d PCA
plt.figure('2d Data PCA')
#plt.title()
plt.scatter(pcad_data[:,0],pcad_data[:,1],alpha=0.1)
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.tight_layout()
#plt.savefig('PCA_CY.pdf')

