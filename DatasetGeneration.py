'''Data Generation'''
#Import libraries
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from ast import literal_eval
from math import gcd

####################################################################################
'''CY Data'''
#%% #Import CY data
with open('./Data/CY.txt','r') as f0:
    CY = np.array(literal_eval(f0.read()))
del(f0)

#%% #Plot histogram of CY weight entries
CYWeightHistograms = [np.unique(CY[:,i],return_counts=True) for i in range(5)]

plt.figure('Random Weights Frequency Dist.')
for i in range(5):
    plt.plot(CYWeightHistograms[i][0],CYWeightHistograms[i][1]/7555,label=str(i+1))
plt.xlabel('Weight Values')
plt.ylabel('Frequency Density')
#plt.xlim(0,100)
#plt.yscale('log')
plt.legend(loc='upper right')
plt.grid()
#plt.savefig('./CYWeightFreqDists.pdf')

#%% #Histogram over all CY entries - used for fitting exponential distribution
CY_freq_values, CY_freq_counts = np.unique(np.ndarray.flatten(CY),return_counts=True)

#Define and fit exponential probability dist
ExpDist = expon.fit(np.ndarray.flatten(CY)) #...params: (1.0, 49.536307081403045)
pts = np.linspace(0,1750,10000)
exp_pts = expon.pdf(pts, *ExpDist)

#Plot data of all weight with exponential distribution overlaid
plt.figure('WP4 Weights Frequency Dist.')
plt.plot(CY_freq_values,CY_freq_counts/(7555*5),label='CY data')
plt.plot(pts,exp_pts,label='exp')
plt.legend(loc='upper right')
plt.xlabel('Weight Values')
plt.ylabel('Frequency Density')
#plt.yscale('log')
plt.grid()
#plt.savefig('./CYAllWeightswithExpDist.pdf')

####################################################################################
'''Random Integer Data'''
#%% #Generate data of random integers (sorted unique 5-vectors)
Random = []
use_exp = False      #...decide whether to use exponential (true) or uniform (false) distributions to sample generated weights
max_value = 2000    #...for uniform dist, use random integers up to 2000 (to instead use the highest integer occuring in the true dataset (1743) use: max(map(max,OG)) )

while len(Random) < len(CY):
    if use_exp: trial = np.sort(np.round(np.random.exponential(ExpDist[1],5))).astype(np.int32)
    else:       trial = np.sort(np.random.choice(range(1,max_value+1),size=5,replace=True)) #...allow repeated numbers, alike true dataset          
    #Check no weights of '0' generated and 5-vector does not satisfy coprime condition
    if 0 in trial or np.gcd.reduce(trial) == 1: continue
    #Check 5-vector not already been generated
    for vec in Random:         
        if np.array_equal(trial,vec): continue
    Random.append(trial)
Random = np.array(Random)
del(trial,vec)

#%% #Write Random data to a file
with open('./Random.txt','w') as file:
    file.write(str([list(x) for x in Random])) #...convert to lists to avoid 'np.array' in .txt file
del(file) 

#%% #Import Random data
with open('./Data/Random.txt','r') as f1:
    Random = np.array(literal_eval(f1.read()))
del(f1)

#%% #Plot histogram of random integer entries
RandomWeightHistograms = [np.unique(Random[:,i],return_counts=True) for i in range(5)]

plt.figure('Random Weights Frequency Dist.')
for i in range(5):
    plt.plot(RandomWeightHistograms[i][0],RandomWeightHistograms[i][1]/7555,label=str(i+1))
plt.xlabel('Weight Values')
plt.ylabel('Frequency Density')
#plt.yscale('log')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('./RandomWeightFreqDists_pre-exp.pdf')

####################################################################################
'''Random Coprime Data'''
#%% #Generate data of random coprime integers (sorted)
Coprime = []
use_exp = True      #...decide whether to use exponential (true) or uniform (false) distributions to sample generated weights
max_value = 2000    #...for uniform dist, use random integers up to 2000 (to instead use the highest integer occuring in the true dataset (1743) use: max(map(max,OG)) )

while len(Coprime) < len(CY):
    #Generate trial 5-vectors
    if use_exp: trial = np.sort(np.round(np.random.exponential(ExpDist[1],5))).astype(np.int32)
    else:       trial = np.sort(np.random.choice(range(1,max_value+1),size=5,replace=True)) #...allow repeated numbers, alike true dataset
    if 0 in trial or np.gcd.reduce(trial) != 1: continue #...ensure the integers are coprime as a set
    #Check 5-vector does not have transverse property
    checks = np.array([(sum(trial)-trial)/k for k in trial]) #...compute all divisibility checks
    checks_boolean = np.array([[val.is_integer() for val in row] for row in checks]) #...evaluate if array entries are integers
    if np.all(np.array([np.any(check) for check in checks_boolean])): continue #...if transverse condition satisfied then there is a True in every row (i.e. all weights have at least one weight st divisible)
    #Check 5-vector not already been generated
    for vec in Coprime:            
        if np.array_equal(trial,vec): continue
    Coprime.append(trial)
Coprime = np.array(Coprime)
del(trial,vec)

#%% #Write Coprime data to a file
with open('./Coprime.txt','w') as file:
    file.write(str([list(x) for x in Coprime])) #...convert to lists to avoid 'np.array' in .txt file
del(file) 
    
#%% #Import Coprime data
with open('./Data/Coprime.txt','r') as f2:
    Coprime = np.array(literal_eval(f2.read()))
del(f2)

#%% #Plot histogram of coprime integer entries
CoprimeWeightHistograms = [np.unique(Coprime[:,i],return_counts=True) for i in range(5)]

plt.figure('Coprime Weights Frequency Dist.')
for i in range(5):
    plt.plot(CoprimeWeightHistograms[i][0],CoprimeWeightHistograms[i][1]/7555,label=str(i+1))
plt.xlabel('Weight Values')
plt.ylabel('Frequency Density')
#plt.yscale('log')
plt.legend(loc='upper right')
plt.grid()
#lt.savefig('./CoprimeWeightFreqDists.pdf')

####################################################################################
'''Random Transverse Coprime Data'''
#%% #Generate data of random coprime integers that satisfy transverse condition (sorted)
Transverse = []
use_exp = True      #...decide whether to use exponential (true) or uniform (false) distributions to sample generated weights
max_value = 2000    #...for uniform dist, use random integers up to 2000 (to instead use the highest integer occuring in the true dataset (1743) use: max(map(max,OG)) )

while len(Transverse) < len(CY):
    #Generate trial 5-vectors
    if use_exp: trial = np.sort(np.round(np.random.exponential(ExpDist[1],5))).astype(np.int32)
    else:       trial = np.sort(np.random.choice(range(1,max_value+1),size=5,replace=True)) #...allow repeated numbers, alike true dataset
    if 0 in trial or np.gcd.reduce(trial) != 1: continue #...ensure the integers are coprime as a set
    #Check 5-vector satisfies transverse property
    checks = np.array([(sum(trial)-trial)/k for k in trial]) #...compute all divisibility checks
    checks_boolean = np.array([[val.is_integer() for val in row] for row in checks]) #...evaluate if array entries are integers
    if not np.all(np.array([np.any(check) for check in checks_boolean])): continue #...if transverse condition not satisfied, then a row with all False's st no weights satisfy divisibility condition for that weight
    #Check 5-vector not already been generated
    for vec in Transverse:            
        if np.array_equal(trial,vec): continue
    Transverse.append(trial)
Transverse = np.array(Transverse)
del(trial,vec)
###Note: the transverse condition is very rare for uniform sampling, so could not be feasibly generated

#%% #Write Transverse data to a file
with open('./Transverse.txt','w') as file:
    file.write(str([list(x) for x in Transverse])) #...convert to lists to avoid 'np.array' in .txt file
del(file)    

#%% #Import Transverse data
with open('./Data/Transverse.txt','r') as f3:
    Transverse = np.array(literal_eval(f3.read()))
del(f3)

#%% #Plot histogram of transverse integer entries
TransverseWeightHistograms = [np.unique(Transverse[:,i],return_counts=True) for i in range(5)]

plt.figure('Transverse Weights Frequency Dist.')
for i in range(5):
    plt.plot(TransverseWeightHistograms[i][0],TransverseWeightHistograms[i][1]/7555,label=str(i+1))
plt.xlabel('Weight Values')
plt.ylabel('Frequency Density')
#plt.yscale('log')
plt.legend(loc='upper right')
plt.grid()
#plt.savefig('./TransverseWeightFreqDists.pdf')

