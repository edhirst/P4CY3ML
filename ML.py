'''Supervised ML'''
#Import Libraries
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from ast import literal_eval
from pandas import qcut
from copy import deepcopy
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import matthews_corrcoef as MCC

#%% #Import data
with open('./Data/CY.txt','r') as f0:
    CY = np.array(literal_eval(f0.read()))
    
#Import Hodge numbers
with open('./Data/Hodge.txt','r') as f_hodge:
    Hodge = np.array(literal_eval(f_hodge.read()))
    
with open('./Data/Random.txt','r') as f1:
    Random = np.array(literal_eval(f1.read()))

with open('./Data/Coprime.txt','r') as f2:
    Coprime = np.array(literal_eval(f2.read()))

with open('./Data/Transverse.txt','r') as f3:
    Transverse = np.array(literal_eval(f3.read()))
    8765
del(f0,f_hodge,f1,f2,f3)

################################################################################
'''Topological Properties'''
#%% #Set-up investigation data
investigation = 0       #...choose what property to ML from: [h11, h21, [h11,h21], Euler number] with the respective index
k = 5                   #...number of k-fold cross-validations to perform (k = 5 => 80(train) : 20(test) splits approx.)

if   investigation == 0: outputs = Hodge[:,0]
elif investigation == 1: outputs = Hodge[:,1]
elif investigation == 2: outputs = Hodge
elif investigation == 3: outputs = np.array([-2*(h[0]-h[1]) for h in Hodge]) 

#Zip input and output data together
data_size = len(CY)
ML_data = [[CY[index],outputs[index]] for index in range(data_size)]

#Shuffle data ordering
np.random.shuffle(ML_data)
s = int(floor(data_size/k)) #...number of datapoints in each validation split

#Define data lists, each with k sublists witht he relevant data for that cross-validation run
Train_inputs, Train_outputs, Test_inputs, Test_outputs = [], [], [], []
for i in range(k):
    Train_inputs.append([datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
    Train_outputs.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
    Test_inputs.append([datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
    Test_outputs.append([datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])

del(ML_data,outputs,i) #...zipped list no longer needed

#%% #Run NN train & test
#Define measure lists
MSEs, MAPEs, Rsqs = [], [], []    #...lists of regression measures
seed = 1                          #...select a random seeding (any integer) for regressor initialisation

#Loop through each cross-validation run
for i in range(k):
    #Define & Train NN Regressor directly on the data
    nn_reg = MLPRegressor((32,64,32),activation='relu',solver='adam',random_state=seed)  #...can edit the NN structure here
    nn_reg.fit(Train_inputs[i], Train_outputs[i]) 

    #Compute NN predictions on test data, and calculate learning measures
    Test_pred = nn_reg.predict(Test_inputs[i])
    Rsqs.append(nn_reg.score(Test_inputs[i],Test_outputs[i]))
    MSEs.append(MSE(Test_outputs[i],Test_pred,squared=True))             #...True -> mse, False -> root mse
    if investigation != 3: MAPEs.append(MAPE(Test_outputs[i],Test_pred)) #...note not defined for learning euler, as true value can be 0

#Output averaged learning measures with standard errors
print('####################################')
print('Average Measures (investigation '+str(investigation)+'):')
print('R^2: ',sum(Rsqs)/k,'\pm',np.std(Rsqs)/np.sqrt(k))
print('MSE: ',sum(MSEs)/k,'\pm',np.std(MSEs)/np.sqrt(k))
if investigation != 3: print('MAPE:',sum(MAPEs)/k,'\pm',np.std(MAPEs)/np.sqrt(k))

################################################################################
'''Calabi-Yau Property'''
#%% #Set-up investigation data
nonCY_data = 2          #...choose which dataset to binary classify the CY data against from: [Random, Coprime, Transverse] with the respective index, or all the datasets with '-1'
k = 5                   #...number of k-fold cross-validations to perform (k = 5 => 80(train) : 20(test) splits approx.)

#Zip input and selected output data together
data_size = len(CY)
if   nonCY_data == 0:  ML_data = [[CY[index],0] for index in range(data_size)]+[[Random[index],1] for index in range(data_size)]
elif nonCY_data == 1:  ML_data = [[CY[index],0] for index in range(data_size)]+[[Coprime[index],1] for index in range(data_size)]
elif nonCY_data == 2:  ML_data = [[CY[index],0] for index in range(data_size)]+[[Transverse[index],1] for index in range(data_size)]
elif nonCY_data == -1: ML_data = [[CY[index],0] for index in range(data_size)]+[[Random[index],1] for index in range(data_size)]+[[Coprime[index],2] for index in range(data_size)]+[[Transverse[index],3] for index in range(data_size)]

#Shuffle data ordering
np.random.shuffle(ML_data)
s = int(floor(data_size/k)) #...number of datapoints in each validation split

#Define data lists, each with k sublists with the relevant data for that cross-validation run
Train_inputs, Train_outputs, Test_inputs, Test_outputs = [], [], [], []
for i in range(k):
    Train_inputs.append([datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
    Train_outputs.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
    Test_inputs.append([datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
    Test_outputs.append([datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])

del(ML_data,i) #...zipped list no longer needed

#%% #Run Classifier train & test
ml_architecture = 1  #...choose architecture from: [Logistic Regressor, Support Vector Machine, Neural Network Classifier] with respective index
Accs, MCCs = [], []  #...lists of classifier measures
seed = 1             #...select a random seeding (any integer) for classifier initialisation

for i in range(k):
    #Reinitialise the selected classifier architecture
    if   ml_architecture == 0: clf = LR(tol=0.1,C=100,solver='newton-cg',random_state=seed)
    elif ml_architecture == 1: clf = SVC(C=1,kernel='linear',random_state=seed)
    elif ml_architecture == 2: clf = MLPClassifier((32,64,32),activation='relu',solver='adam',random_state=seed)
    clf.fit(Train_inputs[i], Train_outputs[i]) 

    #Compute classifier predictions on test data, and calculate learning measures
    Test_pred = clf.predict(Test_inputs[i])
    print('CM_'+str(i+1)+': ',[list(row) for row in CM(Test_outputs[i],Test_pred,normalize='all')]) #...rows -> true class, columns -> predicted class
    Accs.append(Acc(Test_outputs[i],Test_pred,normalize=True))
    MCCs.append(MCC(Test_outputs[i],Test_pred))

#Output averaged learning measures with standard errors
print('\n####################################')
print('Average Measures (data '+str(nonCY_data)+', with architecture '+str(ml_architecture)+'):')
print('Accuracy: ',sum(Accs)/k,'\pm',np.std(Accs)/np.sqrt(k))
print('MCC:      ',sum(MCCs)/k,'\pm',np.std(MCCs)/np.sqrt(k))

################################################################################
'''Misclassification Analysis'''
#%% #Predict trained architecture on full CY dataset and identify misclassifications
ml_architecture = 0  #...choose architecture from: [Logistic Regressor, Support Vector Machine, Neural Network Classifier] with respective index
nonCY_data = 0       #...choose which dataset to binary classify the CY data against from: [Random, Coprime, Transverse] with the respective index, or all the datasets with '-1'
train_size = 50      #...specify the number of each dataset in the training dataset
misclassifications_CYs,classifications_CYs = [],[] #...lists to store classification results
seed = 1             #...select a random seeding (any integer) for classifier initialisation

#Set-up train data
if   nonCY_data == 0:  Train = np.append(CY[np.random.choice(7555,train_size,False),:],Random[np.random.choice(7555,train_size,False),:],axis=0)
elif nonCY_data == 1:  Train = np.append(CY[np.random.choice(7555,train_size,False),:],Coprime[np.random.choice(7555,train_size,False),:],axis=0)
elif nonCY_data == 2:  Train = np.append(CY[np.random.choice(7555,train_size,False),:],Transverse[np.random.choice(7555,train_size,False),:],axis=0)
Train_labels = np.append(np.zeros(train_size),np.ones(train_size),axis=0)
#Shuffle training data
shuffle_order = np.random.choice(2*train_size,2*train_size,False)
Train = Train[shuffle_order,:]
Train_labels = Train_labels[shuffle_order]

#Initialise the selected classifier architecture
if   ml_architecture == 0: clf = LR(tol=0.1,C=100,solver='newton-cg',random_state=seed)
elif ml_architecture == 1: clf = SVC(C=1,kernel='linear',random_state=seed)
elif ml_architecture == 2: clf = MLPClassifier((32,64,32),activation='relu',solver='adam',random_state=seed)
clf.fit(Train,Train_labels)

#Predict on the full CY dataset
predictions_CYs = clf.predict(CY)
for i in range(len(predictions_CYs)):
    if predictions_CYs[i] == 1:     #...predicted non-CY
        misclassifications_CYs.append([list(CY[i]),Hodge[i]]) 
    elif predictions_CYs[i] == 0:   #...predicted CY
        classifications_CYs.append([list(CY[i]),Hodge[i]]) 
print('Total number of misclassifications: '+str(len(misclassifications_CYs))+'\n(Total number of correct classifications: '+str(len(classifications_CYs))+')')
print('Fraction of misclassified data in training data: '+str(len([x for x in misclassifications_CYs if (np.array(x[0])==Train).all(1).any()])/7555))
    
#%% #Plot misclassifed CYs hodge numbers
misclassified_hodges = [i[1] for i in misclassifications_CYs]
classified_hodges = [i[1] for i in classifications_CYs]

plt.figure('Classified CY Hodge Numbers')
plt.scatter([h11[0] for h11 in classified_hodges],[h21[1] for h21 in classified_hodges],label='classified',alpha=0.1,c='#ff7f0e')
plt.scatter([h11[0] for h11 in misclassified_hodges],[h21[1] for h21 in misclassified_hodges],label='misclassified',alpha=0.1,c='#1f77b4')
#plt.scatter([h11[0] for h11 in classified_hodges],[h21[1] for h21 in classified_hodges],label='classified',alpha=0.1,c='#ff7f0e')
plt.xlabel(r'$h^{1,1}$')
plt.ylabel(r'$h^{2,1}$')
plt.axes().set_aspect('equal') #...depreciated, find another way?
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('./LRRandom_Misclass().pdf')

################################################################################
'''Calabi-Yau Property with Hodge binning'''
#%% #Predict CY vs non-CY using LR trained on data partitioned by hodge #s (note no cross-validation here)
nonCY_data = Transverse     #...choose which dataset to binary classify the CY data against from: [Random, Coprime, Transverse]
h21_check = 0           #...set to 0 to use h11, set to 1 to use h21
n = 20                  #...number of runs to average over
number_bins = 50        #...set the number of bins

#Set-up output lists & partitioning bins
Weights, Accuracies = [], []
max_hodges = [max([x[0] for x in Hodge]),max([x[1] for x in Hodge])] #...find max (h11,h21) = (491,491))
#partition_bins = list(range(0,120,3))+list(range(120,200,8))+list(range(200,240,20))+[max_hodges[h21_check]] #list(range(0,260,20))+[max_hodges[h21_check]] #...set the bins (to sort hodge number by)
partition_bins = list(qcut(Hodge[:,h21_check],q=number_bins,retbins=True,precision=3)[1]) #...partition bins to be roughly even sizes
partition_bins[0] = 0.999 #...reset initial bin st h=1 will be included within this
#print(partition_bins)

#Partition data according to Hodge number
partitioned_hodge = [[] for i in range(len(partition_bins)-1)]
for CYi in range(len(CY)): ####also save hodge in #s below line?
    partitioned_hodge[np.where(partition_bins<Hodge[CYi][h21_check])[0][-1]].append(CY[CYi]) #...identify which bin the equiv hodge number is in, and add the CY to that bins dataset
for x in partitioned_hodge: np.random.shuffle(x) #...shuffle partitioned datasets
Bin_sizes = [len(d) for d in partitioned_hodge]  #...save bin sizes/frequencies
print('Bin edges: '+str(partition_bins)+'\n\nBin sizes: '+str(Bin_sizes))
del(x,CYi)

#Run n investigations for averaging
for i in range(n):
    partitioned_hodge_copy = deepcopy(partitioned_hodge)
    #Add fake data to dataset
    partitioned_hodge_labels = [] #...labels for the real/fake data for binary classification
    for dset in range(len(partitioned_hodge_copy)):
        partitioned_hodge_copy[dset] = np.concatenate((np.array(partitioned_hodge_copy[dset]),nonCY_data[np.random.choice(nonCY_data.shape[0],len(partitioned_hodge_copy[dset]),False),:]))
        partitioned_hodge_labels.append(np.concatenate((np.ones(int(len(partitioned_hodge_copy[dset])/2)),np.zeros(int(len(partitioned_hodge_copy[dset])/2))))) #... 1 -> real, 0 -> fake
    #Train/Test partition each experiment, for each hodge number bin
    ML_data = [list(train_test_split(partitioned_hodge_copy[i],partitioned_hodge_labels[i],test_size=0.2)) for i in range(len(partitioned_hodge_copy))]
    
    #Complete all investigations for each hodge bin, saving accuracies and losses
    Weights.append([])
    Accuracies.append([])
    for exp in range(len(ML_data)):
        clf = LR(tol=1,C=100,solver='newton-cg')    #...set up model
        clf.fit(ML_data[exp][0],ML_data[exp][2])    #...train model
        Weights[-1].append(clf.coef_[0])            #...save training weights
        Predictions = clf.predict(ML_data[exp][1])  #...test model
        Accuracies[-1].append(np.sum(Predictions == ML_data[exp][3])/len(ML_data[exp][3])) #...evaluate & save testing accuracies
    del(dset,exp,clf,Predictions)
    
#Average Accuracies & Weights
Accuracies_avg = [np.mean([Accuracies[exp][hbin] for exp in range(n)]) for hbin in range(len(partition_bins)-1)]
Accuracies_var = [np.var([Accuracies[exp][hbin] for exp in range(n)]) for hbin in range(len(partition_bins)-1)]
Weights_avg = [[np.mean([Weights[exp][hbin][weight] for exp in range(n)]) for weight in range(len(CY[0]))] for hbin in  range(len(partition_bins)-1)]
Weights_var = [[np.var([Weights[exp][hbin][weight] for exp in range(n)]) for weight in range(len(CY[0]))] for hbin in  range(len(partition_bins)-1)]
#print(Accuracies_avg,'\n\n',Weights_avg)

#%% #Plotting of experiment: accuracies, frequencies, weights
#partition_bin_midpoints = (np.array(partition_bins[1:])+np.array(partition_bins[:-1]))/2
x_labels = ['$h^{1,1}$','$h^{2,1}$']
save_labels = ['h11','h21']

#Plot Bin frequnecies
plt.figure('Bin Frequencies')
#plt.title(r'Bin Frequencies: '+x_labels[h21_check])
plt.step(partition_bins[1:],Bin_sizes)
plt.xlabel(r''+x_labels[h21_check])#+' bins')
#plt.xticks(partition_bins,rotation='vertical')
plt.ylabel('Frequency')
plt.ylim(0)
plt.grid()
plt.tight_layout()
#plt.savefig('./HodgeBinFrequencies'+save_labels[h21_check]+'.pdf')

#%% #Plot Accuracies
accvar_check = 0  #...decide whether to include variance bounds on accuracy plot
plt.figure('LR Accuracies')
#plt.title(r'LR Accuracies: '+x_labels[h21_check])
plt.step(partition_bins[1:],Accuracies_avg)
if accvar_check:
    plt.fill_between(partition_bins[1:], np.array(Accuracies_avg)-np.array(Accuracies_var),np.array(Accuracies_avg)+np.array(Accuracies_var),step='pre',alpha=0.2)
plt.xlabel(r''+x_labels[h21_check])#+' bins')
#plt.xticks(partition_bins,rotation='vertical')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.grid()
plt.tight_layout()
#if accvar_check: plt.savefig('./LR_Accuracies'+save_labels[h21_check]+'.pdf')
#else: plt.savefig('./LR_Accuracies_withVariance'+save_labels[h21_check]+'.pdf')

#%% #Plot Weights
plt.figure('LR Weights')
#plt.title(r'LR Weights: '+x_labels[h21_check])
for i in range(5):
    plt.step(partition_bins[1:],[hbin[i] for hbin in Weights_avg],label=str(i+1))
    plt.fill_between(partition_bins[1:], np.array([hbin[i] for hbin in Weights_avg])-np.array([hbin[i] for hbin in Weights_var]), np.array([hbin[i] for hbin in Weights_avg])+np.array([hbin[i] for hbin in Weights_var]),step='pre',alpha=0.2)
plt.xlabel(r''+x_labels[h21_check])#+' bins')
#plt.xticks(partition_bins,rotation='vertical')
plt.ylabel('LR Weight Value')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
#plt.savefig('./LR_Weights_withVariance'+save_labels[h21_check]+'.pdf')

#%% #Plot Accuracy variances
plt.figure('LR Accuracy Variances')
#plt.title(r'LR Accuracy Variances: '+x_labels[h21_check])
plt.step(partition_bins[1:],Accuracies_var)
plt.xlabel(r''+x_labels[h21_check])#+' bins')
#plt.xticks(partition_bins,rotation='vertical')
plt.ylabel('Accuracy Variance')
#plt.ylim(0,1)
plt.grid()
plt.tight_layout()
#plt.savefig('./LR_AccuraciesVariances'+save_labels[h21_check]+'.pdf')

#%% #Plot Weights variances
plt.figure('LR Weights Variances')
#plt.title(r'LR Weights Variances: '+x_labels[h21_check])
for i in range(5):
    plt.step(partition_bins[1:],[hbin[i] for hbin in Weights_var],label=str(i+1))
plt.xlabel(r''+x_labels[h21_check])#+' bins')
#plt.xticks(partition_bins,rotation='vertical')
plt.ylabel('LR Weight Value Variance')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
#plt.savefig('./LR_WeightsVariances'+save_labels[h21_check]+'.pdf')
