'''Hodge Plots & K-Means clustering'''
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ast import literal_eval
from sklearn.cluster import KMeans
from collections import Counter

#%% #Import data
with open('./Data/CY.txt','r') as f0:
    CY = np.array(literal_eval(f0.read()))
    
#Import Hodge numbers
with open('./Data/Hodge.txt','r') as f_hodge:
    Hodge = np.array(literal_eval(f_hodge.read()))
    
del(f0,f_hodge)

#%% #Plot each weight vs h11 or h21
#Select hyperparams to plot
w_idx = 5  #...choose weight to plot: 1->5
h = 0      #...choose: 0 for h11, 1 for h21

#Plot scatter graph across CY dataset
plt.figure('Hodge Number Correlations: '+str(w_idx))
plt.scatter(CY[:,w_idx-1],Hodge[:,h],alpha=0.1)
plt.xlabel(r'$weight $ '+str(w_idx))
plt.ylabel(r'$h^{2,1}$')
plt.tight_layout()
plt.grid()
#plt.savefig('h21_vs_w'+str(w_idx)+'.pdf')

#%% #3d plot each weight vs h11 and h21
#Select weight to plot
w_idx = 5

#Plot 3d scatter graph with both Hodge numbers
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(CY[:,w_idx-1],Hodge[:,0],Hodge[:,1],alpha=0.1)
ax.set_xlabel(r'$weight $ '+str(w_idx))
ax.set_ylabel(r'$h^{1,1}$')
ax.set_zlabel(r'$h^{2,1}$')
ax.view_init(30, 30) #...adjust viewpoint of 3d plot here
ax.xaxis.labelpad=15
ax.yaxis.labelpad=12
ax.zaxis.labelpad=12
ax.dist = 12.5
#plt.savefig('./3d_w'+str(w_idx)+'_vs_hs.pdf')

#%% #Plot each weight vs Euler number
#Select weight to plot
w_idx = 5
plt.figure('Weights vs Euler number')
plt.scatter(CY[:,w_idx-1],[-2*(y[0]-y[1]) for y in Hodge],alpha=0.1)
plt.xlabel(r'$weight $ '+str(w_idx))
plt.ylabel(r'$\chi$')
plt.grid()
#plt.savefig('Euler_vs_w'+str(w_idx)+'.pdf')


################################################################################
'''Clustering'''
#%% #Plot histogram of h11/w5 gradient data
#Select hyperparams to consider
w_idx = 5  #...choose weight to plot: 1->5
h = 0      #...choose: 0 for h11, 1 for h21
all_data = False #...choose whether to plot ratios for all the data, or just the 'outer' data

if all_data:
    raw_k_data = np.array([float(Hodge[x][h])/CY[x][w_idx-1] for x in range(len(CY))])
    plt.hist(raw_k_data,bins=int(max(raw_k_data)*50),range=(0,max(raw_k_data)+0.01),histtype='step')
else:
    raw_outer_k_data = np.array([float(Hodge[x][h])/CY[x][w_idx-1] for x in range(len(CY)) if CY[x][w_idx-1] > 250])
    plt.hist(raw_outer_k_data,bins=int(max(raw_outer_k_data)*50),range=(0,max(raw_outer_k_data)+0.01),histtype='step')
#for clust_cent in kmeans.cluster_centers_.flatten(): plt.axvline(x=clust_cent,color='black',lw=0.8,linestyle='--') #...unhash this to add the cluster centres to the plot (must run subsequent cell first to define them)
plt.xlabel(r'$h^{1,1}/w_5$')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
#plt.savefig('grad(h11w5)_histogram.pdf')

#%% #Perform K-Means clustering
#Select hyperparams to consider
w_idx = 5  #...choose weight to plot: 1->5
h = 0      #...choose: 0 for h11, 1 for h21
all_data = False            #...choose whether to cluster based on all the data, or focus on the outer data where classes more prominent
preset_number_clusters = 0  #...set to chosen number of clusters, or to zero to determine optimal number of clusters
max_inertia = True          #...select True to calculate optimum number of clusters using max distance to a cluster ('max-inertia'), or False to use average distance (inertia)

#Define datasets of ratios, one with only 'outer' data to encourage good cluster identification
all_ratio_data = np.array([float(Hodge[x][h])/CY[x][w_idx-1] for x in range(len(CY))]).reshape(-1,1)
outer_ratio_data = np.array([float(Hodge[x][h])/CY[x][w_idx-1] for x in range(len(CY)) if CY[x][w_idx-1] > 250]).reshape(-1,1) #...update cutoff if not considering h11 vs w5

if all_data: ratio_data = all_ratio_data
else:        ratio_data = outer_ratio_data

#Run K-Means CLustering
if preset_number_clusters:
    #Perform K-Means clustering (use preset number of clusters)
    kmeans = KMeans(n_clusters=preset_number_clusters).fit(ratio_data)  
else:
    if max_inertia:
        #Plot scaled max-inertia distribution to determine optimal number of clusters
        max_dists = []
        #Compute single cluster max squared distance
        kmeans = KMeans(n_clusters=1).fit(ratio_data)
        transformed_data = kmeans.transform(ratio_data) 
        single_clust_max_dist = max([min(x)**2 for x in transformed_data])
        #Compute the max distances to nearest cluster for each datapoint for all numbers of clusters
        for k in range(1,21):
            kmeans = KMeans(n_clusters=k).fit(ratio_data)
            transformed_data = kmeans.transform(ratio_data)    #...data transformed to list distance to all centres
            max_dists.append(max([min(x)**2 for x in transformed_data])/single_clust_max_dist + 0.01*(k-1))   #...compute the scaled max distance over the full dataset
        
        #Determine optimal number of clusters
        k_optimal = list(range(1,21))[max_dists.index(min(max_dists))]
        print('Optimal number of clusters: '+str(k_optimal))
        
        plt.figure('K-Means Max-Inertia')
        plt.scatter(list(range(1,21)),max_dists)
        plt.xlabel('Number of Clusters')
        plt.xticks(range(21))
        plt.ylabel('Scaled Max-Inertia')
        plt.ylim(0,1.05)
        plt.grid()
        plt.tight_layout()
        #plt.savefig('./KMeansScaledMaxSquared-Distance.pdf')

    else:
        #Plot scaled inertia distribution to determine optimal number of clusters
        inertia_list = []
        single_clust_inertia = KMeans(n_clusters=1).fit(ratio_data).inertia_
        for k in range(1,21):
            scaled_inertia = KMeans(n_clusters=k).fit(ratio_data).inertia_ / single_clust_inertia + 0.01*(k-1)
            inertia_list.append(scaled_inertia)
            
        #Determine optimal number of clusters
        k_optimal = list(range(1,21))[inertia_list.index(min(inertia_list))]
        print('Optimal number of clusters: '+str(k_optimal))
        
        plt.figure('K-Means Inertia')
        plt.scatter(list(range(1,21)),inertia_list)
        plt.xlabel('Number of Clusters')
        plt.xticks(range(21))
        plt.ylabel('Scaled Inertia')
        plt.ylim(0,1.05)
        plt.grid()
        plt.tight_layout()
        #plt.savefig('./KMeansInertia.pdf')
    
    #Perform K-Means clustering (use computed optimal number of clusters)
    kmeans = KMeans(n_clusters=k_optimal).fit(ratio_data)   

#Compute clustering over the full ratio data (irrespective of whether full or outer used to identify clusters)
transformed_full_data = kmeans.transform(all_ratio_data)                  #...data transformed to list distance to all centres
kmeans_labels = np.argmin(transformed_full_data,axis=1)                   #...identify the closest cluster centre to each datapoint
full_data_inertia = np.sum([min(x)**2 for x in transformed_full_data])    #...compute the inertia over the full dataset
cluster_sizes = Counter(kmeans_labels)                                    #...compute the frequencies in each cluster
print('\nCluster Centres: '+str(kmeans.cluster_centers_.flatten())+'\nCluster sizes: '+str([cluster_sizes[x] for x in range(10)])+'\n\nInertia: '+str(full_data_inertia)+'\nNormalised Inertia: '+str(full_data_inertia/7555)+'\nNormalised Inertia / range: '+str((full_data_inertia/(7555*(max(all_ratio_data)-min(all_ratio_data))))[0]))

#%% #Plot full data with cluster centre lines overlaid
plt.figure('K-Means centres overlaid')
plt.scatter(CY[:,w_idx-1],Hodge[:,h],alpha=0.1)
for grad in kmeans.cluster_centers_.flatten():
    plt.plot(np.linspace(0,2000,2),grad*np.linspace(0,2000,2),color='black',lw=0.5)
plt.xlim(0,1800)
plt.ylim(0,np.round(max(Hodge[:,h]),-1)+50)
plt.xlabel(r'$weight $ '+str(w_idx))
plt.ylabel(r'$h^{1,1}$')
plt.tight_layout()
plt.grid()
#plt.savefig('kmeans_overlaidcentres_h11vsw'+str(w_idx)+'.pdf')

#%% #Plot data with cluster bounds overlaid
centres = np.sort(kmeans.cluster_centers_.flatten())
cluster_bounds = (centres[:-1]+centres[1:])/2

plt.figure('K-Means bounds overlaid')
plt.scatter(CY[:,w_idx-1],Hodge[:,h],alpha=0.1)
for grad in cluster_bounds:
    plt.plot(np.linspace(0,2000,2),grad*np.linspace(0,2000,2),color='black',lw=0.5)
plt.xlim(0,1800)
plt.ylim(0,np.round(max(Hodge[:,h]),-1)+50)
plt.xlabel(r'$weight $ '+str(w_idx))
plt.ylabel(r'$h^{1,1}$')
plt.tight_layout()
plt.grid()
#plt.savefig('kmeans_overlaidbounds_h11vsw'+str(w_idx)+'.pdf')

#%% #Plot the clusters in different colours
plt.figure('K-Means clusters coloured')
for cluster_idx in list(range(len(cluster_sizes))):
    plt.scatter([CY[x][w_idx-1] for x in range(len(CY)) if kmeans_labels[x] == cluster_idx],[Hodge[y][h] for y in range(len(Hodge)) if kmeans_labels[y] == cluster_idx],alpha=0.1)
plt.xlim(0,1800)
plt.ylim(0,np.round(max(Hodge[:,h]),-1)+50)
plt.xlabel(r'$weight $ '+str(w_idx))
plt.ylabel(r'$h^{1,1}$')
plt.tight_layout()
plt.grid()
#plt.savefig('kmeans_clusterscoloured_h11vsw'+str(w_idx)+'.pdf')

#%% #Save the outer data in each cluster to a file for comparison
#Identify the number of clusters
if preset_number_clusters: number_clusters = preset_number_clusters
else:                      number_clusters = k_optimal
#Extract outer data and cluster labels
CY_outer = [list(CY[x]) for x in range(len(CY)) if CY[x][w_idx-1] > 250]
CY_outer_lables = kmeans.labels_
#Sort into respective clusters
clusters = [[] for i in range(number_clusters)]
for v_idx in range(len(CY_outer)):
    clusters[CY_outer_lables[v_idx]].append(list(CY_outer[v_idx]))
#Save to a file
with open('./CluseredOuterData.txt','w') as file:
    file.write('Cluster Sorted Outer Data: (note clusters not ordered)')
    for c_idx in range(len(clusters)-1):
        file.write('\n\nCluster '+str(c_idx+1)+' (centre: '+str(kmeans.cluster_centers_.flatten()[c_idx])+')\n'+str(clusters[c_idx])+'\n\n##########')
    file.write('\n\nCluster '+str(len(clusters))+' (centre: '+str(kmeans.cluster_centers_.flatten()[-1])+')\n'+str(clusters[-1]))
    