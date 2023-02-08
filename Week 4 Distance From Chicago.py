"""
@Name: Week 4 Distance From Chicago.py
@Creation Date: February 2, 2023
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import random
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

from sklearn import metrics

def getPositionSRS (nObs, nCluster):
   '''Get positions of centroids for the clusters by simple random sampling method

   Arguments:
   1. trainData - training data
   2. nObs - number of observations in training data (assume nObs > nCluster)
   3. nCluster - number of clusters

   Output:
   1. centroid_pos - positions in training data chosen as centroids
   '''

   centroid_pos = []

   kObs = 0
   iSample = 0
   for iObs in range(nObs):
      kObs = kObs + 1
      uThreshold = (nCluster - iSample) / (nObs - kObs + 1)
      if (random.random() < uThreshold):
         centroid_pos.append(iObs)
         iSample = iSample + 1

      if (iSample == nCluster):
         break

   return (centroid_pos)

def assignMember (trainData, centroid, distType):
   '''Assign observations to their nearest clusters.

   Arguments:
   1. trainData - training data
   2. centroid - centroid
   3. distType - distance metric

   Output:
   1. member - cluster memberships
   2. wc_distance - distances of observations to the nearest centroid
   '''

   pair_distance = metrics.pairwise_distances(trainData, centroid, metric = distType)
   member = pandas.Series(numpy.argmin(pair_distance, axis = 1), name = 'Cluster')
   wc_distance = pandas.Series(numpy.min(pair_distance, axis = 1), name = 'Distance')

   return (member, wc_distance)

def KMeansCluster (trainData, nCluster, distType = 'euclidean', nIteration = 500, nTrial = 10, randomSeed = None):
   n_obs = trainData.shape[0]

   if (randomSeed is not None):
      random.seed(a = randomSeed)

   list_centroid = []
   list_wcss = []
   for iTrial in range(nTrial):
      centroid_pos = getPositionSRS (n_obs, nCluster)
      centroid = trainData.iloc[centroid_pos]
      member_prev = pandas.Series([-1] * n_obs, name = 'Cluster')

      for iter in range(nIteration):
         member, wc_distance = assignMember (trainData, centroid, distType)

         centroid = trainData.join(member).groupby(by = ['Cluster']).mean()
         member_diff = numpy.sum(numpy.abs(member - member_prev))
         if (member_diff > 0):
            member_prev = member
         else:
            break

      list_centroid.append(centroid)
      list_wcss.append(numpy.sum(wc_distance))

   best_solution = numpy.argmin(list_wcss)
   centroid = list_centroid[best_solution]
   
   member, wc_distance = assignMember (trainData, centroid, distType)
   
   return (member, centroid, wc_distance)

DistanceFromChicago = pandas.read_csv('/Users/nfk/Repos/CS484/DistanceFromChicago.csv')

trainData = DistanceFromChicago[['DrivingMilesFromChicago']]
nCity = trainData.shape[0]

# Determine the number of clusters
maxNClusters = 15

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
Silhouette = numpy.zeros(maxNClusters)
Calinski_Harabasz = numpy.zeros(maxNClusters)
Davies_Bouldin = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   member, centroid, wc_distance = KMeansCluster (trainData, KClusters, distType = 'euclidean',
                                                  nIteration = 500, nTrial = 10, randomSeed = 484000)

   if (1 < KClusters):
       Silhouette[c] = metrics.silhouette_score(trainData, member)
       Calinski_Harabasz[c] = metrics.calinski_harabasz_score(trainData, member)
       Davies_Bouldin[c] = metrics.davies_bouldin_score(trainData, member)
   else:
       Silhouette[c] = numpy.NaN
       Calinski_Harabasz[c] = numpy.NaN
       Davies_Bouldin[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nCity):
      k = member.iloc[i]
      nC[k] += 1
      diff = trainData.iloc[i] - centroid.iloc[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, TotalWCSS, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Total WCSS")
plt.xticks(range(1, maxNClusters+1))
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(range(1, maxNClusters+1))
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(range(1, maxNClusters+1))
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, Calinski_Harabasz, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.xticks(range(1, maxNClusters+1))
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, Davies_Bouldin, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

result_df = pandas.DataFrame({'N Cluster': nClusters, 'Total WCSS': TotalWCSS,
                              'Elbow': Elbow, 'Silhouette': Silhouette,
                              'CH Score': Calinski_Harabasz, 'DB Score': Davies_Bouldin})

# Display the 4-cluster solution
member, centroid, wc_distance = KMeansCluster (trainData, 4, distType = 'euclidean',
                                               nIteration = 500, nTrial = 10, randomSeed = 484000)

print("Cluster Centroids = \n", centroid)
cluster_0 = DistanceFromChicago[member == 0]
cluster_1 = DistanceFromChicago[member == 1]
cluster_2 = DistanceFromChicago[member == 2]
cluster_3 = DistanceFromChicago[member == 3]

for i in range(4):
   wcss_i = numpy.sum(wc_distance[member == i])
   print('Cluster = ', i, 'WCSS = ', wcss_i)

overall_mean = numpy.mean(DistanceFromChicago)
overall_css = numpy.sum(numpy.power(DistanceFromChicago - overall_mean, 2))

cmap = ['indianred','sandybrown','royalblue', 'olivedrab']

fig, ax = plt.subplots(figsize = (10,6), dpi = 200)
for c in range(4):
   subData = DistanceFromChicago[member == c]
   plt.hist(subData['DrivingMilesFromChicago'], color = cmap[c], label = str(c), linewidth = 2, histtype = 'step')
ax.set_ylabel('Number of Cities')
ax.set_xlabel('DrivingMilesFromChicago')
ax.set_xticks(numpy.arange(0,2500,250))
plt.grid(axis = 'y')
plt.legend(loc = 'lower left', bbox_to_anchor = (0.15, 1), ncol = 4, title = 'Cluster ID')
plt.show()
