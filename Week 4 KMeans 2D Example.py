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

def KMeansCluster (trainData, nCluster, distType = 'chebyshev', nIteration = 500, nTrial = 10, randomSeed = None):
   n_obs = trainData.shape[0]

   if (randomSeed is not None):
      random.seed(a = randomSeed)

   list_centroid = []
   list_wcss = []
   for iTrial in range(nTrial):
      centroid_pos = getPositionSRS(n_obs, nCluster)
      centroid = trainData.iloc[centroid_pos]
      member_prev = pandas.Series([-1] * n_obs, name = 'Cluster')

      for iter in range(nIteration):
         member, wc_distance = assignMember(trainData, centroid, distType)

         centroid = trainData.join(member).groupby(by = ['Cluster']).mean()
         member_diff = numpy.sum(numpy.abs(member - member_prev))
         if (member_diff > 0):
            member_prev = member
         else:
            break

      print(centroid)

      list_centroid.append(centroid)
      list_wcss.append(numpy.sum(wc_distance))

   best_solution = numpy.argmin(list_wcss)
   centroid = list_centroid[best_solution]
   
   member, wc_distance = assignMember(trainData, centroid, distType)
   
   return (member, centroid, wc_distance)

trainData = pandas.DataFrame({'x': [2, 4, 2, 4, 6, 8, 6, 8, 4.5, 5.5, 5.5, 4.5],
                              'y': [11, 11, 9, 9, 11, 11, 9, 9, 5.5, 5.5, 4.5, 4.5]})


n_sample = trainData.shape[0]
max_nCluster = 10

nClusters = []
Elbow = []
Silhouette = []
CH_score = []
DB_score = []
TotalWCSS = []

for k in range(max_nCluster):
   nCluster = k + 1
   member, centroid, wc_distance = KMeansCluster(trainData, nCluster, nTrial = 20, randomSeed = 20231225)

   if (nCluster > 1):
       S = metrics.silhouette_score(trainData, member)
       CH = metrics.calinski_harabasz_score(trainData, member)
       DB = metrics.davies_bouldin_score(trainData, member)
   else:
       S = numpy.NaN
       CH = numpy.NaN
       DB = numpy.NaN

   WCSS = numpy.zeros(nCluster)
   nC = numpy.zeros(nCluster)

   for i in range(n_sample):
      k = member[i]
      nC[k] += 1 
      diff = trainData.iloc[i] - centroid.iloc[k]
      WCSS[k] += diff.dot(diff)

   E = 0.0
   T = 0.0
   for k in range(nCluster):
      E += WCSS[k] / nC[k]
      T += WCSS[k]
 
   nClusters.append(nCluster)
   Elbow.append(E)
   Silhouette.append(S)
   CH_score.append(CH)
   DB_score.append(DB)
   TotalWCSS.append(T)
  
plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, TotalWCSS, marker = 'o', color = 'royalblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Within-Cluster Sum of Squares')
plt.xticks(range(1,max_nCluster+1))
plt.grid()
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, Elbow, marker = 'o', color = 'royalblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Elbow Value')
plt.xticks(range(1,max_nCluster+1))
plt.grid()
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, Silhouette, marker = 'o', color = 'royalblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Value')
plt.xticks(range(1,max_nCluster+1))
plt.grid()
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, CH_score, marker = 'o', color = 'royalblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.xticks(range(1,max_nCluster+1))
plt.grid()
plt.show()

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, DB_score, marker = 'o', color = 'royalblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.xticks(range(1,max_nCluster+1))
plt.grid()
plt.show()

result_df = pandas.DataFrame({'N Cluster': nClusters,
                              'Total WCSS': TotalWCSS,
                              'Elbow': Elbow,
                              'Silhouette': Silhouette,
                              'CH Score': CH_score,
                              'DB Score': DB_score})


print(result_df)