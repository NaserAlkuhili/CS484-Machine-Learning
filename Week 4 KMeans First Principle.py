"""
@Name: Week 4 KMeans First Principle.py
@Creation Date: February 2, 2023
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

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

trainData = pandas.DataFrame({'x': [0.1, 0.3, 0.4, 0.8, 0.9]})

member, centroid, wc_distance = KMeansCluster(trainData, 2)
twcss = numpy.sum(numpy.power(wc_distance,2))
