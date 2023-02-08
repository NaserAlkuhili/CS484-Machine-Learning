"""
@Name: Week 4 KModes hmeq.py
@Creation Date: September 19, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import random

# Return the modes of the columns of a dataframe in a list
def mymode (df, columns):
   column_mode = []
   for v in columns:
      feature_count = df[v].value_counts(dropna = False)
      column_mode.append(feature_count.index[0])

   return (column_mode)

# Return the first nCluster unique rows in a dataframe
def unique_row (df, nCluster, columns):
   centroid = []
   n_centroid = 0
   for index_0, row_0 in trainData.iterrows():
      if (0 < n_centroid and n_centroid < nCluster):
         for i in range(n_centroid):
            if (numpy.sum(numpy.where(row_0 == centroid[i], 0, 1)) > 0):
               centroid.append(row_0.to_list())
               n_centroid += 1
               break
      elif (n_centroid == 0):
         centroid.append(row_0.to_list())
         n_centroid += 1
      elif (n_centroid == nCluster):
         break

   centroid_df = pandas.DataFrame(centroid, columns = columns)
   return (centroid_df)

# Return the Boolean distances between each row in a dataframe and each row of centroid in a nested list
def boolean_distance (df, centroid):
   df_distance = []
   for index_0, row_0 in df.iterrows():
      row_distance = []
      for index_1, row_1 in centroid.iterrows():
         a_distance = numpy.sum([v0 != v1 for (v0, v1) in zip(row_0, row_1)])
         row_distance.append(a_distance)
      df_distance.append(row_distance)

   return (df_distance)

# Return the Global Frequency distances between each row in a dataframe and each row of centroid in a nested list
def global_frequency (df, centroid, columns, feature_count):

   df_distance = []
   for index_0, row_0 in df.iterrows():
      row_distance = []

      reciprocal_0 = []
      for v_0 in columns:
         count_0 = feature_count.loc[v_0]
         reciprocal_0.append(1.0 / count_0[row_0[v_0]])

      for index_1, row_1 in centroid.iterrows():
         if (numpy.sum(numpy.where(row_1 == row_0, 0, 1)) > 0):
            reciprocal_1 = []
            for v_1 in columns:
               count_1 = feature_count.loc[v_1]
               reciprocal_1.append(1.0 / count_1[row_1[v_1]])
            a_distance = sum(reciprocal_0 + reciprocal_1)
         else:
            a_distance = 0.0
         row_distance.append(a_distance)

      df_distance.append(row_distance)

   return (df_distance)

# A function to identify cluster membership using the categorical distance
def KModesCluster (trainData, nCluster, distance = 'BOOLEAN', nIteration = 500, tolerance = 1e-7):

   # Initialize
   feature_name = trainData.columns

   feature_count = pandas.Series(index = feature_name, dtype = object)
   for f in feature_name:
      feature_count[f] = trainData[f].value_counts(dropna = False)

   cur_centroid = unique_row (trainData, nCluster, feature_name)
   member_prev = numpy.zeros(trainData.shape[0])

   for iter in range(nIteration):

      if (distance == 'BOOLEAN'):
          distance = boolean_distance(trainData, cur_centroid)
      else:
          distance = global_frequency(trainData, cur_centroid, feature_name, feature_count)
      member = numpy.argmin(distance, axis = 1)
      wc_distance = numpy.min(distance, axis = 1)

      nxt_centroid = []
      for cluster in range(nCluster):
         inCluster = [m == cluster for m in member]
         if (numpy.sum(inCluster) > 0):
            row_centroid = mymode(trainData.iloc[inCluster], feature_name)
         else:
            this_row = random.randint(0,trainData.shape[0])
            row_centroid = trainData.iloc[this_row]
         nxt_centroid.append(row_centroid)
      cur_centroid = pandas.DataFrame(nxt_centroid, columns = feature_name)

      member_diff = numpy.sum(numpy.abs(member - member_prev))
      if (member_diff > 0):
          member_prev = member
      else:
          break

   return (member, cur_centroid, wc_distance)

hmeq = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\hmeq.csv')

trainData = hmeq[['BAD','REASON','JOB','DEROG']].dropna().reset_index(drop = True)
random.seed(484)

print(trainData['BAD'].value_counts())

print(trainData['REASON'].value_counts())

print(trainData['JOB'].value_counts())

print(trainData['DEROG'].value_counts())

# Define the aggregation procedure outside of the groupby operation
aggregations = {
    'member':'count',
    'wc_distance': 'sum'
}

sse_cluster = []
elbow_cluster = []
num_cluster = range(1,11,1)

for nCluster in num_cluster:
   print('Number of Cluster = ', nCluster)
   member, centroid, wc_distance = KModesCluster(trainData, nCluster, distance = 'GLOBAL')
   sse_cluster.append(sum(wc_distance))

   df = pandas.DataFrame(list(zip(member, wc_distance)), columns = ['member','wc_distance'])
   elbow = df.groupby('member').agg(aggregations)
   elbow['elbow'] = elbow['wc_distance'] / elbow['member']
   elbow_cluster.append(sum(elbow['elbow']))

plt.figure(figsize = (8,5), dpi = 200)
plt.plot(num_cluster, sse_cluster, marker = 'o', linestyle = '--' )
plt.xticks(num_cluster)
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.grid(axis = 'both')
plt.show()

plt.figure(figsize = (8,5), dpi = 200)
plt.plot(num_cluster, elbow_cluster, marker = 'o', linestyle = '--' )
plt.xticks(num_cluster)
plt.xlabel('Number of Clusters')
plt.ylabel('Elbow Value')
plt.grid(axis = 'both')
plt.show()

# Four Clusters Solution with Boolean Distance
member, centroid, wc_distance = KModesCluster(trainData, 4, distance = 'GLOBAL')

cluster_id, cluster_count = numpy.unique(member, return_counts=True)
