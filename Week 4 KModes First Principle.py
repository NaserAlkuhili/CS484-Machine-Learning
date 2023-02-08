"""
@Name: Week 4 KModes First Principle.py
@Creation Date: September 19, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas

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

# A function to identify cluster membership using the Euclidean distance
def KModesCluster (trainData, nCluster, nIteration = 500, tolerance = 1e-7):

   # Initialize
   feature_name = trainData.columns
   n_feature = len(feature_name)

   print('=== Feature Count ===')
   feature_count = pandas.Series(index = feature_name, dtype = object)
   for f in feature_name:
      feature_count[f] = trainData[f].value_counts(dropna = False)
      print('\nFeature: ', f)
      print(feature_count[f])

   centroid = unique_row (trainData, nCluster, feature_name)
   member_prev = numpy.zeros(trainData.shape[0])

   for iter in range(nIteration):
      distance = boolean_distance(trainData, centroid)
      # distance = global_frequency(trainData, centroid, feature_name, feature_count)
      member = numpy.argmin(distance, axis = 1)
      wc_distance = numpy.min(distance, axis = 1)

      print('==================')
      print('Iteration = ', iter)
      print('Centroid: \n', centroid)
      print('Distance: \n', distance)
      print('Member: \n', member)

      centroid = []
      for cluster in range(nCluster):
         inCluster = [m == cluster for m in member]
         if (numpy.sum(inCluster) > 0):
            cur_centroid = mymode(trainData.iloc[inCluster], feature_name)
         else:
            cur_centroid = [' '] * n_feature
         centroid.append(cur_centroid)
      centroid = pandas.DataFrame(centroid, columns = feature_name)

      member_diff = numpy.sum(numpy.abs(member - member_prev))
      if (member_diff > 0):
          member_prev = member
      else:
          break

   return (member, centroid, wc_distance)

trainData = pandas.DataFrame({'Color': ['Black', 'Black', 'Black', 'Yellow', 'Black', 'Black', 'Yellow', 'Black', 'Yellow', 'Yellow'],
                              'Pet': ['Cat', 'Dog', 'Cat', 'Cat', 'Cat', 'Cat', 'Dog', 'Cat', 'Dog', 'Dog'],
                              'Gender': ['Female', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male']})

nCluster = 2
member, centroid, wc_distance = KModesCluster(trainData, nCluster)

for i in range(nCluster):
   print('\nCluster Number = ', i)
   print(trainData.iloc[member == i])

sse_cluster = []
elbow_cluster = []
for nCluster in range(1,7,1):
   member, centroid, wc_distance = KModesCluster(trainData, nCluster)
   sse_cluster.append(sum(wc_distance))
   elbow = 0.0
   for i in range(nCluster):
      sse = 0.0
      nc = 0
      for u, v in zip(member, wc_distance):
         if (u == i):
            nc = nc + 1
            sse = sse + v
      if (nc > 0):
         elbow = elbow + sse / nc
   elbow_cluster.append(elbow)

plt.figure(figsize = (8,5), dpi = 200)
plt.plot(range(1,7,1), sse_cluster, marker = 'o', linestyle = '--' )
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.grid(axis = 'both')
plt.show()

plt.figure(figsize = (8,5), dpi = 200)
plt.plot(range(1,7,1), elbow_cluster, marker = 'o', linestyle = '--' )
plt.xlabel('Number of Clusters')
plt.ylabel('Elbow Value')
plt.grid(axis = 'both')
plt.show()