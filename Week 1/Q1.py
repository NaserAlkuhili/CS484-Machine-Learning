import matplotlib.pyplot as plt
import numpy
import pandas as pd

# Data
input_data = pd.read_csv('Week 1/data/Gamma4804.csv')
Y = input_data['x']


# Q1: What are the count, the mean, the standard deviation, the minimum, the 25th percentile, the median, the 75th percentile, 
# and the maximum of the feature x? Please round your answers to the seventh decimal place.
print(Y.describe())


# Q2: Use the Shimazaki and Shinomoto (2007) method to recommend a bin width.  We will try d = 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, and 100.
# What bin width would you recommend if we want the number of bins to be between 10 and 100 inclusively?
def shimazaki_Cd(y, d_list):
   number_bins = []
   matrix_boundary = {}
   Cd_list = []
   y_low_list = []
   y_middle_list = []
   y_high_list = []
   
   y_max = numpy.max(Y)
   y_min = numpy.min(Y)
   y_mean = numpy.mean(Y)

      # Loop through the bin width candidates
   for delta in d_list:
      y_middle = delta * numpy.round(y_mean / delta)
      nBinLeft = numpy.ceil((y_middle - y_min) / delta)
      y_low = y_middle - nBinLeft * delta
      # Assign observations to bins starting from 0
      m = numpy.ceil((y_max-y_min)/delta)
      y_high = y_low + m * delta
      list_boundary = []

      bin_index = 0
      bin_boundary = y_low

      for i in numpy.arange(m):
         bin_boundary = bin_boundary + delta
         bin_index = numpy.where(y > bin_boundary, i+1, bin_index)
         list_boundary.append(bin_boundary)


      # Count the number of observations in each bins
      uvalue, ucount = numpy.unique(bin_index, return_counts = True)

      # Calculate the average frequency
      mean_ucount = numpy.mean(ucount)
      ssd_ucount = numpy.mean(numpy.power((ucount - mean_ucount), 2))
      Cd = (2.0 * mean_ucount - ssd_ucount) / delta / delta

      number_bins.append(m)
      matrix_boundary[delta] = list_boundary
      Cd_list.append(Cd)
      y_low_list.append(y_low)
      y_middle_list.append(y_middle)
      y_high_list.append(y_high)

   return(number_bins, matrix_boundary, Cd_list, y_low_list, y_middle_list, y_high_list)


d_list = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]
number_bins, matrix_boundary, Cd_list, y_low, y_middle, y_high = shimazaki_Cd (Y, d_list)


df = pd.DataFrame(columns = ['Bin Width','Cd', 'y_low', 'y_middle', 'y_high', 'N Bins'])
df['Bin Width'] = d_list
df['Cd'] = Cd_list
df['y_low'] = y_low
df['y_middle'] = y_middle
df['y_high'] = y_high
df['N Bins'] = number_bins

sorted_df = df.sort_values(by=['Cd']).reset_index(drop=True)
print(sorted_df)


# Q3: Draw the density estimator using your recommended bin width answer in (b). 

delta = sorted_df['Bin Width'][0]
bin_boundary = matrix_boundary[sorted_df['Bin Width'][0]]

plt.figure(figsize = (10,6), dpi = 200)
plt.hist(Y, bins = bin_boundary, align = 'mid')
plt.title('Delta = ' + str(delta))
plt.ylabel('Number of Observations')
plt.grid(axis = 'y')
plt.show()


# The density estimator for all the deltas.
for delta in d_list:
   plt.figure(figsize = (10,6), dpi = 200)
   bin_boundary = matrix_boundary[delta]
   plt.hist(Y, bins = bin_boundary, align = 'mid')
   plt.title('Delta = ' + str(delta))
   plt.ylabel('Number of Observations')
   plt.grid(axis = 'y')
   plt.show()