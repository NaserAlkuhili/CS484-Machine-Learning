import numpy
import pandas as pd
import random
from sklearn.model_selection import train_test_split


# Data
input_data = pd.read_csv('Week 1/data/hmeq.csv')

#initializing the random seed
random.seed(a = 20230101)

print("-"*10+" Q1 "+"-"*10)
# Q1: Before we partition the observations, we need a baseline for reference.  How many observations are in the dataset?  
# What are the frequency distributions of BAD (including missing)? What are the means and the standard deviations of DEBTINC, LOAN, MORTDUE, and VALUE?

num_of_observations = len(input_data)
BAD_freq = input_data['BAD'].value_counts()
DEBTINC_mean = input_data['DEBTINC'].mean()
DEBTINC_std = input_data['DEBTINC'].std()
LOAN_mean = input_data['LOAN'].mean()
LOAN_std = input_data['LOAN'].std()
MORTDUE_mean = input_data['MORTDUE'].mean()
MORTDUE_std = input_data['MORTDUE'].std()
VALUE_mean = input_data['VALUE'].mean()
VALUE_std = input_data['VALUE'].std()


print(f"The number of observations  in the dataset: {num_of_observations}\n")
print(f"frequency distributions of BAD \n{BAD_freq}\n")
# OR
# uvalue, ucount = numpy.unique(input_data["BAD"], return_counts = True)
# print(f"Unique Values of BAD:\n {uvalue}")
# print(f"Unique Counts of BAD:\n {ucount}")
print(f"mean of DEBTINC: {DEBTINC_mean}")
print(f"standard deviation of DEBTINC: {DEBTINC_std}\n")
print(f"mean of LOAN: {LOAN_mean}")
print(f"standard deviation of LOAN: {LOAN_std}\n")
print(f"mean of MORTDUE: {MORTDUE_mean}")
print(f"standard deviation of MORTDUE: {MORTDUE_std}\n")
print(f"mean of VALUE: {VALUE_mean}")
print(f"standard deviation of VALUE: {VALUE_std}")

# OR 
# print(input_data.describe())

print("-"*10+" Q2 "+"-"*10)
# Q2: We first try the simple random sampling method. How many observations (including those with missing values in at least one variable) are in each partition?
# What are the frequency distributions of BAD (including missing) in each partition? What are the means and the standard deviations of DEBTINC, LOAN, MORTDUE, and VALUE in each partition?

def simpleRandomSample (obsIndex, trainFraction = 0.7):
   '''Generate a simple random sample

   Parameters
   ----------
   obsIndex - a list of indices to the observations
   trainFraction - the fraction of observations assigned to Training partition
                   (a value between 0 and 1)

   Output
   ------
   trainIndex - a list of indices of original observations assigned to Training partition
   '''

   trainIndex = []

   nPopulation = len(obsIndex)
   nSample = numpy.round(trainFraction * nPopulation)
   kObs = 0
   iSample = 0
   for oi in obsIndex:
      kObs = kObs + 1
      U = random.random()
      uThreshold = (nSample - iSample) / (nPopulation - kObs + 1)
      if (U < uThreshold):
         trainIndex.append(oi)
         iSample = iSample + 1

      if (iSample == nSample):
         break

   testIndex = list(set(obsIndex) - set(trainIndex))
   return (trainIndex, testIndex)

obsIndex = [x for x in range(len(input_data))]

trainIndex, testIndex = simpleRandomSample (obsIndex, trainFraction = 0.7)
train_data = []
test_data = []

for index in trainIndex:
    train_data.append(input_data.iloc[index])

for index in testIndex:
    test_data.append(input_data.iloc[index])

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# Answers for the training partition
print("For the training partition:")
BAD_freq_train = train_data['BAD'].value_counts()
DEBTINC_mean_train = train_data['DEBTINC'].mean()
DEBTINC_std_train = train_data['DEBTINC'].std()
LOAN_mean_train = train_data['LOAN'].mean()
LOAN_std_train = train_data['LOAN'].std()
MORTDUE_mean_train = train_data['MORTDUE'].mean()
MORTDUE_std_train = train_data['MORTDUE'].std()
VALUE_mean_train = train_data['VALUE'].mean()
VALUE_std_train = train_data['VALUE'].std()
print(f"The number of observations  in the dataset: {len(train_data)}\n")
print(f"frequency distributions of BAD \n{BAD_freq_train}\n")
print(f"mean of DEBTINC: {DEBTINC_mean_train}")
print(f"standard deviation of DEBTINC: {DEBTINC_std_train}\n")
print(f"mean of LOAN: {LOAN_mean_train}")
print(f"standard deviation of LOAN: {LOAN_std_train}\n")
print(f"mean of MORTDUE: {MORTDUE_mean_train}")
print(f"standard deviation of MORTDUE: {MORTDUE_std_train}\n")
print(f"mean of VALUE: {VALUE_mean_train}")
print(f"standard deviation of VALUE: {VALUE_std_train}\n")

# Answers for the testing partition
print("For the testing partition:")
BAD_freq_test = test_data['BAD'].value_counts()
DEBTINC_mean_test = test_data['DEBTINC'].mean()
DEBTINC_std_test = test_data['DEBTINC'].std()
LOAN_mean_test = test_data['LOAN'].mean()
LOAN_std_test = test_data['LOAN'].std()
MORTDUE_mean_test = test_data['MORTDUE'].mean()
MORTDUE_std_test = test_data['MORTDUE'].std()
VALUE_mean_test = test_data['VALUE'].mean()
VALUE_std_test = test_data['VALUE'].std()
print(f"The number of observations  in the dataset: {len(test_data)}\n")
print(f"frequency distributions of BAD \n{BAD_freq_test}\n")
print(f"mean of DEBTINC: {DEBTINC_mean_test}")
print(f"standard deviation of DEBTINC: {DEBTINC_std_test}\n")
print(f"mean of LOAN: {LOAN_mean_test}")
print(f"standard deviation of LOAN: {LOAN_std_test}\n")
print(f"mean of MORTDUE: {MORTDUE_mean_test}")
print(f"standard deviation of MORTDUE: {MORTDUE_std_test}\n")
print(f"mean of VALUE: {VALUE_mean_test}")
print(f"standard deviation of VALUE: {VALUE_std_test}\n")

# OR
# print(train_data.describe())
# print(test_data.describe())

print("-"*10+" Q3 "+"-"*10)
# We next try the stratified random sampling method.  We use BAD and REASON to jointly define the strata. Since the strata variables may contain missing values, 
# we will replace the missing values in BAD with the integer 99 and in REASON with the string  ‘MISSING’.  What are the frequency distributions of BAD (including missing) in each partition?
# What are the means and the standard deviations of DEBTINC, LOAN, MORTDUE, and VALUE in each partition?

# replacing missing values in BAD and REASON.
input_data['BAD'] = input_data['BAD'].fillna(99)
input_data['REASON'] = input_data['REASON'].fillna("MISSING")


stratified_train_data, stratified_test_data = train_test_split(input_data, test_size=0.3, random_state=20230101,stratify=input_data[["BAD", "REASON"]])

# Answers for the training partition
print("For the training partition:")
BAD_freq_train = stratified_train_data['BAD'].value_counts()
DEBTINC_mean_train = stratified_train_data['DEBTINC'].mean()
DEBTINC_std_train = stratified_train_data['DEBTINC'].std()
LOAN_mean_train = stratified_train_data['LOAN'].mean()
LOAN_std_train = stratified_train_data['LOAN'].std()
MORTDUE_mean_train = stratified_train_data['MORTDUE'].mean()
MORTDUE_std_train = stratified_train_data['MORTDUE'].std()
VALUE_mean_train = stratified_train_data['VALUE'].mean()
VALUE_std_train = stratified_train_data['VALUE'].std()
print(f"The number of observations  in the dataset: {len(stratified_train_data)}\n")
print(f"frequency distributions of BAD \n{BAD_freq_train}\n")
print(f"mean of DEBTINC: {DEBTINC_mean_train}")
print(f"standard deviation of DEBTINC: {DEBTINC_std_train}\n")
print(f"mean of LOAN: {LOAN_mean_train}")
print(f"standard deviation of LOAN: {LOAN_std_train}\n")
print(f"mean of MORTDUE: {MORTDUE_mean_train}")
print(f"standard deviation of MORTDUE: {MORTDUE_std_train}\n")
print(f"mean of VALUE: {VALUE_mean_train}")
print(f"standard deviation of VALUE: {VALUE_std_train}\n")

# Answers for the testing partition
print("For the testing partition:")
BAD_freq_test = stratified_test_data['BAD'].value_counts()
DEBTINC_mean_test = stratified_test_data['DEBTINC'].mean()
DEBTINC_std_test = stratified_test_data['DEBTINC'].std()
LOAN_mean_test = stratified_test_data['LOAN'].mean()
LOAN_std_test = stratified_test_data['LOAN'].std()
MORTDUE_mean_test = stratified_test_data['MORTDUE'].mean()
MORTDUE_std_test = stratified_test_data['MORTDUE'].std()
VALUE_mean_test = stratified_test_data['VALUE'].mean()
VALUE_std_test = stratified_test_data['VALUE'].std()
print(f"The number of observations  in the dataset: {len(stratified_test_data)}\n")
print(f"frequency distributions of BAD \n{BAD_freq_test}\n")
print(f"mean of DEBTINC: {DEBTINC_mean_test}")
print(f"standard deviation of DEBTINC: {DEBTINC_std_test}\n")
print(f"mean of LOAN: {LOAN_mean_test}")
print(f"standard deviation of LOAN: {LOAN_std_test}\n")
print(f"mean of MORTDUE: {MORTDUE_mean_test}")
print(f"standard deviation of MORTDUE: {MORTDUE_std_test}\n")
print(f"mean of VALUE: {VALUE_mean_test}")
print(f"standard deviation of VALUE: {VALUE_std_test}\n")
