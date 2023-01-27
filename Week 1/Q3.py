import matplotlib.pyplot as plt
import numpy
import pandas as pd
import random
from sklearn.model_selection import train_test_split


# Data
df = pd.read_csv('Week 1/data/Fraud.csv')

print("-"*10+" Q1 "+"-"*10)
# Q1
# What percent of investigations are found to be frauds?  This is the empirical fraud rate.  Please round your answers to the fourth decimal place.
num_of_frauds = df['FRAUD'].value_counts()[1]
print(f'The percentage of investigations  found to be frauds is {num_of_frauds/len(df)}')


print("-"*10+" Q2 "+"-"*10)
# Q2
# We will divide the complete observations into 80% Training and 20% Testing partitions. A complete observation does not contain missing values in any of the variables.
# The random seed is 20230225.  The stratum variable is FRAUD.  How many observations are in each partition?

#initializing the random seed

df_train, df_test = train_test_split(df, test_size=0.2, random_state=20230101, stratify=df["FRAUD"])

print(f'The number of observations in the training partition is {len(df_train)}')
print(f'The number of observations in the testing partition is {len(df_test)}')