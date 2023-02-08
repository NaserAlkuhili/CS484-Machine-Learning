import matplotlib.pyplot as plt
import numpy
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Data
df = pd.read_csv('Week1/data/Fraud.csv')
random.seed(a = 20230225)


print("-"*10+" Q1 "+"-"*10)
# Q1
# What percent of investigations are found to be frauds?  This is the empirical fraud rate.  Please round your answers to the fourth decimal place.
num_of_frauds = df['FRAUD'].value_counts()[1]
print(f'The percentage of investigations  found to be frauds is {round((num_of_frauds/len(df))*100, 4)}%')


print("-"*10+" Q2 "+"-"*10)
# Q2
# We will divide the complete observations into 80% Training and 20% Testing partitions. A complete observation does not contain missing values in any of the variables.
# The random seed is 20230225.  The stratum variable is FRAUD.  How many observations are in each partition?

X = df[['TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']]

Y = df['FRAUD']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20230225, stratify=df["FRAUD"])

print(f'The number of observations in the training partition is {len(x_train)}')
print(f'The number of observations in the testing partition is {len(x_test)}')


print("-"*10+" Q3 "+"-"*10)
# Q3
# Use the KNeighborsClassifier module to train the Nearest Neighbors algorithm.  We will try the number of neighbors from 2 to 7 inclusively.
# We will classify an observation as a fraud if the proportion of FRAUD = 1 among its neighbors is greater than or equal to the empirical fraud rate (rounded to the fourth decimal place).
# What are the misclassification rates of these numbers of neighbors in each partition?
def nbrs_metric (class_prob, y_class, n_class):
    empirical_rate = round((num_of_frauds/len(df)), 4)
    nbrs_pred = numpy.array([])
    for pred in class_prob:
        if pred[1] >= empirical_rate:
            nbrs_pred = numpy.append(nbrs_pred, 1)
        else:
            nbrs_pred = numpy.append(nbrs_pred, 0)
    
    mce = numpy.mean(numpy.where(nbrs_pred == y_class, 0, 1))

    rase = 0.0
    for i in range(y_class.shape[0]):
        for j in range(n_class):
            if (y_class.iloc[i] == j):
                rase = rase + numpy.power((1.0 - class_prob[i,j]),2)
            else:
                rase = rase + numpy.power(class_prob[i,j],2)
    rase = numpy.sqrt(rase / y_class.shape[0] / n_class)
    return (mce, rase)   
result = []
for i in range(2,8):
    neigh = KNeighborsClassifier(n_neighbors = i, metric = 'euclidean')
    nbrs = neigh.fit(x_train, y_train)
   
    cprob_train = nbrs.predict_proba(x_train)
    mce_train, rase_train = nbrs_metric(cprob_train, y_train, 2)


    cprob_test = nbrs.predict_proba(x_test)
    mce_test, rase_test = nbrs_metric(cprob_test, y_test, 2)



    result.append([i, round(mce_train,4), round(mce_test,4), round(rase_train,4), round(rase_test,4)])

result_df = pd.DataFrame(result, columns = ['k', 'MCE_Train', 'MCE_Test', 'RASE_Train', 'RASE_Test'])
print(result_df)



print("-"*10+" Q4 "+"-"*10)
# Q4
# Which number of neighbors will yield the lowest misclassification rate in the Testing partition?
# In the case of ties, choose the smallest number of neighbors.

print('The number of neighbors that will yield the lowest misclassification rate in the Testing partition is 6')

print("-"*10+" Q5 "+"-"*10)
# Q5
# Consider this focal observation where DOCTOR_VISITS is 8, MEMBER_DURATION is 178, NUM_CLAIMS is 0, NUM_MEMBERS is 2, OPTOM_PRESC is 1, and TOTAL_SPEND is 16300.
# Use your selected model from Part (d) and find its neighbors.  What are the neighbors’ observation values?  Also, calculate the predicted probability that this observation is a fraud.
neigh = KNeighborsClassifier(n_neighbors = 6, metric = 'euclidean')
nbrs = neigh.fit(x_train, y_train)


focal = pd.DataFrame([[16300, 8, 0, 178, 1, 2]], columns=X.columns)

myNeighbors = neigh.kneighbors(focal, return_distance=False)
print('The Neighbors: ')
print(df.iloc[x_train.iloc[myNeighbors[0]].index])

print(f'The predicted probability that this observation is a fraud is: {round(neigh.predict_proba(focal)[0][1]*100, 4)}%')
