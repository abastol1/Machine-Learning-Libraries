#!/usr/bin/env python
# coding: utf-8

# ___
# 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

projectData = pd.read_csv("KNN_Project_Data")

# **Check the head of the dataframe.**

projectData.head()

# Plot pairwise relationships in a dataset.
# hue='TARGET CLASS' --> Variable in `projectData` to map plot aspects to different colors.

sns.pairplot(projectData, palette='coolwarm')

from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

scaler = StandardScaler()

# ** Fit scaler to the features.**

scaler.fit(projectData.drop('TARGET CLASS', axis=1))


# **Use the .transform() method to transform the features to a scaled version.**

scaled_features = scaler.transform(projectData.drop('TARGET CLASS', axis = 1))

# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**


df_feature = pd.DataFrame(scaled_features, columns=projectData.columns[:-1])
df_feature.head()


# # Train Test Split
# **Use train_test_split to split your data into a training set and a testing set.**

from sklearn.model_selection import train_test_split


X = df_feature
y = projectData['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# **Import KNeighborsClassifier from scikit learn.**

from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**
knn = KNeighborsClassifier(n_neighbors=1)


# **Fit this KNN model to the training data.**

knn.fit(X_train, y_train)


# **Use the predict method to predict values using your KNN model and X_test.**


prediction = knn.predict(X_test)


# ** Create a confusion matrix and classification report.**

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))


# checking to see in what value of n_neighbor, the data will be predicted more accuretly
error_rate= []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    prediction_I = knn.predict(X_test)
    error_rate.append(np.mean(prediction_I != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red' )


knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

