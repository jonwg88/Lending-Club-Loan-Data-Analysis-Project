#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libraries
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Create dataframe to store our dataset
loan_df = pd.read_csv("Dropbox/Coding/AI and ML Bootcamp/Course 4 Deep Learning with Keras and TensorFlow/loan_data.csv")


# In[3]:


#Feature Transformation
loan_df = pd.get_dummies(loan_df, columns=['purpose'], drop_first=True)
print(loan_df.head())


# In[4]:


#Exploratory data analysis
#Look at descriptions of variables
print(loan_df.describe())


# In[5]:


#Check distribution of target variable
sns.countplot(x='not.fully.paid', data=loan_df)
plt.title('Distribution of not.fully.paid')
plt.show()


# In[6]:


#Create correlation matrix to check variable correlations
correlation_matrix = loan_df.corr()
plt.figure(figsize=(14,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap')
plt.show()


# In[7]:


#int.rate and inq.last.6mths are most closely correlated to not.fully.paid
#Features with highest correlation are int.rate and revol.util, installment and log.annual.inc, and revol.bal and log.annual.inc


# In[8]:


#'int.rate' variable correlation with not.fully.paid
plt.figure(figsize=(14,10))
sns.countplot(x='int.rate', hue='not.fully.paid', data=loan_df)
plt.title('int.rate vs not.fully.paid')
plt.show()


# In[9]:


#'inq.last.6mths' variable correlation with not.fully.paid
plt.figure(figsize=(14,10))
sns.countplot(x='inq.last.6mths', hue='not.fully.paid', data=loan_df)
plt.title('inq.last.6mths vs not.fully.paid')
plt.show()


# In[10]:


#Additional feature engineering
#Drop features with low correlation to target variable. The least correlated features are 'credit.policy' and 'fico'.
loan_df_mod = loan_df.drop('credit.policy', axis=1)
loan_df_mod = loan_df_mod.drop('fico', axis=1)


# In[11]:


#Drop 'log.annual.inc' due to low correlation with target and high correlation with 'installment'
loan_df_mod = loan_df_mod.drop('log.annual.inc', axis=1)


# In[12]:


#Prepare data for input into model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = loan_df_mod.drop('not.fully.paid', axis=1)
y = loan_df_mod['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[72]:


#Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
model = Sequential()
#Input layer
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
#Hidden layer 1 (with added weight regularization to help fix overfitting)
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
#Hidden layer 2
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
#Output layer
model.add(Dense(1, activation='sigmoid'))
#Add learning rate to optimizer to reduce overfitting
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
#Compile model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# In[73]:


#View model summary
model.summary()


# In[74]:


#Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=400, batch_size=32)


# In[75]:


#Evaluate the matrix
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# In[78]:


#Plot training and validation loss
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()

#Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Value')
plt.legend()

plt.tight_layout()
plt.show()


# During exploratory data analysis and feature engineering, the variables 'credit.policy' and 'fico' were dropped because they had the lowest correlation with the target variable not.fully.paid. The variable 'log.annual.inc' was dropped because it had a high correlation with 'installment' and a relatively low correlation with the target. The model was initially built with one hidden layer, an unspecified learning rate and no regularizer attached. The results heavily suggested overfitting, and the model had a low initial validation accuracy and high validation loss. A regularizer was added to the hidden layer to reduce overfitting and multiple values were tested to find the best result. The optimizer was then given a learning rate, and 0.1 was found to be the best value. After the validation results were better but still somewhat unsatisfying, another two hidden layers were added, but only two hidden layers were kept because having three yielded unfavorable results.

# In[ ]:




