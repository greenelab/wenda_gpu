#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import GPy
from IPython.display import display
from scipy.stats import norm


# In[2]:


# Load data to train model on

source_table = pd.read_csv("source_data.csv", sep=" ")
source_matrix = np.asfortranarray(source_table.values.T) 
#Note: This was originally source_table.as_matrix().T but that function is now deprecated 


# In[3]:


# Normalize data

epsilon = 1e-6
means = np.mean(source_matrix, axis=0)
stds = np.std(source_matrix, axis=0) + epsilon

normed = (source_matrix - means) / stds
normed_source_matrix = normed


# In[4]:


# Set parameters and make output dir

gene_number = 1

feature_model_format = 'model_{0:05d}'
output_dir = os.path.join("gpy_feature_models", feature_model_format.format(gene_number))
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)


# In[5]:


kernel = GPy.kern.Linear(input_dim=source_table.shape[0]-1)


# In[6]:


# Split out feature to predict using all other features

is_i = np.in1d(np.arange(normed_source_matrix.shape[1]), gene_number)
data_x_train = normed_source_matrix[:, ~is_i]
data_y_train = normed_source_matrix[:, is_i]


# In[7]:


# Fit model

model = GPy.models.GPRegression(data_x_train, data_y_train, kernel=kernel.copy())
model.optimize()


# In[8]:


np.savetxt(os.path.join(output_dir, "param_array.txt"),  model.param_array)


# In[9]:


display(model)


# In[10]:


# Load target data
target_table = pd.read_csv("target_data.csv", sep = " ")
target_matrix = np.asfortranarray(target_table.values.T)


# In[11]:


# Normalize data

epsilon = 1e-6
means = np.mean(target_matrix, axis=0)
stds = np.std(target_matrix, axis=0) + epsilon

normed = (target_matrix - means) / stds
normed_target_matrix = normed


# In[12]:


# Split feature out of target data

is_feature = np.in1d(np.arange(normed_target_matrix.shape[1]), gene_number)
data_x_test = normed_source_matrix[:, ~is_feature]
data_y_test = normed_source_matrix[:, is_feature]


# In[13]:


# Calculate confidence score using target data
mu, sigma_sq = model.predict(data_x_test)
res_normed = (data_y_test - mu) / np.sqrt(sigma_sq)
confidences = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))


# In[14]:


np.savetxt(os.path.join(output_dir, "confidences.txt"), confidences, fmt='%.10f')
np.savetxt(os.path.join(output_dir, "predicted_means.txt"), mu, fmt='%.5f')
np.savetxt(os.path.join(output_dir, "predicted_variances.txt"), sigma_sq)

