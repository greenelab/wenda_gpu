#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import gpytorch
import pandas as pd
import numpy as np
from scipy.stats import norm


# In[2]:


# Load source data
source = pd.read_csv("source_data.csv", sep=" ")
source = source.to_numpy()


# In[3]:


# Normalize data

epsilon=1e-6
means = np.mean(source, axis=0)
stds = np.std(source, axis=0) + epsilon

normed = (source - means) / stds
source = normed


# In[4]:


# Transpose data so features/genes are columns and samples are rows as gpytorch expects

source = np.transpose(source)
print(source.shape)


# In[5]:


# Set parameters and make output dir

learning_rate = 1 #Can't remember where this number came from, but I've been hardcoding it for a while
# let me know if you think that's a problem source
gene_number = 2

feature_model_format = 'model_{0:05d}'
output_dir = os.path.join("gpytorch_feature_models", feature_model_format.format(gene_number))
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)


# In[6]:


# Split out feature to predict using all other features

y = source[:, gene_number]
x = np.delete(source, gene_number, 1)


# In[7]:


# Convert to torch objects

x_train_tensor = torch.Tensor(x)
y_train_tensor = torch.Tensor(y)


# In[8]:


# Initialize model and likelihood

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()
  
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train_tensor, y_train_tensor, likelihood)   


# In[9]:


# Put everything on the GPU

x_train_tensor = x_train_tensor.cuda()
y_train_tensor = y_train_tensor.cuda()
model = model.cuda()
likelihood = likelihood.cuda()


# In[10]:


model.train()
likelihood.train()


# In[11]:


def train_model(model, likelihood, x, y, learning_rate,
                training_iter=40):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
    
    return model, likelihood


# In[12]:


model, likelihood = train_model(model, likelihood, x_train_tensor,
                                         y_train_tensor, learning_rate)


# In[13]:


model.eval()
likelihood.eval()


# In[14]:


model.state_dict()


# In[15]:


# Save state dict

torch.save(model.state_dict(),os.path.join(output_dir, "state_dict.pth"))


# In[16]:


# Load target data, confidence scores are calculated based on
# how well the GP models predict on the target distribution

target = pd.read_csv("target_data.csv", sep = " ")
target = target.to_numpy()


# In[17]:


# Normalize data

epsilon=1e-6
means = np.mean(target, axis=0)
stds = np.std(target, axis=0) + epsilon

normed = (target - means) / stds
target = normed


# In[18]:


# Transpose data so features/genes are columns and samples are rows as gpytorch expects

target = np.transpose(target)
print(target.shape)


# In[19]:


# Split out gene to be predicted

y_test = target[:, gene_number]
y_test_tensor = torch.Tensor(y_test).cuda()
x_test = np.delete(target, gene_number, 1)
x_test_tensor = torch.Tensor(x_test).cuda()


# In[20]:


# Get confidence score based on CDF of true target value on GP model
def getConfidence(model, x, y):
    with gpytorch.settings.fast_pred_var():
        f_preds = model(x)
    mu = f_preds.mean
    sigma_sq = f_preds.variance
    sigma_sq = torch.sqrt(sigma_sq)
    res_normed = (y - mu) / sigma_sq
    res_normed = res_normed.cpu().detach().numpy()
    confidences = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))
    mu = mu.cpu().detach().numpy()
    sigma_sq = sigma_sq.cpu().detach().numpy()
    return mu, sigma_sq, confidences


# In[21]:


# Write out confidence scores and predicted means and variances on target data
mean, var, conf = getConfidence(model, x_test_tensor, y_test_tensor)
conf_file = os.path.join(output_dir, "confidences.txt")
np.savetxt(conf_file, conf, fmt='%.10f')
mean_file = os.path.join(output_dir, "predicted_means.txt")
np.savetxt(mean_file, mean, fmt='%.5f')
var_file = os.path.join(output_dir, "predicted_variances.txt")
np.savetxt(var_file, var)

