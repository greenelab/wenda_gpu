#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gpytorch
import math
import numpy as np
import pandas as pd
import torch

from LBFGS import FullBatchLBFGS
from scipy.stats import norm


# In[2]:


# Try running 300 models, one at a time
ix_start = 0
ix_range = 300

# torch.float64 to run in double precision, torch.float32 for single
dtype = torch.float32
device = 'cuda'


# In[3]:


# Load data
source_table = pd.read_csv("source_data.csv", sep=" ")
source_matrix = np.asfortranarray(source_table.values.T)

target_table = pd.read_csv("target_data.csv", sep=" ")
target_matrix = np.asfortranarray(target_table.values.T)

# Normalize based on source data
epsilon = 1e-6
means = np.mean(source_matrix, axis=0)
stds = np.std(source_matrix, axis=0) + epsilon

normed = (source_matrix - means) / stds
normed_source_matrix = normed

normed = (target_matrix - means) / stds
normed_target_matrix = normed


# In[4]:


# Make feature model directory
output_dir = "feature_models"
os.makedirs(output_dir, exist_ok=True)

# Make confidence directory
conf_dir = "confidences"
os.makedirs(conf_dir, exist_ok=True)


# In[5]:


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()
        
        # Initialize variance to 1/d so that inner products between data points are ~ 1.
        # Unscaled inner products in train_x are so large that we lose precision.
        self.covar_module.variance = 1. / train_x.size(-1)
  
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model_bfgs(model, likelihood, x, y, learning_rate,
                training_iter=10):
    lbfgs = FullBatchLBFGS(model.parameters(), lr=learning_rate)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        model.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        return loss

    loss = closure()
    loss.backward()

    for i in range(training_iter):
        options = {"closure": closure, "current_loss": loss, "max_ls": 10}
        loss, _, lr, _, F_eval, G_eval, _, fail = lbfgs.step(options)

        if fail:
            break

    return model, likelihood


def getConfidence(model, likelihood, x, y):
    with gpytorch.settings.fast_pred_var():
        f_preds = likelihood(model(x))
    mu = f_preds.mean
    sigma_sq = f_preds.variance
    sigma_sq = torch.sqrt(sigma_sq)
    res_normed = (y - mu) / sigma_sq
    res_normed = res_normed.cpu().detach().numpy()
    confidences = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))
    mu = mu.cpu().detach().numpy()
    sigma_sq = sigma_sq.cpu().detach().numpy()
    return mu, sigma_sq ** 2, confidences


# In[6]:


for i in range(ix_range):
    gene_number = ix_start + i
    print(gene_number)

    # If confidences have already been calculated, skip
    conf_file = os.path.join(conf_dir, "model_%s_confidence.txt" % gene_number)
    if os.path.isfile(conf_file):
        continue

    # Split out feature to predict using all other features
    train_y = torch.from_numpy(normed_source_matrix[:, gene_number])
    train_x = torch.from_numpy(np.delete(normed_source_matrix, gene_number, 1)).squeeze(-1)
    test_y = torch.from_numpy(normed_target_matrix[:, gene_number])
    test_x = torch.from_numpy(np.delete(normed_target_matrix, gene_number, 1)).squeeze(-1)

    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)
    test_x = test_x.to(device=device, dtype=dtype)
    test_y = test_y.to(device=device, dtype=dtype)

    # Train model if it has not been previously generated
    modelfile = os.path.join(output_dir, "model_%s.pth" % gene_number)
    if os.path.isfile(modelfile) is False:
        
        # Initialize model and likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        model = ExactGPModel(train_x, train_y, likelihood).to(device=device, dtype=dtype)

        model.train()
        likelihood.train()

        # TODO: consider adding a logfile back in with time to run, likelihood noise, and kernel variance
        with gpytorch.settings.max_cholesky_size(100000), gpytorch.settings.cholesky_jitter(1e-5):
            model, likelihood = train_model_bfgs(
                model, likelihood, train_x, train_y, learning_rate=1., training_iter=15
                )

        model.eval()
        likelihood.eval()

        # Save feature model
        torch.save(model.state_dict(),os.path.join(output_dir, "model_%s.pth" % gene_number))
    
    # Else, load previously generated model for confidence score calculation
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        state_dict = torch.load(modelfile)
        model = ExactGPModel(train_x, train_y, likelihood).to(device=device, dtype=dtype)
        model.load_state_dict(state_dict)

        model.eval()
        likelihood.eval()

    # Write out confidence scores
    mean, var, conf = getConfidence(model, likelihood, test_x, test_y)
    np.savetxt(conf_file, conf, fmt='%.10f')


# In[ ]:




