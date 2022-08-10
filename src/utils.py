""" Author: Ariel Hippen
Date Created: 4 August 2022
"""

import gpytorch
import torch
import numpy as np

from scipy.stats import norm
from LBFGS import FullBatchLBFGS


class ExactGPModel(gpytorch.models.ExactGP):
    """ A Gaussian process latent function used for exact inference.
    Parameters:
        train_x: the source data for all other features
        train_y: the source data for the feature of interest.
        likelihood: for exact inference, must be a Gaussian likelihood.
    Methods:
        The forward function describes how to compute the prior distribution.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()

        # Initialize variance to 1/d so inner products between data points are ~ 1.
        # Unscaled inner products in train_x are so large that we lose precision.
        self.covar_module.variance = 1. / train_x.size(-1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model_bfgs(model, likelihood, x, y, learning_rate,
                     training_iter=10):
    """ Run training of GPyTorch models.
    Arguments:
        model: the GPyTorch model to be trained.
        likelihood: the GPyTorch GaussianLikelihood.
        x: source data for all other features.
        y: source data for the feature of interest.
        learning_rate: determines how fast model will converge.
        trainer_iter: number of epochs to run training for.
    Returns:
        model: the trained GPyTorch model
        likelihood: the GPyTorch GaussianLikelihood.
    """
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


def confidence_to_weights(x, k):
    """ Convert mean confidence scores into a weight matrix.
    Arguments:
        x: an array of mean confidence scores.
        k: a hyperparameter governing amount of penalization
           for poor transferability. A higher k results in
           greater penalization of low confidence scores.
    """
    return np.power(1-x, k)


def get_confidence(model, likelihood, x, y):
    """ Calculate likelihood of target data values based
    on the predicted feature model's distribution.
    Arguments:
        model: the trained gpytorch model
        likelihood: the gpytorch GaussianLikelihood
        x: target data for all other features
        y: target data for the feature of interest
    Returns:
        confidences: vector of confidence scores for
                     each sample in target data.
    """
    with gpytorch.settings.fast_pred_var():
        f_preds = likelihood(model(x))
    mu = f_preds.mean
    sigma_sq = f_preds.variance
    sigma_sq = torch.sqrt(sigma_sq)
    res_normed = (y - mu) / sigma_sq
    res_normed = res_normed.cpu().detach().numpy()
    confidences = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))
    return confidences


def age_transform(age, adult_age=20):
    """ Implements the Horvath transformation for age data,
    which helps adjust for greater biological changes
    in the first few years of age than in later life.
    Arguments:
        age: a vector of sample ages in years.
        adult_age: age at which to not apply log transformation
    Returns:
        horvath: a vector of Horvath-transformed ages.
    """
    age = (age+1)/(1+adult_age)
    horvath = np.where(age <= 1, np.log(age), age-1)
    return horvath


def age_back_transform(trans_age, adult_age=20):
    """ Reverses the Horvath transformation for age data.
    Arguments:
        trans_age: vector of Horvath-transformed ages.
        adult_age: age at which log transformation isn't applied.
    Returns:
        age: a vector of sample ages in years.
    """
    age = np.where(trans_age < 0,
                  (1+adult_age)*np.exp(trans_age)-1,
                  (1+adult_age)*trans_age+adult_age)
    return age
