""" Author: Ariel Hippen
Date Created: 4 August 2022
"""

import os
import torch
import gpytorch
import numpy as np
import pandas as pd

from scipy.stats import norm
from wenda_gpu.LBFGS import FullBatchLBFGS


def organize_directory_structure(prefix="original",
                                 feature_model_path="feature_models",
                                 confidence_path="confidences"):
    """ Create feature model and confidence directories if they don't exist
    and creates dataset-specific subdirectories if they don't exist.
    Arguments:
        prefix:             a dataset identifier used to name subfolders.
        feature_model_path: the directory to store feature models, with the var
                            'prefix' being used as a subdirectory.
        confidence_path:    the directory to store feature model confidence
                            scores, with prefix used as a subdirectory.
    Returns:
        feature_dir:        the dataset-specific subdirectory for feature models.
        confidence_dir:     the dataset-specific subdirectory for confidence scores.
    """
    os.makedirs(feature_model_path, exist_ok=True)
    feature_dir = os.path.join(feature_model_path, prefix)
    os.makedirs(feature_dir, exist_ok=True)

    os.makedirs(confidence_path, exist_ok=True)
    confidence_dir = os.path.join(confidence_path, prefix)
    os.makedirs(confidence_dir, exist_ok=True)

    return feature_dir, confidence_dir


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


def aggregate_confidences(source_matrix,
                          prefix="original",
                          confidence_path="confidences"):
    """ The confidence scores for each feature model are stored in
    a separate file. This script concatenates them into one file.
    Arguments:
        source_matrix:   the normalized source data.
        prefix:          a dataset identifier used to name subfolders.
        confidence_path: directory used to store feature model confidence
                         scores, with the var prefix as a subdirectory.
    Returns:
        scores:          a vector of mean confidence scores in target data
                         for each feature model.
    """
    # Check if there's an aggregated confidence scores file,
    # and if not create it
    confidence_dir = os.path.join(confidence_path, prefix)
    confidence_file = os.path.join(confidence_dir, "confidences.tsv")
    if os.path.isfile(confidence_file) is False:

        # Check all confidence score files have been run
        filenumber = len(os.listdir(confidence_dir))
        expected_files = source_matrix.shape[1]
        if filenumber != expected_files:
            raise Exception("Expected %d confidence files and found %d. \
                    Confirm all feature models have been run." % (expected_files, filenumber))

        # Concatenate into one confidence file and save
        features = []
        for i in range(filenumber):
            filename = os.path.join(confidence_dir, "model_%d_confidence.txt" % i)
            feature_scores = np.loadtxt(filename)
            feature_scores.shape = (feature_scores.shape[0], 1)
            features.append(feature_scores)

        confidences = np.concatenate(features, axis=1)
        np.savetxt(confidence_file, confidences,
                   delimiter="\t", fmt="%.5f")
    else:
        confidences = pd.read_csv(confidence_file, sep="\t", header=None)
        confidences = np.asfortranarray(confidences)

    scores = np.mean(confidences, axis=0)
    return scores


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
    """ Implements the Horvath transformation (DOI 10.1186/gb-2013-14-10-r115)
    for age data, which helps adjust for greater methylation changes in the 
    first few years of age than in later life.
    Arguments:
        age: a vector of sample ages in years.
        adult_age: age cutoff above which to not apply log transformation
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
        adult_age: age cutoff above which which log transformation isn't applied.
    Returns:
        age: a vector of sample ages in years.
    """
    age = np.where(trans_age < 0,
                  (1+adult_age)*np.exp(trans_age)-1,
                  (1+adult_age)*trans_age+adult_age)
    return age
