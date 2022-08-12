"""
Author: Ariel Hippen
Date Created: 4 August 2022

For each feature, wenda trains a model on the rest of the source data
(ie all other features), and generates a confidence score based on how
well that model is able to predict the observed feature values in the
target data. These confidence values are used as weighted penalties for
the ultimate elastic (or logistic) net task, training the source data
on the source labels. This script will train several models, a vanilla
(unweighted) elastic net and with a variety of penalization amounts
based on confidence score.
"""

import os
import gc
import glmnet
import gpytorch
import numpy as np
import pandas as pd
import torch
from wenda_gpu import utils

def load_data(prefix="original",
              data_path="data",
              delimiter="\t"):
    """ Load source and target data files. These should be a
    tab-delimited matrix where each row is a sample and each
    column is a feature.
    """
    source_file = os.path.join(data_path, prefix, "source_data.tsv")
    source_table = pd.read_csv(source_file, sep=delimiter, header=None)
    source_matrix = np.asfortranarray(source_table.values)
    target_file = os.path.join(data_path, prefix, "target_data.tsv")
    target_table = pd.read_csv(target_file, sep=delimiter, header=None)
    target_matrix = np.asfortranarray(target_table.values)
    return source_matrix, target_matrix


def train_feature_models(source_matrix,
                         target_matrix,
                         prefix="original",
                         batch_size=100,
                         feature_model_path="feature_models",
                         confidence_path="confidences",
                         dtype=torch.float32,
                         device='cuda',
                         verbose=True):
    """ Train Gaussian process models for each feature in the source data, and
    generate confidence scores based on how well they predict the target data.
    Arguments:
        source_matrix:      the source data. Expects a numpy fortran array. Should
                            have been z-scored using normalizeData().
        target_matrix:      the target data. Expects a numpy fortran array. Should
                            have been z-scored using normalizeData().
        prefix:             a dataset identifier used to name subfolders.
        batch_size:         size of batches run (default is 100). On large datasets,
                            this may need to be adjusted to prevent memory overflow errors.
        feature_model_path: the directory to store feature models, with the var
                            'prefix' being used as a subdirectory.
        confidence_path:    the directory to store feature model confidence
                            scores, with prefix used as a subdirectory.
        dtype:              the level of precision for GPyTorch to use. torch.float32 is
                            used as a default; torch.float64 can be used but is much slower.
        device:             whether to run models on GPU ('cuda') or CPU ('cpu').
        verbose:            prints intermediate messages when each batch of models runs.
    Returns:
        Nothing, writes to files in feature_model_path and confidence_path
    """
    # Set up directories for storing feature models and confidences
    feature_dir, confidence_dir = utils.organize_directory_structure(prefix=prefix,
            feature_model_path=feature_model_path, confidence_path=confidence_path)
    # Calculate number of batches to run
    source_features = source_matrix.shape[1]
    batches = int(source_features / batch_size)
    # If number of features isn't divisible by batch size, add one batch
    if source_features % batch_size != 0:
        batches += 1
    for j in range(batches):
        start = j * batch_size
        stop = start + batch_size - 1
        if verbose:
            print("Training models %d to %d..." % (start, stop))
        _train_feature_model_batch(source_matrix, target_matrix, feature_dir, confidence_dir,
                start=start, batch_size=batch_size, dtype=dtype, device=device)
        gc.collect()
        torch.cuda.empty_cache()


def normalize_data(raw_source_matrix,
                   raw_target_matrix):
    """ Normalize data features based on source data,
    with some noise to avoid dividing by 0.
    Arguments:
        raw_source_matrix:    original source count matrix
        raw_target_matrix:    original target count matrix
    Returns:
        normed_source_matrix: z-scored source matrix
        normed_target_matrix: z-scored target matrix
    """
    epsilon = 1e-6
    means = np.mean(raw_source_matrix, axis=0)
    stds = np.std(raw_source_matrix, axis=0) + epsilon

    normed_source_matrix = (raw_source_matrix - means) / stds
    normed_target_matrix = (raw_target_matrix - means) / stds

    return normed_source_matrix, normed_target_matrix


def _train_feature_model_batch(source_matrix,
                               target_matrix,
                               feature_dir,
                               confidence_dir,
                               start=0,
                               batch_size=100,
                               dtype=torch.float32,
                               device='cuda'):
    """ Run a batch worth of feature model training and generate
    confidence scores.
    Arguments:
        source_matrix:  the source data. Expects a numpy fortran array. Should
                        have been z-scored using normalizeData().
        target_matrix:  the target data. Expects a numpy fortran array. Should
                        have been z-scored using normalizeData().
        feature_dir:    the dataset-specific subdirectory to write feature models to.
        confidence_dir: the dataset-specific subdirectory to write confidences to.
        start:          the index of the first feature model in the batch.
        batch_size:     number of feature models to run in one batch.
        dtype:          the level of precision for GPyTorch to use. torch.float32 is
                        used as a default; torch.float64 can be used but is much slower.
        device:         whether to run models on GPU ('cuda') or CPU ('cpu').
    Returns:
        Nothing, writes to files in feature_dir and confidence_dir.
    """
    first_model = start
    total_features = source_matrix.shape[1]

    for i in range(batch_size):
        # Prevent out of range error if start + range > total number of features
        feature_number = first_model + i
        if feature_number >= total_features:
            break
        try:

            # If confidences have already been calculated, skip feature
            conf_file = os.path.join(confidence_dir, "model_%s_confidence.txt" % feature_number)
            if os.path.isfile(conf_file):
                continue

            # Split out feature to predict using all other features
            train_y = torch.from_numpy(source_matrix[:, feature_number])
            train_y = train_y.to(device=device, dtype=dtype)
            train_x = torch.from_numpy(np.delete(source_matrix, feature_number, 1)).squeeze(-1)
            train_x = train_x.to(device=device, dtype=dtype)
            test_y = torch.from_numpy(target_matrix[:, feature_number])
            test_y = test_y.to(device=device, dtype=dtype)
            test_x = torch.from_numpy(np.delete(target_matrix, feature_number, 1)).squeeze(-1)
            test_x = test_x.to(device=device, dtype=dtype)

            # Train model if it has not been previously generated
            modelfile = os.path.join(feature_dir, "model_%s.pth" % feature_number)
            if os.path.isfile(modelfile) is False:

                # Initialize model and likelihood
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood = likelihood.to(device=device, dtype=dtype)
                model = utils.ExactGPModel(train_x, train_y, likelihood)
                model = model.to(device=device, dtype=dtype)

                model.train()
                likelihood.train()

                with gpytorch.settings.max_cholesky_size(100000), gpytorch.settings.cholesky_jitter(1e-5):
                    model, likelihood = utils.train_model_bfgs(
                        model, likelihood, train_x, train_y, learning_rate=1., training_iter=15
                        )

                model.eval()
                likelihood.eval()

                # Save feature model
                torch.save(model.state_dict(),
                        os.path.join(feature_dir, "model_%s.pth" % feature_number))

            # If model was previously generated, load for confidence score calculation
            else:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood = likelihood.to(device=device, dtype=dtype)
                state_dict = torch.load(modelfile)
                model = utils.ExactGPModel(train_x, train_y, likelihood)
                model = model.to(device=device, dtype=dtype)
                model.load_state_dict(state_dict)

                model.eval()
                likelihood.eval()

            # Write out confidence scores
            confidences = utils.get_confidence(model, likelihood, test_x, test_y)
            np.savetxt(conf_file, confidences, fmt='%.10f')

        # If the model is unable to be trained, we consider it safe to assign that
        # feature a confidence score of 0 for all the target data. This will
        # ultimately penaltze the untrainable model's feature in the elastic net.
        except gpytorch.utils.errors.NotPSDError:
            confidences = np.zeros(test_x.shape[0])
            np.savetxt(conf_file, confidences, fmt='%i')


def load_labels(prefix="original",
                data_path="data",
                horvath=False):
    """ Load source data labels. This should be a file called
    'source_y.tsv' with one column, where each row is a sample.
    Arguments:
        prefix:       a dataset identifier used to name subfolders.
        data_path:    the directory where data is stored, with the var
                      'prefix' being used as a subdirectory.
        horvath:      run horvath transformation on age labels.
    Returns:
        label_matrix: an array of source labels (transformed if needed)
    """
    data_dir = os.path.join(data_path, prefix)
    label_file = os.path.join(data_dir, "source_y.tsv")
    label_table = pd.read_csv(label_file, header=None)
    label_matrix = np.asfortranarray(label_table)

    if horvath:
        label_matrix = utils.age_transform(label_matrix)

    return label_matrix


def train_elastic_net(source_matrix,
                      source_labels,
                      target_matrix,
                      prefix="original",
                      confidence_path="confidences",
                      elastic_net_path="output",
                      alpha=0.8,
                      n_splits=10,
                      k_values=[0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 25, 35],
                      logistic=False,
                      horvath=False,
                      export_coef=False,
                      verbose=True):
    """ Trains several elastic nets on the source data and labels across a
    range of hyperparameters.
    Arguments:
        source_matrix:    the source data. Expects a numpy fortran array. Should
                          have been z-scored using normalizeData().
        source_labels:    the source labels. Expects a numpy fortran array.
        target_matrix:    the target data. Expects a numpy fortran array. Should
                          have been z-scored using normalizeData().
        prefix:           a dataset identifier used to name subfolders.
        confidence_path:  the directory to store feature model confidence
                          scores, with prefix used as a subdirectory.
        elastic_net_path: the directory to store elastic net results, with
                          prefix used as a subdirectory.
        alpha:            weighting of lasso vs. ridge regression,
                          should be between 0 and 1.
        n_splits:         number of cross-validation splits, minimum 3.
        k_values:         weight of penalization to assign to low confidence
                          scores. This should be a range of non-negative
                          values, with 0 being a regular elastic net.
        logistic:         for binary data, create a logistic net.
        horvath:          run horvath transformation on age labels.
        export_coef:      write model coefficients to file called weights.tsv.
        verbose:          prints status message when each value of k is tested.
    Returns:
       Nothing, writes to files in elastic_net_path
    """
    scores = utils.aggregate_confidences(source_matrix, prefix, confidence_path)

    elastic_net_dir = os.path.join(elastic_net_path, prefix)

    for k in k_values:
        k_output_dir = os.path.join(elastic_net_dir, "k_{0:02d}".format(k))
        os.makedirs(k_output_dir, exist_ok=True)

        if verbose:
            print("k_wnet =", k, flush=True)
        if logistic:
            model = glmnet.LogitNet(alpha=alpha, n_splits=n_splits)
        else:
            model = glmnet.ElasticNet(alpha=alpha, n_splits=n_splits)

        # Train model on source data and labels
        if k == 0:
            model = model.fit(source_matrix, source_labels)
        else:
            weights = utils.confidence_to_weights(scores, k)
            model = model.fit(source_matrix, source_labels, relative_penalties=weights)

        if export_coef:
            weight_file = os.path.join(k_output_dir, "weights.txt")
            np.savetxt(weight_file, model.coef_)

        # Predict on target data
        try:
            target_y = model.predict(target_matrix)
        except ValueError:
            continue

        # Save predicted target labels to file
        target_file = os.path.join(k_output_dir, "target_predictions.txt")
        if horvath:
            age_y = utils.age_back_transform(target_y)
            np.savetxt(target_file, age_y)
        else:
            np.savetxt(target_file, target_y, fmt="%i")
        # For logistic regression, save probability of each label
        if logistic:
            prob = model.predict_proba(target_matrix)
            prob_file = os.path.join(k_output_dir, "target_probabilities.txt")
            np.savetxt(prob_file, prob, fmt="%.5e")
