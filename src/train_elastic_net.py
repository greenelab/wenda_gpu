"""
Author: Ariel Hippen
Date Created: 4 August 2022

Takes the confidence scores from the feature models and uses them as weighted
penalties for the ultimate elastic (or logistic) net task, training the
source data on the source labels. This script will train several models, a
vanilla (unweighted) elastic net and with a variety of penalization amounts
based on confidence score.
"""

import os
import glmnet
import numpy as np
import pandas as pd
import utils

def _aggregate_confidences(source_matrix,
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
                      prefix = "original",
                      confidence_path = "confidences",
                      elastic_net_path = "output",
                      alpha = 0.8,
                      n_splits = 10,
                      k_values = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 25, 35],
                      logistic = False,
                      horvath = False,
                      export_coef = False,
                      verbose = True):
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
    scores = _aggregate_confidences(source_matrix, prefix, confidence_path)

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
