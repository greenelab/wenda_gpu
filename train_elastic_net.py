# Takes the confidence scores from the feature models and uses them as weighted
# penalties for the ultimate elastic (or logistic) net task, training the
# source data on the source labels. This script will train several models, a
# vanilla (unweighted) elastic net and with a variety of penalization amounts
# based on confidence score.

import os
import utils
import glmnet
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prefix", help="Dataset identifier used to name subfolders.")
parser.add_argument("-s", "--splits", default=10, help="Number of splits, minimum 3.")
parser.add_argument("-d", "--delimiter", default="\t", help="Separator used in input data files.")
parser.add_argument("--data_path", default="data", help="Location of input data.")
parser.add_argument("--confidence_path", default="confidences", help="Where confidence scores for each model will be written to.")
parser.add_argument("--elastic_net_path", default="output", help="Where model predictions will be written to.")
parser.add_argument("--horvath", action="store_true", help="Run Horvath transformation for age labels")
parser.add_argument("--logistic", action="store_true",help="For binary data, create logistic net model instead of elastic net")
args = parser.parse_args()


data_dir = os.path.join("data", args.prefix)

# Check if there's an aggregated confidence scores file,
# and if not create it
confidence_file = os.path.join(data_dir, "confidences.tsv")
if os.path.isfile(confidence_file) is False:

    # Get all confidence score files
    filenumber = len(os.listdir("confidences/%s/" % args.prefix))
    print(filenumber)

    # Concatenate into one confidence file and save
    models = []
    for i in range(filenumber):
        filename = "confidences/%s/model_%d_confidence.txt" % (args.prefix, i)
        model = np.loadtxt(filename)
        model.shape = (model.shape[0], 1)
        models.append(model)

    confidences = np.concatenate(models, axis=1)
    np.savetxt("data/%s/confidences.tsv" % args.prefix, confidences,
               delimiter=args.delimiter, fmt="%.5f")
else:
    confidences = pd.read_csv(confidence_file, sep=args.delimiter, header=None)
    confidences = np.asfortranarray(confidences)
scores = np.mean(confidences, axis=0)


# Load source and target data
data_dir = os.path.join("data", args.prefix)

source_file = os.path.join(data_dir, "source_data.tsv")
source_table = pd.read_csv(source_file, sep=args.delimiter, header=None)
source_matrix = np.asfortranarray(source_table.values)

target_file = os.path.join(data_dir, "target_data.tsv")
target_table = pd.read_csv(target_file, sep=args.delimiter, header=None)
target_matrix = np.asfortranarray(target_table.values)


# Normalize data based on source data, with noise to avoid dividing by 0
epsilon = 1e-6
means = np.mean(source_matrix, axis=0)
stds = np.std(source_matrix, axis=0) + epsilon

normed = (source_matrix - means) / stds
normed_source_matrix = normed

normed = (target_matrix - means) / stds
normed_target_matrix = normed

print(normed_source_matrix.shape)


# Load source data labels
y_file = os.path.join(data_dir, "source_y.tsv")
y_table = pd.read_csv(y_file)
y_matrix = np.asfortranarray(y_table)

if args.horvath:
    normed_y_matrix = utils.age_transform(y_matrix)
else:
    normed_y_matrix = y_matrix

print(normed_y_matrix.shape)


# Elastic net parameters. Alpha = 0.8 is set as a design decision.
# Higher values of k penalize features with low confidence scores.
# k=0 is equivalent to a vanilla elastic net.
alpha = 0.8
n_splits = args.splits
k_values = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 25, 35]


# Run elastic net
for k in k_values:
    print("k_wnet =", k, flush=True)
    if args.logistic:
        model = glmnet.LogitNet(alpha=alpha, n_splits=n_splits)
    else:
        model = glmnet.ElasticNet(alpha=alpha, n_splits=n_splits)
    if k == 0:
        model = model.fit(normed_source_matrix, normed_y_matrix)
    else:
        weights = utils.confidence_to_weights(scores, k)
        model = model.fit(normed_source_matrix, normed_y_matrix, relative_penalties=weights)
    try:
        target_y = model.predict(normed_target_matrix)
    except ValueError:
        continue

    # Save predicted target labels to file
    output_dir = os.path.join("output/%s/k_{0:02d}".format(k) % args.prefix)
    os.makedirs(output_dir, exist_ok=True)
    target_file = os.path.join(output_dir, "target_predictions.txt")
    if args.horvath:
        age_y = utils.age_back_transform(target_y)
        np.savetxt(target_file, age_y)
    else:
        np.savetxt(target_file, target_y, fmt="%i")
    # For logistic regression, save probability of each label
    if args.logistic:
        prob = model.predict_proba(normed_target_matrix)
        prob_file = os.path.join(output_dir, "target_probabilities.txt")
        np.savetxt(prob_file, prob, fmt="%.5e")
