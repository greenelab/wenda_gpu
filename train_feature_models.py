# For each feature, wenda trains a model on the rest of the source data
# (e.g. all other features), and tests how well that model is able to predict
# the observed feature values in the target data. Given the necessity of
# running these steps in batches, it is most efficient to do the training and
# testing in the same script.

import os
import argparse
import gpytorch
import numpy as np
import pandas as pd
import torch
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prefix', help="Dataset identifier used to name subfolders.")
parser.add_argument('-s', '--start', type=int, default=0, help="What feature number to start training on. This is needed for batch training.")
parser.add_argument('-r', '--range', type=int, default=100, help="How many feature models to train. This is needed for batch training.")
parser.add_argument('--separator', default="\t", help="Separator used in input data files.")
args = parser.parse_args()

# Set torch to run with float32 instead of float64, which exponentially
# increases speed with neglibile decrease in precision.
dtype = torch.float32
device = 'cuda'

# Load data
source_file = os.path.join("data", args.prefix, "source_data.tsv")
source_table = pd.read_csv(source_file, sep=args.separator, header=None)
source_matrix = np.asfortranarray(source_table.values)

target_file = os.path.join("data", args.prefix, "target_data.tsv")
target_table = pd.read_csv(target_file, sep=args.separator, header=None)
target_matrix = np.asfortranarray(target_table.values)

# Normalize based on source data, with some noise to avoid dividing by 0
epsilon = 1e-6
means = np.mean(source_matrix, axis=0)
stds = np.std(source_matrix, axis=0) + epsilon

normed = (source_matrix - means) / stds
normed_source_matrix = normed

normed = (target_matrix - means) / stds
normed_target_matrix = normed

# Make directory to store feature models
output_dir = os.path.join("feature_models", args.prefix)
os.makedirs(output_dir, exist_ok=True)

# Make directory to store confidence scores from target data
conf_dir = os.path.join("confidences", args.prefix)
os.makedirs(conf_dir, exist_ok=True)

first_model = args.start

for i in range(args.range):
    feature_number = first_model + i
    try:

        # If confidences have already been calculated, skip feature
        conf_file = os.path.join(conf_dir, "model_%s_confidence.txt" % feature_number)
        if os.path.isfile(conf_file):
            continue

        # Split out feature to predict using all other features
        train_y = torch.from_numpy(normed_source_matrix[:, feature_number])
        train_y = train_y.to(device=device, dtype=dtype)
        train_x = torch.from_numpy(np.delete(normed_source_matrix, feature_number, 1)).squeeze(-1)
        train_x = train_x.to(device=device, dtype=dtype)
        test_y = torch.from_numpy(normed_target_matrix[:, feature_number])
        test_y = test_y.to(device=device, dtype=dtype)
        test_x = torch.from_numpy(np.delete(normed_target_matrix, feature_number, 1)).squeeze(-1)
        test_x = test_x.to(device=device, dtype=dtype)

        # Train model if it has not been previously generated
        modelfile = os.path.join(output_dir, "model_%s.pth" % feature_number)
        if os.path.isfile(modelfile) is False:

            # Initialize model and likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
            model = utils.ExactGPModel(train_x, train_y, likelihood).to(device=device, dtype=dtype)

            model.train()
            likelihood.train()

            with gpytorch.settings.max_cholesky_size(100000), gpytorch.settings.cholesky_jitter(1e-5):
                model, likelihood = utils.train_model_bfgs(
                    model, likelihood, train_x, train_y, learning_rate=1., training_iter=15
                    )

            model.eval()
            likelihood.eval()

            # Save feature model
            torch.save(model.state_dict(),os.path.join(output_dir, "model_%s.pth" % feature_number))

        # If model was previously generated, load for confidence score calculation
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
            state_dict = torch.load(modelfile)
            model = utils.ExactGPModel(train_x, train_y, likelihood).to(device=device, dtype=dtype)
            model.load_state_dict(state_dict)

            model.eval()
            likelihood.eval()

        # Write out confidence scores
        mean, var, conf = utils.get_confidence(model, likelihood, test_x, test_y)
        np.savetxt(conf_file, conf, fmt='%.10f')

    # If the model is unable to be trained, we consider it safe to assign that
    # feature a confidence score of 0 for all the target data. This will
    # ultimately penaltze the untrainable model's feature in the elastic net.
    except gpytorch.utils.errors.NotPSDError:
        conf = np.zeros(test_x.shape[0])
        np.savetxt(conf_file, conf, fmt='%i')
