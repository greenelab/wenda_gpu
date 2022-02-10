import gpytorch
import torch
import numpy as np

from LBFGS import FullBatchLBFGS
from scipy.stats import norm


class ExactGPModel(gpytorch.models.ExactGP):
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
    return np.power(1-x, k)


def get_confidence(model, likelihood, x, y):
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


def age_transform(age, adult_age=20):
    age = (age+1)/(1+adult_age)
    y = np.where(age <= 1, np.log(age), age-1)
    return(y)


def age_back_transform(trans_age, adult_age=20):
    y = np.where(trans_age < 0,
                 (1+adult_age)*np.exp(trans_age)-1,
                 (1+adult_age)*trans_age+adult_age)
    return(y)
