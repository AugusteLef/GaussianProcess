import numpy as np
from scipy.stats import norm
import torch
import gpytorch

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""

# In order to complete this task we followed the GPyTorch Regression tutorial that can
# be found here : [https://docs.gpytorch.ai/en/v1.2.0/examples/01_Exact_GPs/Simple_GP_Regression.html]
# We copyed partially or in integrality some part of these tutorial/


class Model():

    def __init__(self):
        """
            TODO: enter your code here
        """
        # Initialize de likelihood (not the model yet as we need training set to do it)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # the number of iteration we will use to train our model
        self.iteration = 50

    def predict(self, test_x):
        """
            TODO: enter your code here
        """

        # Create a tensor of the test set
        test_x = torch.Tensor(test_x)

        # Turn the model into evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        # Give us MultivariateNormals (for each prediction)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_preds_obs = self.likelihood(self.model(test_x))

        #take the mean of the multivariate observation
        y_preds = y_preds_obs.mean.numpy()

        prop = 1 - norm(y_preds, y_preds_obs.variance.detach().numpy()).cdf([0.5]*len(y_preds))
        y_preds[(y_preds < 0.5) & (prop > 0.3)] = 0.5

        return y_preds

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        # Construct tensors with training data
        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)

        # Initialize the model with our training set and likelihood
        self.model = ExactGPModel(train_x, train_y, self.likelihood)

        # Train both the likelihood and the model in order to find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the Adam optimizer
        # We tried the SGD, but results were not good enough compare to the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.iteration):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backpropagation gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, self.iteration, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item()
            ))
            optimizer.step()


# Simplest form of Gaussian Model, Inference (cf. Tutorial)
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Here maybe we can use another kernel to compute the covar => ??
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()
