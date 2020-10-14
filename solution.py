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

# Test of replicate the function cost but adapter for Tensor
def cost_function_torch(true, predicted):
    """

    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = (predicted>=true) & mask
    mask_w2 = ((predicted<true) & (predicted >=THRESHOLD)) & mask
    mask_w3 = (predicted<THRESHOLD) & mask

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = (predicted>true) & mask
    mask_w2 = (predicted<=true) & mask

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*((predicted < THRESHOLD) & (true<THRESHOLD))
    if reward is None:
        reward = 0
    return torch.mean(cost) - torch.mean(torch.Tensor(reward))

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
        self.model = None

        # the number of iteration we will use to train our model (training loop)
        self.iteration = 50

    def predict(self, test_x):
        """
            TODO: enter your code here
        """

        # Create a tensor of the test set
        nbr = len(test_x)
        test_x = torch.Tensor(test_x)

        # Turn the model into evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Get the prediction (in form of MutlvariateNormal)
        y_obs = self.likelihood(self.model(test_x))
        # For each observation we take the mean of this one as the prediction
        y_preds = y_obs.mean.detach().numpy()

        ### Trick to avoid too much penality from the loss function of the assignment ###

        # For each observation we also get the variance of the distribution
        variance_obs = y_obs.variance.detach().numpy()
        # Then we can define normal continuous random variable with mean(=loc)
        # and var(=scale) from the predicted distribution
        norm_ditribution = norm(y_preds, variance_obs)
        # evaluated cdf at THRESHOLD for each predicted distibution
        cdf_half = norm_ditribution.cdf([THRESHOLD] * nbr)
        # find the best confidence value using a for loop with value 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9
        confidence = 0.7

        # Update prediction in order to be less penalized by the loss function
        y_preds[(y_preds < THRESHOLD) & (cdf_half < confidence)] = THRESHOLD + 0.000000001

        #############################################################################

        return y_preds


    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        # Construct tensors with training data (and keep a copy of y_train for the loss function test)
        # copy_train_y = train_y
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

            ######################################
            # Here we should try to use the given loss function to train our model
            # It would probably allow us to not manually modify the predications at the end

            #output_np = output.mean.detach().numpy()
            #loss = cost_function_torch(train_y, output.mean)
            #print(loss)
            ######################################

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
