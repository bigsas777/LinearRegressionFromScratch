import numpy as np
from utils import add_ones, ensure_2d

class LogisticRegression:
    """
    Class implementing logistic regression using the gradient descent method.
    """

    def __init__(self, input, output, n_iters=1000, learning_rate=0.01):
        """
        Initializes the logistic regression model.

        Args:
            input (numpy.ndarray): Input matrix (features) of dimensions (m, n).
            output (numpy.ndarray): Output vector (target) of dimensions (m, 1).
            n_iters (int): Number of iterations for gradient descent.
            learning_rate (float): Learning rate (alpha) for gradient descent.

        Returns:
            None
        """
        self.input = add_ones(ensure_2d(input))  # Ensures input is 2D
        self.output = ensure_2d(output)  # Ensures output is 2D
        self.m = self.input.shape[0]
        self.thetas = np.random.rand(self.input.shape[1], 1)
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.costs = []

    def train(self):
        """
        Trains the model using gradient descent.

        Args:
            None

        Returns:
            None
        """
        for _ in range(self.n_iters):
            self.costs.append(self.cost_function())
            self.gradient_descent()

    def predict(self, input_pred):
        """
        Makes predictions using the calculated theta parameters.

        Args:
            input_pred (numpy.ndarray): Input matrix for prediction of dimensions (m, n).

        Returns:
            numpy.ndarray: Predicted output vector of dimensions (m, 1), with values 0 or 1.
        """
        return np.where(self.predict_with_ones(add_ones(ensure_2d(input_pred))) >= 0.5, 1, 0)

    def predict_with_ones(self, input_pred):
        """
        Makes predictions using an input matrix that already includes a column of ones.

        Args:
            input_pred (numpy.ndarray): Input matrix with a column of ones included.

        Returns:
            numpy.ndarray: Predicted output vector with dimensions (m, 1).
        """
        return self.sigmoid(np.dot(input_pred, self.thetas))

    def sigmoid(self, z):
        """
        Computes the sigmoid function.

        Args:
            z (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Sigmoid of the input array.
        """
        return 1 / (1 + np.exp(-z))

    def cost_function(self):
        """
        Computes the cost function (log loss).

        Args:
            None

        Returns:
            float: Value of the cost function.
        """
        epsilon = 1e-9  # Small constant to avoid log(0)
        p1 = np.dot(-self.output.T, np.log(self.predict_with_ones(self.input) + epsilon))
        p2 = np.dot((1 - self.output).T, np.log(1 - self.predict_with_ones(self.input) + epsilon))
        return 1 / self.m * (p1 + p2)
    
    def gradient_descent(self):
        """
        Updates the theta parameters using gradient descent.

        Args:
            None

        Returns:
            None
        """
        self.thetas -= self.learning_rate * (1 / self.m) * np.dot(self.input.T, self.predict_with_ones(self.input) - self.output)
