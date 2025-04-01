import numpy as np

class LinearRegression():
    """
    Class implementing linear regression using the gradient descent method.
    """

    def __init__(self, input, output, n_iters=1000, learning_rate=0.01):
        """
        Initializes the linear regression model.

        Args:
            input (numpy.ndarray): Input matrix (features) of dimensions (m, n).
            output (numpy.ndarray): Output vector (target) of dimensions (m, 1).
            n_iters (int): Number of iterations for gradient descent.
            learning_rate (float): Learning rate (alpha) for gradient descent.

        Returns:
            None
        """
        self.input = self.add_ones(input) # Adds a column of ones to handle theta0
        self.output = output
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
            numpy.ndarray: Predicted output vector of dimensions (m, 1).
        """
        return self.predict_with_ones(self.add_ones(input_pred))
    
    def predict_with_ones(self, input_pred):
        """
        Makes predictions using an input matrix that already includes a column of ones.

        Args:
            input_pred (numpy.ndarray): Input matrix with a column of ones included.

        Returns:
            numpy.ndarray: Predicted output vector with dimensions (m, 1).
        """
        return np.dot(input_pred, self.thetas)

    def cost_function(self):
        """
        Computes the cost function (mean squared error).

        Args:
            None

        Returns:
            float: Value of the cost function.
        """
        error_vector = self.error()
        return (1/(2*self.m)) * np.dot(np.transpose(error_vector), error_vector)

    def gradient_descent(self):
        """
        Updates the theta parameters using gradient descent.

        Args:
            None

        Returns:
            None
        """
        self.thetas -= self.learning_rate * (1/self.m) * np.dot(np.transpose(self.input), self.error())
    
    def add_ones(self, input_matrix):
        """
        Adds a column of ones to the input matrix to account for the theta0 parameter.

        Args:
            input_matrix (numpy.ndarray): Input matrix with dimensions (m, n).

        Returns:
            numpy.ndarray: Input matrix with an added column of ones, dimensions (m, n+1).
        """
        m = input_matrix.shape[0]
        ones = np.ones((m, 1)) # Column of ones to handle theta0
        input_matrix = np.hstack((ones, input_matrix))
        return input_matrix
    
    def error(self):
        """
        Computes the error vector between predictions and actual values.

        Args:
            None

        Returns:
            numpy.ndarray: Error vector with dimensions (m, 1).
        """
        return self.predict_with_ones(self.input) - self.output