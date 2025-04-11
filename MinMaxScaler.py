import numpy as np

class MinMaxScaler():
    """
    Class implementing Min-Max scaling for feature normalization.
    """

    def __init__(self):
        """
        Initializes the MinMaxScaler.

        Args:
            None

        Returns:
            None
        """
        self.mins = None
        self.maxs = None

    def fit(self, input_data): 
        """
        Computes the minimum and maximum values for each feature.

        Args:
            input_data (numpy.ndarray): Input data matrix of dimensions (m, n).

        Returns:
            None
        """
        self.mins = np.min(input_data, axis=0)
        self.maxs = np.max(input_data, axis=0)

    def transform(self, input_data):
        """
        Scales the input data using the computed minimum and maximum values.

        Args:
            input_data (numpy.ndarray): Input data matrix of dimensions (m, n).

        Returns:
            numpy.ndarray: Scaled data matrix of dimensions (m, n).
        """
        return (input_data - self.mins) / (self.maxs - self.mins)

    def fit_transform(self, input_data):
        """
        Fits the scaler to the data and transforms it.

        Args:
            input_data (numpy.ndarray): Input data matrix of dimensions (m, n).

        Returns:
            numpy.ndarray: Scaled data matrix of dimensions (m, n).
        """
        self.fit(input_data)
        return self.transform(input_data)

    def inverse_transform(self, normalized_data):
        """
        Reverts the scaling transformation.

        Args:
            normalized_data (numpy.ndarray): Scaled data matrix of dimensions (m, n).

        Returns:
            numpy.ndarray: Original data matrix of dimensions (m, n).
        """
        return normalized_data * (self.maxs - self.mins) + self.mins