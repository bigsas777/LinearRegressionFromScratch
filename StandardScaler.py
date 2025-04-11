import numpy as np

class StandardScaler():
    """
    Class implementing standard scaling for feature normalization.
    """

    def __init__(self):
        """
        Initializes the StandardScaler.

        Args:
            None

        Returns:
            None
        """
        self.means = None
        self.std_devs = None

    def fit(self, input_data):
        """
        Computes the mean and standard deviation for each feature.

        Args:
            input_data (numpy.ndarray): Input data matrix of dimensions (m, n).

        Returns:
            None
        """
        self.means = np.mean(input_data, axis=0)
        self.std_devs = np.std(input_data, axis=0)

    def transform(self, input_data):
        """
        Scales the input data using the computed mean and standard deviation.

        Args:
            input_data (numpy.ndarray): Input data matrix of dimensions (m, n).

        Returns:
            numpy.ndarray: Scaled data matrix of dimensions (m, n).
        """
        return (input_data - self.means) / self.std_devs 

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
        return normalized_data * self.std_devs + self.means