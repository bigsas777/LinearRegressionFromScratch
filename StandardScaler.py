import numpy as np

class StandardScaler():
    def __init__(self):
        self.means = 0.
        self.std_devs = 0.

    def fit(self, input_data):
        self.means = np.mean(input_data, axis=0)
        self.std_devs = np.std(input_data, axis=0)

    def transform(self, input_data):
        return (input_data - self.means) / self.std_devs 

    def fit_transform(self, input_data):
        self.fit(input_data)
        return self.transform(input_data)

    def inverse_transform(self, normalized_data):
        return normalized_data * self.std_devs + self.means