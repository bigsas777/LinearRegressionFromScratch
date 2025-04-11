import numpy as np

class MinMaxScaler():
    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, input_data): 
        self.mins = np.min(input_data, axis=0)
        self.maxs = np.max(input_data, axis=0)

    def transform(self, input_data):
        return (input_data - self.mins) / (self.maxs - self.mins)

    def fit_transform(self, input_data):
        self.fit(input_data)
        return self.transform(input_data)

    def inverse_transform(self, normalized_data):
        return normalized_data * (self.maxs - self.mins) + self.mins