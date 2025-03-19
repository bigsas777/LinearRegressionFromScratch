from numpy import random

class LinearRegression():
    # params: learning_rate(alpha): float, n_iters: int
    # return: void
    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.theta0 = random.rand()
        self.theta1 = random.rand()
        self.m = 0
        self.input = None
        self.output = None
        self.thetas0 = []
        self.thetas1 = []
        self.costs = []

    # params: input, output: numpy_array
    # return: void
    def train(self, input, output):
        self.m = input.size
        self.input = input
        self.output = output
        for i in range(self.n_iters):
            self.costs.append(self.cost_function())
            self.thetas0.append(self.theta0)
            self.thetas1.append(self.theta1)
            self.gradient_descent()

    # params: input_val: qualsiasi input numerico o array di num
    # return: float
    def predict(self, input_val):
        out_pred = self.theta0 + self.theta1 * input_val
        return out_pred

    # Funzione di costo
    def cost_function(self):
        sum = 0.
        for i in range(self.m):
           sum += (self.predict(self.input[i]) - self.output[i]) ** 2
        
        return (1/2*self.m) * sum

    # Applicazione del gradient descent per la minimizzazione dei parametri theta
    def gradient_descent(self):
        self.theta0 = self.theta0 - self.learning_rate * self.cf_der_theta0()
        self.theta1 = self.theta1 - self.learning_rate * self.cf_der_theta1()

    # Derivata parziale della funzione di costo rispetto a theta0
    def cf_der_theta0(self):
        sum = 0.
        for i in range(self.m):
           sum += self.predict(self.input[i]) - self.output[i]
        
        return (1/self.m) * sum
    
    # Derivata parziale della funzione di costo rispetto a theta1
    def cf_der_theta1(self):
        sum = 0.
        for i in range(self.m):
           sum += (self.predict(self.input[i]) - self.output[i]) * self.input[i]
        
        return (1/self.m) * sum