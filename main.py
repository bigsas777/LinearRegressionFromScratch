from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from math import sqrt

# Generazione dataset di training
np.random.seed(42)
input = np.random.rand(100, 1) * 10
noise = np.random.randn(100, 1) * 2  
output = 3 * input + 7 + noise

# Generazione dataset di test
np.random.seed(99)
input_test = np.random.rand(100, 1) * 10  
noise_test = np.random.randn(100, 1) * 2  
output_test = 3 * input_test + 7 + noise_test

# Creazione e addestramento modello
model = LinearRegression(0.01, 2000)
model.train(input, output)
theta0, theta1 = model.theta0, model.theta1

# Stampa dei parametri theta trovati
print(f"Theta0: {theta0}, Theta1: {theta1}")

# Metriche del modello
test_pred = model.predict(input_test)
r2 = r2_score(output_test, test_pred)
print(f"R2 Score: {r2}")
mae = mean_absolute_error(output_test, test_pred)
print('Mean Absolute Error (MAE):', mae)
mse = mean_squared_error(output_test, test_pred)
print('Root Mean Squared Error:', sqrt(mse))

# Scatter plot con retta regressione
plt.figure(figsize=(8,6))
plt.scatter(input, output, color='blue', alpha=0.6, label="Dati di Training")
plt.scatter(input_test, output_test, color='orange', alpha=0.6, label="Dati di Test")

# Retta di regressione
X_line = np.linspace(min(input), max(input), 100)
Y_line = theta1 * X_line + theta0  
plt.plot(X_line, Y_line, 'r-', linewidth=2, label="Modello 1")

plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title("Test Linear Regression from Scratch", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
