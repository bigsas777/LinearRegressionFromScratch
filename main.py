from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time

# Generate training dataset
np.random.seed(420)
input = np.random.rand(100, 1) * 10
noise = np.random.randn(100, 1) * 2  
output = 3 * input + 7 + noise

# Generate test dataset
np.random.seed(777)
input_test = np.random.rand(100, 1) * 10  
noise_test = np.random.randn(100, 1) * 2  
output_test = 3 * input_test + 7 + noise_test

# Create and train model
model = LinearRegression(input, output)
start = time.perf_counter()
model.train()
end = time.perf_counter()

# Calculate model metrics
train_time = end - start
test_pred = model.predict(input_test)
r2 = r2_score(output_test, test_pred)

# Print metrics
print(f"Training time: {train_time}s")
print(f"R2 Score: {r2}")

# Scatter plot with regression line
plt.figure(figsize=(8,6))
plt.scatter(input, output, color='blue', alpha=0.6, label="Training data")
plt.scatter(input_test, output_test, color='orange', alpha=0.6, label="Test data")

# Regression line
X_line = np.linspace(min(input), max(input), 100)
Y_line = model.thetas[1] * X_line + model.thetas[0]  
plt.plot(X_line, Y_line, 'r-', linewidth=2, label="Model 1")

plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title("Linear Regression from Scratch", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
