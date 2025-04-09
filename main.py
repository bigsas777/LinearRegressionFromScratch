from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import time

# Function to generate datasets
def generate_dataset(size, regression_type):
    np.random.seed(42)
    input = np.random.rand(size, 1) * 10
    noise = np.random.randn(size, 1) * 2

    if regression_type == "linear":
        output = 3 * input + 7 + noise
    elif regression_type == "logistic":
        output = (input > 5).astype(int)  # Binary classification
    else:
        raise ValueError("Invalid regression type. Choose 'linear' or 'logistic'.")

    return input, output

# Menu for model selection
print("Choose the model to test:")
print("1. Linear Regression")
print("2. Logistic Regression")
choice = int(input("Enter your choice (1 or 2): "))

# User input for dataset size
dataset_size = int(input("Enter the size of the dataset to generate: "))

# Generate dataset based on user choice
if choice == 1:
    input, output = generate_dataset(dataset_size, "linear")
elif choice == 2:
    input, output = generate_dataset(dataset_size, "logistic")
else:
    raise ValueError("Invalid choice. Please select 1 or 2.")

# Split dataset into training and testing sets
input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

# Train and test the selected model
if choice == 1:
    # Linear Regression
    model = LinearRegression(input_train, output_train)
    start = time.perf_counter()
    model.train()
    end = time.perf_counter()

    # Calculate metrics
    train_time = end - start
    test_pred = model.predict(input_test)
    r2 = r2_score(output_test, test_pred)

    # Print metrics
    print(f"Training time: {train_time}s")
    print(f"R2 Score: {r2}")

    # Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(input_train, output_train, color='blue', alpha=0.6, label="Training data")
    plt.scatter(input_test, output_test, color='orange', alpha=0.6, label="Test data")

    # Regression line
    X_line = np.linspace(min(input), max(input), 100)
    Y_line = model.thetas[1] * X_line + model.thetas[0]
    plt.plot(X_line, Y_line, 'r-', linewidth=2, label="Regression Line")

    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Linear Regression", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

elif choice == 2:
    # Logistic Regression
    model = LogisticRegression(input_train, output_train)
    start = time.perf_counter()
    model.train()
    end = time.perf_counter()

    # Calculate metrics
    train_time = end - start
    test_pred = model.predict(input_test)
    accuracy = accuracy_score(output_test, test_pred)

    # Print metrics
    print(f"Training time: {train_time}s")
    print(f"Accuracy: {accuracy}")

    # Scatter plot for logistic regression
    plt.figure(figsize=(8, 6))
    plt.scatter(input_train, output_train, color='blue', alpha=0.6, label="Training data")
    plt.scatter(input_test, output_test, color='orange', alpha=0.6, label="Test data")

    # Sigmoid curve
    X_line = np.linspace(min(input), max(input), 100).reshape(-1, 1)
    Y_line = model.predict(X_line)
    plt.plot(X_line, Y_line, 'r-', linewidth=2, label="Sigmoid Curve")

    plt.xlabel("X", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title("Logistic Regression", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
