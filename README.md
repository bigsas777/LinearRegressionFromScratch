# 🚀 Machine Learning Algorithms from Scratch

This repository contains implementations of various machine learning algorithms built entirely from scratch using Python and NumPy. The goal is to gain a deep understanding of these models by coding them from the ground up, without relying on high-level machine learning libraries.

## ✨ Features
✅ Implementations of fundamental ML algorithms  
✅ Focus on mathematical foundations and efficiency  
✅ No external ML libraries, only NumPy and core Python  

## 📌 Implemented Algorithms
- **🔢 Linear Regression**: Uses matrix operations for efficient computation and supports multiple features (multivariate regression). Optimized with gradient descent for parameter tuning.
- **🧠 Logistic Regression**: Implements binary classification using the sigmoid function and gradient-based optimization. The model predicts probabilities and classifies inputs into binary categories (0 or 1) based on a threshold (default: 0.5). Includes a cost function based on log loss and supports gradient descent for parameter updates.

## 📊 Feature Scaling
This repository also includes implemetations of feature scaling techniques to preprocess data:
- **Min-Max Scaling**: Rescales features to a fixed range, typically [0, 1], by adjusting the minimum and maximum values of each feature.
- **Standard Scaling**: Standardizes features by centering them around the mean and scaling them to have unit variance, making the data suitable for algorithms sensitive to feature magnitudes.

## 🛠️ Input Handling
All models in this repository accept inputs as NumPy arrays. If the input arrays do not match the required shape, the models will internally reshape or modify them as needed to ensure compatibility with the algorithm's computations.

This project is a continuous learning journey—stay tuned for more algorithms! 🚀

