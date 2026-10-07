import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert categorical columns to numbers
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.factorize(X[col])[0]

X = X.fillna(X.median())

# Train-test split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X.iloc[train_idx].values
X_test = X.iloc[test_idx].values
y_train = y.iloc[train_idx].values
y_test = y.iloc[test_idx].values

classes = np.unique(y_train)

# Calculate mean, variance and prior probability
mean = {}
var = {}
prior = {}

for c in classes:
    Xc = X_train[y_train == c]

    mean[c] = np.mean(Xc, axis=0)
    var[c] = np.var(Xc, axis=0) + 1e-9
    prior[c] = len(Xc) / len(X_train)

# Gaussian probability
def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * \
           np.exp(-((x - mean) ** 2) / (2 * var))

# Prediction
def predict(X):
    predictions = []

    for sample in X:
        probabilities = {}

        for c in classes:
            probability = prior[c]

            for j in range(len(sample)):
                probability *= gaussian(
                    sample[j],
                    mean[c][j],
                    var[c][j]
                )

            probabilities[c] = probability

        predictions.append(
            max(probabilities, key=probabilities.get)
        )

    return np.array(predictions)

# Predict
y_pred = predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)

print("Actual :", y_test)
print("Predicted:", y_pred)
print("Accuracy:", accuracy)