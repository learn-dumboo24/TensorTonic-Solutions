import kagglehub
import pandas as pd
import numpy as np
import os

# loading dataset (first time trying kagglehub)
path = kagglehub.dataset_download("aiwithcagri/multiple-linear-regression-tesla-model-y-price")
files = os.listdir(path)

print("files i got:", files)

df = pd.read_csv(os.path.join(path, files[0]))

print("quick look")
print(df.head())

# trying to encode color manually
def make_dummies(col):
    codes, uniques = pd.factorize(col)
    print("colors:", list(uniques))

    n = len(col)
    k = len(uniques) - 1  # dropping first

    mat = np.zeros((n, k))

    for i in range(n):
        if codes[i] != 0:
            mat[i][codes[i] - 1] = 1

    return mat

color_data = make_dummies(df['color'])

# building features
num_data = df[['year', 'km']].values

X = np.concatenate([num_data, color_data], axis=1)
y = df['price'].values.reshape(-1, 1)

print("shape:", X.shape)

# scaling (otherwise numbers too big)
mean = X.mean(axis=0)
std = X.std(axis=0)

X = (X - mean) / std

# adding bias column
m = X.shape[0]
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# random init (just trying)
theta = np.random.randn(X.shape[1], 1)

print("starting theta:", theta.T)

# gradient descent
lr = 0.01
epochs = 1000

for i in range(epochs):
    pred = X @ theta
    err = pred - y

    grad = (X.T @ err) / m
    theta = theta - lr * grad

    if i % 200 == 0:
        print("epoch", i, "loss", np.mean(err**2))

print("done")

print("theta:", theta.T)

preds = X @ theta

print("first few predictions vs actual")
for i in range(5):
    print(preds[i][0], y[i][0])

print("mse:", np.mean((preds - y)**2))