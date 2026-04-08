import numpy as np

# sigmoid (keeping it simple but stable enough)
def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.1, steps=500):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, 1)

    n, d = X.shape

    # init
    w = np.zeros((d, 1))
    b = 0.0

    for i in range(steps):
        # forward
        z = X @ w + b
        p = _sigmoid(z)

        # loss (just for checking)
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

        # gradients
        dz = p - y
        dw = (X.T @ dz) / n
        db = np.mean(dz)

        # update
        w = w - lr * dw
        b = b - lr * db

        if i % 100 == 0:
            print("step", i, "loss", loss)

    return w.flatten(), float(b)


# quick test
X = [[0],[1],[2],[3]]
y = [0,0,1,1]

w, b = train_logistic_regression(X, y)

print("w:", w)
print("b:", b)

# predictions
def predict(X, w, b):
    X = np.array(X)
    probs = _sigmoid(X @ w.reshape(-1,1) + b)
    return (probs >= 0.5).astype(int)

preds = predict(X, w, b)
print("preds:", preds.flatten())