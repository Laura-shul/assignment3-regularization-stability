import numpy as np

np.random.seed(42)

# ridge regression from scratch
def ridge_fit(X, y, lam):
    d = X.shape[1]
    w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    return w

# unregularized least squares
def least_squares_fit(X, y):
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

# lasso via coordinate descent from scratch
def lasso_fit(X, y, lam):
    n, d = X.shape
    w = np.zeros(d)
    for iteration in range(1000):
        w_old = w.copy()
        for j in range(d):
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual
            if rho < -lam * n:
                w[j] = (rho + lam * n) / (X[:, j] @ X[:, j])
            elif rho > lam * n:
                w[j] = (rho - lam * n) / (X[:, j] @ X[:, j])
            else:
                w[j] = 0.0
        if np.max(np.abs(w - w_old)) < 1e-6:
            break
    return w

def predict(X, w):
    return X @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# average change in predictions after removing one training point
# lower = more stable
def compute_stability_remove(X_train, y_train, X_test, lam):
    w_full = ridge_fit(X_train, y_train, lam)
    pred_full = predict(X_test, w_full)
    diffs = []
    for i in range(len(y_train)):
        X_loo = np.delete(X_train, i, axis=0)
        y_loo = np.delete(y_train, i)
        w_loo = ridge_fit(X_loo, y_loo, lam)
        pred_loo = predict(X_test, w_loo)
        diffs.append(np.mean(np.abs(pred_full - pred_loo)))
    return np.mean(diffs)

# average change in loss after removing one training point
# lower = more stable
def compute_stability_remove_loss(X_train, y_train, X_test, y_test, lam):
    w_full = ridge_fit(X_train, y_train, lam)
    loss_full = mse(y_test, predict(X_test, w_full))
    diffs = []
    for i in range(len(y_train)):
        X_loo = np.delete(X_train, i, axis=0)
        y_loo = np.delete(y_train, i)
        w_loo = ridge_fit(X_loo, y_loo, lam)
        loss_loo = mse(y_test, predict(X_test, w_loo))
        diffs.append(abs(loss_full - loss_loo))
    return np.mean(diffs)

# average change in predictions after replacing one training point
# lower = more stable
def compute_stability_replace(X_train, y_train, X_test, lam):
    w_full = ridge_fit(X_train, y_train, lam)
    pred_full = predict(X_test, w_full)
    diffs = []
    rng = np.random.default_rng(0)
    for i in range(len(y_train)):
        X_rep = X_train.copy()
        y_rep = y_train.copy()
        j = rng.integers(0, len(y_train))
        X_rep[i] = X_train[j]
        y_rep[i] = y_train[j]
        w_rep = ridge_fit(X_rep, y_rep, lam)
        pred_rep = predict(X_test, w_rep)
        diffs.append(np.mean(np.abs(pred_full - pred_rep)))
    return np.mean(diffs)

# stability for unregularized least squares
# used to compare with ridge
def compute_stability_ls(X_train, y_train, X_test):
    w_full = least_squares_fit(X_train, y_train)
    pred_full = predict(X_test, w_full)
    diffs = []
    for i in range(len(y_train)):
        X_loo = np.delete(X_train, i, axis=0)
        y_loo = np.delete(y_train, i)
        w_loo = least_squares_fit(X_loo, y_loo)
        pred_loo = predict(X_test, w_loo)
        diffs.append(np.mean(np.abs(pred_full - pred_loo)))
    return np.mean(diffs)

# theoretical stability bound for ridge regression
# beta <= 2 * kappa^2 * B^2 / (lambda * m)
# lower = more stable
def theoretical_bound(X_train, y_train, lam):
    m = len(y_train)
    kappa2 = np.max(np.sum(X_train ** 2, axis=1))
    B = np.max(np.abs(y_train))
    sigma = 2 * B
    beta = (sigma ** 2) * kappa2 / (2 * lam * m)
    return beta