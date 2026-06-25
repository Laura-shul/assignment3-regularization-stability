import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from utils import ridge_fit, least_squares_fit, predict, mse
from utils import compute_stability_remove, compute_stability_ls

np.random.seed(42)
lambdas = np.logspace(-3, 3, 20)

print("=" * 50)
print("SUMMARY OF RESULTS")
print("=" * 50)

# synthetic dataset
print("\n--- Synthetic Dataset ---")
X, y = make_regression(n_samples=250, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mean_X = X_train.mean(axis=0); std_X = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean_X) / std_X; X_test = (X_test - mean_X) / std_X
mean_y = y_train.mean(); std_y = y_train.std() + 1e-8
y_train = (y_train - mean_y) / std_y; y_test = (y_test - mean_y) / std_y

w_ls = least_squares_fit(X_train, y_train)
ls_test_syn = mse(y_test, predict(X_test, w_ls))
ls_stab_syn = compute_stability_ls(X_train, y_train, X_test)

ridge_errors = [mse(y_test, predict(X_test, ridge_fit(X_train, y_train, l))) for l in lambdas]
ridge_stabs = [compute_stability_remove(X_train, y_train, X_test, l) for l in lambdas]
best_i = np.argmin(ridge_errors)

print(f"Best lambda:        {lambdas[best_i]:.4f}")
print(f"Ridge min test MSE: {ridge_errors[best_i]:.4f}")
print(f"LS test MSE:        {ls_test_syn:.4f}")
print(f"LS stability:       {ls_stab_syn:.6f}")
print(f"Min ridge stability:{min(ridge_stabs):.6f}")

# diabetes dataset
print("\n--- Diabetes Dataset ---")
data = load_diabetes()
X2, y2 = data.data, data.target
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
mean_X2 = X2_train.mean(axis=0); std_X2 = X2_train.std(axis=0) + 1e-8
X2_train = (X2_train - mean_X2) / std_X2; X2_test = (X2_test - mean_X2) / std_X2
mean_y2 = y2_train.mean(); std_y2 = y2_train.std() + 1e-8
y2_train = (y2_train - mean_y2) / std_y2; y2_test = (y2_test - mean_y2) / std_y2

w_ls2 = least_squares_fit(X2_train, y2_train)
ls_test_dia = mse(y2_test, predict(X2_test, w_ls2))
ls_stab_dia = compute_stability_ls(X2_train, y2_train, X2_test)

ridge_errors2 = [mse(y2_test, predict(X2_test, ridge_fit(X2_train, y2_train, l))) for l in lambdas]
ridge_stabs2 = [compute_stability_remove(X2_train, y2_train, X2_test, l) for l in lambdas]
best_i2 = np.argmin(ridge_errors2)

print(f"Best lambda:        {lambdas[best_i2]:.4f}")
print(f"Ridge min test MSE: {ridge_errors2[best_i2]:.4f}")
print(f"LS test MSE:        {ls_test_dia:.4f}")
print(f"LS stability:       {ls_stab_dia:.6f}")
print(f"Min ridge stability:{min(ridge_stabs2):.6f}")

print("\n" + "=" * 50)
print("Numbers used in report Table 1")
print("=" * 50)
print(f"Synthetic: best_lambda={lambdas[best_i]:.2f}, min_test={ridge_errors[best_i]:.4f}, ls_test={ls_test_syn:.4f}, ls_stab={ls_stab_syn:.6f}")
print(f"Diabetes:  best_lambda={lambdas[best_i2]:.2f}, min_test={ridge_errors2[best_i2]:.4f}, ls_test={ls_test_dia:.4f}, ls_stab={ls_stab_dia:.6f}")