import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from experiment1 import run_main_experiment, run_comparison_plot
from experiment2 import run_size_experiment
from experiment3 import run_l1_vs_l2_experiment

np.random.seed(42)

lambdas = np.logspace(-3, 3, 20)

# synthetic dataset
# 250 samples, 10 features, gaussian noise
print("=== Synthetic Dataset ===")
X, y = make_regression(n_samples=250, n_features=10, noise=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean_X) / std_X
X_test = (X_test - mean_X) / std_X

mean_y = y_train.mean()
std_y = y_train.std() + 1e-8
y_train = (y_train - mean_y) / std_y
y_test = (y_test - mean_y) / std_y

res_syn = run_main_experiment(X_train, y_train, X_test, y_test, lambdas, 'Synthetic Dataset', 'synthetic')

# diabetes dataset — real world regression benchmark
print("=== Diabetes Dataset ===")
data = load_diabetes()
X2 = data.data
y2 = data.target

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

mean_X2 = X2_train.mean(axis=0)
std_X2 = X2_train.std(axis=0) + 1e-8
X2_train = (X2_train - mean_X2) / std_X2
X2_test = (X2_test - mean_X2) / std_X2

mean_y2 = y2_train.mean()
std_y2 = y2_train.std() + 1e-8
y2_train = (y2_train - mean_y2) / std_y2
y2_test = (y2_test - mean_y2) / std_y2

res_dia = run_main_experiment(X2_train, y2_train, X2_test, y2_test, lambdas, 'Diabetes Dataset', 'diabetes')

# compare results across datasets
print("=== Comparison Plot ===")
run_comparison_plot(res_syn, res_dia, lambdas)

# effect of dataset size on stability
print("=== Dataset Size Experiment ===")
run_size_experiment(X_train, y_train, lam_fixed=1.0)

# compare L1 and L2 regularization
print("=== L1 vs L2 ===")
run_l1_vs_l2_experiment(X_train, y_train, X_test, y_test, lambdas)

print("\ndone! check plots/ folder")