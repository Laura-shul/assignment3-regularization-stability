import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from utils import ridge_fit, predict, mse, compute_stability_remove

np.random.seed(42)

def run_size_experiment(X_train, y_train, lam_fixed=1.0):
    sizes = [20, 40, 60, 80, 100, 150, 200]
    stabilities = []
    test_errors = []

    # fixed test set from the end of training data
    X_test = X_train[150:]
    y_test = y_train[150:]

    for n in sizes:
        X_tr = X_train[:n]
        y_tr = y_train[:n]
        stab = compute_stability_remove(X_tr, y_tr, X_test, lam_fixed)
        w = ridge_fit(X_tr, y_tr, lam_fixed)
        err = mse(y_test, predict(X_test, w))
        stabilities.append(stab)
        test_errors.append(err)
        print(f'  n={n}: stability={stab:.6f}, test_error={err:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Effect of Dataset Size on Stability (lambda=1.0)')

    axes[0].plot(sizes, stabilities, marker='o', color='green')
    axes[0].set_xlabel('training set size n')
    axes[0].set_ylabel('mean prediction change (lower = more stable)')
    axes[0].set_title('Stability vs n')
    axes[0].grid(True)

    axes[1].plot(sizes, test_errors, marker='s', color='orange')
    axes[1].set_xlabel('training set size n')
    axes[1].set_ylabel('test MSE')
    axes[1].set_title('Test Error vs n')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/size.png', dpi=150)
    plt.close()
    print('saved: plots/size.png')


if __name__ == '__main__':
    print("=== Dataset Size Experiment ===")
    X, y = make_regression(n_samples=250, n_features=10, noise=20, random_state=42)

    X_train = X[:200]
    y_train = y[:200]

    mean_X = X_train.mean(axis=0)
    std_X = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean_X) / std_X

    mean_y = y_train.mean()
    std_y = y_train.std() + 1e-8
    y_train = (y_train - mean_y) / std_y

    run_size_experiment(X_train, y_train, lam_fixed=1.0)