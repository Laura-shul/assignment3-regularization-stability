import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from utils import ridge_fit, lasso_fit, predict, mse, compute_stability_remove

np.random.seed(42)

def compute_lasso_stability(X_train, y_train, X_test, lam):
    # average change in predictions after removing one training point
    w_full = lasso_fit(X_train, y_train, lam)
    pred_full = predict(X_test, w_full)
    diffs = []
    for i in range(len(y_train)):
        X_loo = np.delete(X_train, i, axis=0)
        y_loo = np.delete(y_train, i)
        w_loo = lasso_fit(X_loo, y_loo, lam)
        pred_loo = predict(X_test, w_loo)
        diffs.append(np.mean(np.abs(pred_full - pred_loo)))
    return np.mean(diffs)

def run_l1_vs_l2_experiment(X_train, y_train, X_test, y_test, lambdas):
    ridge_test = []
    lasso_test = []
    ridge_stab = []
    lasso_stab = []

    for lam in lambdas:
        # ridge
        w_r = ridge_fit(X_train, y_train, lam)
        ridge_test.append(mse(y_test, predict(X_test, w_r)))
        ridge_stab.append(compute_stability_remove(X_train, y_train, X_test, lam))

        # lasso
        w_l = lasso_fit(X_train, y_train, lam)
        lasso_test.append(mse(y_test, predict(X_test, w_l)))
        lasso_stab.append(compute_lasso_stability(X_train, y_train, X_test, lam))

        print(f'  lambda={lam:.4f} done')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('L1 (Lasso) vs L2 (Ridge) Regularization')

    axes[0].plot(lambdas, ridge_test, label='ridge test error', marker='o', color='blue')
    axes[0].plot(lambdas, lasso_test, label='lasso test error', marker='s', color='red')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('lambda')
    axes[0].set_ylabel('test MSE')
    axes[0].set_title('Test Error: L1 vs L2')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(lambdas, ridge_stab, label='ridge stability', marker='o', color='blue')
    axes[1].plot(lambdas, lasso_stab, label='lasso stability', marker='s', color='red')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('lambda')
    axes[1].set_ylabel('mean prediction change (lower = more stable)')
    axes[1].set_title('Stability: L1 vs L2')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/l1_vs_l2.png', dpi=150)
    plt.close()
    print('saved: plots/l1_vs_l2.png')


if __name__ == '__main__':
    print("=== L1 vs L2 Experiment ===")
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

    lambdas = np.logspace(-3, 3, 20)
    run_l1_vs_l2_experiment(X_train, y_train, X_test, y_test, lambdas)