import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from utils import ridge_fit, least_squares_fit, predict, mse
from utils import compute_stability_remove, compute_stability_remove_loss
from utils import compute_stability_replace, compute_stability_ls, theoretical_bound

np.random.seed(42)

def compute_rloo(X_train, y_train, lam):
    # leave-one-out error
    losses = []
    for i in range(len(y_train)):
        X_loo = np.delete(X_train, i, axis=0)
        y_loo = np.delete(y_train, i)
        w_loo = ridge_fit(X_loo, y_loo, lam)
        loss_i = (predict(X_train[i:i+1], w_loo) - y_train[i]) ** 2
        losses.append(loss_i[0])
    return np.mean(losses)

def generalization_bound(r_emp, beta, M, m, delta=0.05):
    # upper bound on generalization error
    return r_emp + 2 * beta + (4 * m * beta + M) * np.sqrt(np.log(1/delta) / (2 * m))

def run_main_experiment(X_train, y_train, X_test, y_test, lambdas, title, fname):
    train_errors = []
    test_errors = []
    prediction_stability = []
    loss_stability = []
    replace_stability = []
    theory_bounds = []
    gen_gap = []
    rloo_errors = []
    gen_bounds = []

    m = len(y_train)

    # unregularized least squares baseline
    w_ls = least_squares_fit(X_train, y_train)
    ls_train = mse(y_train, predict(X_train, w_ls))
    ls_test = mse(y_test, predict(X_test, w_ls))
    ls_stab = compute_stability_ls(X_train, y_train, X_test)

    for lam in lambdas:
        w = ridge_fit(X_train, y_train, lam)
        train_error = mse(y_train, predict(X_train, w))
        test_error = mse(y_test, predict(X_test, w))
        train_errors.append(train_error)
        test_errors.append(test_error)
        gen_gap.append(abs(test_error - train_error))

        # stability via prediction change (per assignment)
        beta_pred = compute_stability_remove(X_train, y_train, X_test, lam)
        prediction_stability.append(beta_pred)

        # stability via loss change
        beta_loss = compute_stability_remove_loss(X_train, y_train, X_test, y_test, lam)
        loss_stability.append(beta_loss)

        replace_stability.append(compute_stability_replace(X_train, y_train, X_test, lam))
        theory_bounds.append(theoretical_bound(X_train, y_train, lam))

        rloo_errors.append(compute_rloo(X_train, y_train, lam))

        M = float(np.percentile((y_train - predict(X_train, w)) ** 2, 95))
        gen_bounds.append(generalization_bound(train_error, beta_loss, M, m, delta=0.05))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=13)

    # training and test errors
    axes[0, 0].plot(lambdas, train_errors, label='ridge train', marker='o')
    axes[0, 0].plot(lambdas, test_errors, label='ridge test', marker='s')
    axes[0, 0].axhline(ls_train, color='blue', linestyle='--', alpha=0.6, label='LS train')
    axes[0, 0].axhline(ls_test, color='orange', linestyle='--', alpha=0.6, label='LS test')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('lambda')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Ridge vs Least Squares: errors')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True)

    # stability metrics
    axes[0, 1].plot(lambdas, prediction_stability, label='remove (pred change)',
                    marker='^', color='green')
    axes[0, 1].plot(lambdas, loss_stability, label='remove (loss change)',
                    marker='D', color='blue', linestyle='-.')
    axes[0, 1].plot(lambdas, replace_stability, label='replace (pred change)',
                    marker='v', color='purple')
    axes[0, 1].axhline(ls_stab, color='red', linestyle='--', alpha=0.7, label='LS stability')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('lambda')
    axes[0, 1].set_ylabel('stability metric (lower = more stable)')
    axes[0, 1].set_title('Stability metrics vs Lambda')
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True)

    # empirical stability vs theoretical bound
    axes[0, 2].plot(lambdas, prediction_stability, label='empirical (pred change)',
                    marker='^', color='green')
    axes[0, 2].plot(lambdas, loss_stability, label='empirical (loss change)',
                    marker='D', color='blue', linestyle='-.')
    axes[0, 2].plot(lambdas, theory_bounds, label='theoretical bound',
                    marker='o', color='red', linestyle='--')
    c = loss_stability[0] * lambdas[0]
    fitted = [c / lam for lam in lambdas]
    axes[0, 2].plot(lambdas, fitted, label='fitted c/lambda', color='gray', linestyle=':')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_xlabel('lambda')
    axes[0, 2].set_ylabel('beta (log scale)')
    axes[0, 2].set_title('Empirical vs Theoretical Bound')
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True)

    # compare empirical, leave-one-out and test errors
    axes[1, 0].plot(lambdas, train_errors, label='R_emp (train)', marker='o')
    axes[1, 0].plot(lambdas, rloo_errors, label='R_loo', marker='s', color='purple')
    axes[1, 0].plot(lambdas, test_errors, label='R (test)', marker='^', color='green')
    axes[1, 0].plot(lambdas, gen_bounds, label='generalization bound (delta=0.05)',
                    marker='x', color='red', linestyle='--')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('lambda')
    axes[1, 0].set_ylabel('error')
    axes[1, 0].set_title('Train, LOO, Test and Bound')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True)

    # stability vs test error
    axes[1, 1].scatter(prediction_stability, test_errors, color='green',
                       label='pred change', zorder=5)
    axes[1, 1].scatter(loss_stability, test_errors, color='blue',
                       label='loss change', zorder=5, marker='D')
    axes[1, 1].set_xlabel('stability metric (lower = more stable)')
    axes[1, 1].set_ylabel('test MSE')
    axes[1, 1].set_title('Stability vs Test Error')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True)

    # generalization gap
    axes[1, 2].plot(lambdas, gen_gap, label='|test - train|', marker='o', color='brown')
    axes[1, 2].plot(lambdas, [2 * b for b in loss_stability],
                    label='2 * loss stability', marker='^', color='blue', linestyle='--')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_xlabel('lambda')
    axes[1, 2].set_ylabel('generalization gap')
    axes[1, 2].set_title('Generalization Gap vs Lambda')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{fname}_main.png', dpi=150)
    plt.close()
    print(f'saved: plots/{fname}_main.png')

    return prediction_stability, test_errors, train_errors


def run_comparison_plot(results_synthetic, results_diabetes, lambdas):
    stab_syn, test_syn, train_syn = results_synthetic
    stab_dia, test_dia, train_dia = results_diabetes

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Comparison: Synthetic vs Diabetes Dataset', fontsize=13)

    axes[0].plot(lambdas, test_syn, label='synthetic test error', marker='o', color='blue')
    axes[0].plot(lambdas, test_dia, label='diabetes test error', marker='s', color='orange')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('lambda')
    axes[0].set_ylabel('test MSE')
    axes[0].set_title('Test Error vs Lambda')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(lambdas, stab_syn, label='synthetic stability', marker='^', color='blue')
    axes[1].plot(lambdas, stab_dia, label='diabetes stability', marker='v', color='orange')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('lambda')
    axes[1].set_ylabel('mean prediction change (lower = more stable)')
    axes[1].set_title('Stability vs Lambda')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/comparison.png', dpi=150)
    plt.close()
    print('saved: plots/comparison.png')


if __name__ == '__main__':
    lambdas = np.logspace(-3, 3, 20)

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
    res_syn = run_main_experiment(X_train, y_train, X_test, y_test,
                                  lambdas, 'Synthetic Dataset', 'synthetic')

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
    res_dia = run_main_experiment(X2_train, y2_train, X2_test, y2_test,
                                  lambdas, 'Diabetes Dataset', 'diabetes')

    print("=== Comparison Plot ===")
    run_comparison_plot(res_syn, res_dia, lambdas)