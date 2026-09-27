"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    col_means = np.nanmean(X, axis=0, keepdims=True)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    np.putmask(X, np.isnan(X), col_means)

    return X

# Step 2 - compute_iqr_bounds
import numpy as np

def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return (lower, upper)

# Step 3 - clip_columns
import copy
import numpy as np

def clip_columns(X, lower, upper):
    X_output = copy.copy(X)

    lower_mask = (X < lower)
    upper_mask = (X > upper)

    np.putmask(X_output, lower_mask, lower)
    np.putmask(X_output, upper_mask, upper)
    
    return X_output

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X_output = copy.copy(X)

    X_output = np.hstack([X_output, col.reshape(-1, 1)])

    return X_output

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    unique_cat = np.unique(labels)
    N = len(labels)
    C = len(unique_cat)
    
    indices = np.searchsorted(unique_cat, labels)
    
    result = np.zeros(shape=(N, C), dtype=float)
    
    result[np.arange(N), indices] = 1.0
    
    return result

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    
    X_std = np.where(X_std == 0, 1.0, X_std)
    
    return (X_mean, X_std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    N, F = X.shape

    return np.hstack([np.ones(shape=(N, 1)), X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    np.random.seed(seed)

    return np.random.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    N = len(indices)

    first_border = int(N * train_ratio)
    second_border = int(first_border + N * val_ratio)
    train_idx = indices[:first_border]
    val_idx = indices[first_border:second_border]
    test_idx = indices[second_border:]

    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    theta, residials, rank, s = np.linalg.lstsq(X, y, rcond=None)

    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_true - y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    SS_res = np.sum((y_true - y_pred) ** 2)

    if SS_tot == 0:
        return 0.0

    return 1 - SS_res / SS_tot

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    residials = y_true - y_pred 

    return {
        'mean' : float(np.mean(residials)),
        'std' : float(np.std(residials)),
        'median_abs' : float(np.median(np.abs(residials)))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    impute_nan_with_mean(X)

    lower, upper = compute_iqr_bounds(X, iqr_k)

    return clip_columns(X, lower, upper)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    new_ratio_feature = make_ratio_feature(numerator, denominator)
    X_num = append_column(X_num, new_ratio_feature)

    if cat_labels is not None:
        one_hot_cols = one_hot_encode(cat_labels)
        X_num = np.hstack([X_num, one_hot_cols])

    return X_num

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    n_samples, n_features = X.shape

    indices = make_shuffled_indices(n_samples, seed)
    train_idx, val_idx, test_idx = partition_indices(indices, train_ratio, val_ratio)

    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)

    return {
        'X_train' : X_train,
        'y_train' : y_train,
        'X_val' : X_val,
        'y_val' : y_val,
        'X_test' : X_test,
        'y_test' : y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    keys = ['X_train', 'X_val', 'X_test']

    std_splits = {}
    for key, value in splits.items():
        if key not in keys:
            std_splits[key] = value


    X_mean, X_std = fit_standardizer(splits['X_train'])
    for key in keys:
        st_data = apply_standardizer(splits[key], X_mean, X_std)
        st_and_bias_data = add_bias_column(st_data)

        std_splits[key] = st_and_bias_data

    return (std_splits, X_mean, X_std)

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    return {
        'mae' : mean_absolute_error(y_true, y_pred),
        'rmse' : root_mean_squared_error(y_true, y_pred),
        'r2' : r_squared(y_true, y_pred),
        'residual_summary' : residual_summary(y_true, y_pred)
    }

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

