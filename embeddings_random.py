from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import datetime


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


X_full = np.load("data/X_train.npy")

n = X_full.shape[0]
d_max = X_full.shape[1]

scaler = StandardScaler().fit(X_full)

# Subset.
np.random.seed(123)
idx = np.random.permutation(X_full.shape[0])
X = X_full[idx[0:n], :]

n_orig = np.load("data/X_train.npy").shape[1]
n_proj = np.load("data/Z_train.npy").shape[1]

proj_mat = np.random.normal(size=(n_orig, n_proj))

Z_train = np.dot(scaler.transform(X_full), proj_mat)
Z_val = np.dot(scaler.transform(np.load("data/X_val.npy")), proj_mat)
Z_test = np.dot(scaler.transform(np.load("data/X_test.npy")), proj_mat)

with open("data/Z_train_rand.npy", "wb") as f:
    np.save(f, Z_train)
with open("data/Z_val_rand.npy", "wb") as f:
    np.save(f, Z_val)
with open("data/Z_test_rand.npy", "wb") as f:
    np.save(f, Z_test)

