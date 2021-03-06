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

pca = PCA(n_components=d_max)

# Subset.
np.random.seed(123)
idx = np.random.permutation(X_full.shape[0])
X = X_full[idx[0:n], :]

# Fit on training data and time.

tic = time.time()
pca.fit(scaler.transform(X))
toc = time.time()

print("Fit time: {:}.".format(format_time(toc - tic)))

cumulative = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative += var
    if cumulative > 0.90:
        print(
            "%0.3f percent of variance explained by first %d dimensions."
            % (cumulative, i)
        )
        n_components = i
        break


# Apply to validation and test data.

pca = PCA(n_components=n_components).fit(scaler.transform(X))

tic = time.time()

Z_train = pca.transform(scaler.transform(X_full))
Z_val = pca.transform(scaler.transform(np.load("data/X_val.npy")))
Z_test = pca.transform(scaler.transform(np.load("data/X_test.npy")))

toc = time.time()

with open("data/Z_train.npy", "wb") as f:
    np.save(f, Z_train)
with open("data/Z_val.npy", "wb") as f:
    np.save(f, Z_val)
with open("data/Z_test.npy", "wb") as f:
    np.save(f, Z_test)

print("Transform time: {:}.".format(format_time(toc - tic)))

