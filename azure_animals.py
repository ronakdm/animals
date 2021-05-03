'''
Test script for DATA 558 on Azure.

Runs PCA + 1-NN on Animals dataset.
'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
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

def preprocess(downsample=0.1):
    with open(
        "data/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt", "rb"
    ) as f:
        lines = f.readlines()
        n = len(lines)

        np.random.seed(123)
        idx = np.random.permutation(n)

        step = int(1 / downsample)

        examples = []
        for i in range(0, n, step):
            examples.append(np.array([float(num) for num in lines[idx[i]].split()]))

        X = np.array(examples)

    with open(
        "data/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt", "rb"
    ) as f:
        lines = f.readlines()
        labels = []
        for i in range(0, n, step):
            labels.append(int(lines[idx[i]]))

        y = np.array(labels)

    return X, y

print('Preprocessing...')
X, y = preprocess(downsample=1)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

with open("data/X_train.npy", "wb") as f:
    np.save(f, X_train)
with open("data/y_train.npy", "wb") as f:
    np.save(f, y_train)
with open("data/X_val.npy", "wb") as f:
    np.save(f, X_val)
with open("data/y_val.npy", "wb") as f:
    np.save(f, y_val)
with open("data/X_test.npy", "wb") as f:
    np.save(f, X_test)
with open("data/y_test.npy", "wb") as f:
    np.save(f, y_test)


print('Dimensionality reduction via PCA...')
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

print("PCA complete. Transform time: {:}.".format(format_time(toc - tic)))

X_train, y_train = np.load("data/Z_train.npy"), np.load("data/y_train.npy")
X_val, y_val = np.load("data/Z_val.npy"), np.load("data/y_val.npy")

print("Running 1-NN on train set...")
k=1
tic = time.time()
model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_val)


res = {}
res['model'] = model
res['valid'] = { "macro_f1": f1_score(y_val, y_pred, average="macro"),
"accuracy": accuracy_score(y_val, y_pred)}
res['train'] = { "macro_f1": f1_score(y_train, y_pred_train, average="macro"),
"accuracy": accuracy_score(y_train, y_pred_train)}
toc = time.time()

print('k = {}, Time: {}'.format(k, format_time(toc-tic)))
print('Train results: {:.4f} accuracy || {:.4f} F1'.format(res['train']['accuracy'],
                                                 res['train']['macro_f1']))
print('Valid results: {:.4f} accuracy || {:.4f} F1'.format(res['valid']['accuracy'],
                                                 res['valid']['macro_f1']))

with open('./animals_1nn_res.pkl', 'wb') as f:
    pickle.dump(res, f)