import numpy as np
from sklearn.model_selection import train_test_split


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
print("X_vak shape:", X_val.shape)
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

