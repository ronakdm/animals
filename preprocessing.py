import numpy as np

# TODO: download data.


def load_data(downsample=0.1):
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


X, y = load_data(downsample=0.1)

print(X.shape)
print(y.shape)

