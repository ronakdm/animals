import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle

job_id = 15
C_range = np.logspace(-5, 5, 30)

X_train, y_train = np.load("data/Z_train.npy"), np.load("data/y_train.npy")

model = LogisticRegression(C=C_range[job_id], max_iter=1000).fit(X_train, y_train)

X_val, y_val = np.load("data/Z_val.npy"), np.load("data/y_val.npy")
y_pred = model.predict(X_val)

results = {
    "macro_f1": f1_score(y_val, y_pred, average="macro"),
    "accuracy": accuracy_score(y_val, y_pred),
}

pickle.dump(results, open("out/result_%d.p" % job_id, "wb"))

