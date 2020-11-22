# %%

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from NSW import NSW

# import matplotlib.pyplot as plt

# %%

data = np.load("data/plarge300.npy", allow_pickle=True).item()
X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.5, random_state=1992)

# %%

param_dict = {
    'f': np.arange(3, 11, 2),
    'm': np.arange(3, 21, 2),
    'k': np.arange(1, 10, 2)
}
param_dict

# %%

# nsw = NSW()
# gscv = GridSearchCV(nsw, param_dict, cv=10, scoring="accuracy", n_jobs=-1)
# gscv.fit(X_train, y_train)
# print(gscv.best_params_)
# print(gscv.best_score_)
# %%
best_param = {'f': 3, 'k': 1, 'm': 17}
best_score = 0.7814285714285715
print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)

# %%

nsw = NSW(**best_param)
nsw.fit(X_train, y_train)
y_hat = nsw.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy: ", round(acc, 2))
# %%
