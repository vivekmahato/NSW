# Imports
### Importing the required libraries and our NSW code file.
```python
import numpy as np
from sklearn.metrics import accuracy_score
from NSW import NSW
from sklearn.model_selection import train_test_split,GridSearchCV
```


```python
data = np.load("data/plarge300.npy",allow_pickle=True).item()
X_train, X_test, y_train, y_test = train_test_split(data["X"],data["y"], test_size=0.33, random_state=1992)
```


```python
param_dict = {
    'f' : np.arange(3,11,2),
    'm' : np.arange(3,21,2),
    'k' : np.arange(1,10,2)
}
param_dict
```




    {'f': array([3, 5, 7, 9]),
     'm': array([ 3,  5,  7,  9, 11, 13, 15, 17, 19]),
     'k': array([1, 3, 5, 7, 9])}




```python
nsw = NSW()
gscv = GridSearchCV(nsw, param_dict, cv=10, scoring="accuracy", n_jobs=-1)
gscv.fit(X_train, y_train)
print(gscv.best_params_)
print(gscv.best_score_)
```

    100%|██████████| 150/150 [00:00<00:00, 204.73it/s]


    Model is fitted with the provided data.





    GridSearchCV(cv=10, estimator=<NSW.NSW object at 0x7f97d61fb208>, n_jobs=-1,
                 param_grid={'f': array([3, 5, 7, 9]), 'k': array([1, 3, 5, 7, 9]),
                             'm': array([ 3,  5,  7,  9, 11, 13, 15, 17, 19])},
                 scoring='accuracy')




```python
best_param = {'f': 3, 'k': 1, 'm': 17}
best_score = 0.7814285714285715
print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)
```

    Best Parameters:  {'f': 3, 'k': 1, 'm': 17}
    Best Accuracy:  0.7814285714285715



```python
nsw = NSW(**best_param)
nsw.fit(X_train, y_train)
y_hat = nsw.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy: ", round(acc, 2))
```

    100%|██████████| 201/201 [00:01<00:00, 184.68it/s]
    100%|██████████| 100/100 [00:00<00:00, 106.39it/s]


    Model is fitted with the provided data.
    Model accuracy:  0.58



```python


```
