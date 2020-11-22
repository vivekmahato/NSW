# Imports
### Importing the required libraries and our NSW code file.
```python
import numpy as np
from sklearn.metrics import accuracy_score
from NSW import NSW
from sklearn.model_selection import train_test_split,GridSearchCV
```

# Load Data
## Loading data and splitting it into train and test sets.
```python
data = np.load("data/plarge300.npy",allow_pickle=True).item()
X_train, X_test, y_train, y_test = train_test_split(data["X"],data["y"], test_size=0.5, random_state=1992)
```

# Parameter Search
## We would be using GridSearchCV from sklearn to find the best set of parameters for the model on the train set.
```python
#Creating the parameter space
param_dict = {
    'f' : np.arange(3,11,2),
    'm' : np.arange(3,21,2),
    'k' : np.arange(1,10,2)
}
print(param_dict)
```




    {'f': array([3, 5, 7, 9]),
     'm': array([ 3,  5,  7,  9, 11, 13, 15, 17, 19]),
     'k': array([1, 3, 5, 7, 9])}




```python
#Use GridSearchCV on NSW model with the supplied parameter space
nsw = NSW()
gscv = GridSearchCV(nsw, param_dict, cv=10, scoring="accuracy", n_jobs=-1)
gscv.fit(X_train, y_train)
best_param = gscv.best_params_ # best set of parameters
best_score =  gscv.best_score_ # best accuracy score
```

    100%|██████████| 150/150 [00:00<00:00, 224.81it/s]


    Model is fitted with the provided data.
    
```python
print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)
```

    Best Parameters:  {'f': 3, 'k': 7, 'm': 17}
    Best Accuracy:  0.82


# Model Evaluation
## Supply the best set of parameters to the NSW model: train with Train set, and test on held-out test set.
```python
nsw = NSW(**best_param)
nsw.fit(X_train, y_train)
y_hat = nsw.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy: ", round(acc, 2))
```

    100%|██████████| 150/150 [00:00<00:00, 238.86it/s]
    100%|██████████| 151/151 [00:01<00:00, 141.00it/s]


    Model is fitted with the provided data.
    Model accuracy:  0.75



```python


```
