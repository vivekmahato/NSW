import numpy as np
from sortedcollections import ValueSortedDict
from collections import Counter
from tqdm import tqdm
import math
from tslearn.metrics import dtw


def most_frequent(l: list):
    freq = Counter(l)
    return freq.most_common(1)[0][0]


class Node:
    def __init__(self, index: int, values: list, label=None):
        self.index = index
        self.values = values
        self.label = label
        self.neighbors = ValueSortedDict()

    def connect(self, index, cost, f):
        """
        Calculate distance and store in a sorteddict
        """
        # The dict would be sorted by values
        self.neighbors[index] = cost
        while len(self.neighbors) > f:
            self.neighbors.popitem()
        return self


class nsw:
    def __init__(self,
                 f: int = 1,
                 m: int = 1,
                 k: int = 1,
                 metric: object = "euclidean",
                 metric_params: dict = {},
                 random_seed: int = 1992) -> object:

        self.seed = random_seed
        self.f = f
        self.m = m
        self.k = k
        self.euclidean = metric
        self.metric_params = metric_params
        self.corpus = {}

    def get_params(self, deep=True):
        return {"f": self.f,
                "m": self.m,
                "k": self.k,
                "metric": self.metric,
                "metric_params": self.metric_params,
                "random_seed": self.seed}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def switch_metric(self, ts1=None, ts2=None):
        if self.metric == "euclidean":
            return np.linalg.norm(ts1 - ts2)
        elif self.metric == "dtw":
            return dtw(ts1, ts2, **self.metric_params)
        return None

    def nn_insert(self, index=int, values=[], label=None):
        # create node with the given values
        node = Node(index, values, label)

        # check if the corpus is empty
        if len(self.corpus) < 1:
            self.corpus[node.index] = node
            return self

        neighbors, count = self.knn_search(node, self.f)

        for key, cost in list(neighbors.items())[:self.f]:
            # have the store the updated node back in the corpus
            neighbor = self.corpus[key]
            assert neighbor.index == key
            neighbor = neighbor.connect(node.index, cost, self.f)
            self.corpus[neighbor.index] = neighbor

            # connect new node to its neighbor
            node = node.connect(neighbor.index, cost, self.f)

        # storing new node in the corpus
        self.corpus[node.index] = node
        return self

    def batch_insert(self, indices=[]):

        for i in tqdm(list(range(self.X_train.shape[0]))):
            self.nn_insert(indices[i], self.X_train[i], self.y_train[i])

        return self

    def get_closest(self):
        k = next(iter(self.candidates))
        return {k: self.candidates.pop(k)}

    def check_stop_condition(self, c, k):
        # if c is further than the kth element in the result

        k_dist = self.result[list(self.result.keys())[k - 1]]
        c_dist = list(c.values())[0]

        if c_dist > k_dist:
            return True
        else:
            return False

    def knn_search(self, q=None, k=1):

        self.q = q
        self.visitedset = set()
        self.candidates = ValueSortedDict()
        self.result = ValueSortedDict()
        count = 0

        for i in range(self.m):
            v_ep = self.corpus[np.random.choice(list(self.corpus.keys()))]
            if self.dmat is None:
                cost = self.switch_metric(self.q.values, v_ep.values)
            else:
                cost = self.dmat[q.index][v_ep.index]
            count += 1
            self.candidates[v_ep.index] = cost
            self.visitedset.add(v_ep.index)
            tempres = ValueSortedDict()

            while True:

                # get element c closest from candidates to q, and remove c
                # from candidates
                if len(self.candidates) > 0:
                    c = self.get_closest()
                else:
                    break

                # check stop condition
                if len(self.result) >= k:
                    if self.check_stop_condition(c, k):
                        break

                # add neighbors of c if not in visitedset
                cand = self.corpus[list(c.keys())[0]]
                if (not cand.neighbors) or (cand.index in self.visitedset):
                    break
                else:
                    for key in list(cand.neighbors.keys()):
                        if key not in self.visitedset:
                            if self.dmat is None:
                                cost = self.switch_metric(self.q.values, v_ep.values)
                            else:
                                cost = self.dmat[q.index][v_ep.index]
                            count += 1
                            self.visitedset.add(key)
                            self.candidates[key] = cost
                            tempres[key] = cost

            # add tempres to result
            self.result.update(tempres)
        # return k neighbors/result
        return self.result, count

    def transform(self, s: int = 10) -> object:
        visited_set = set()
        for key, node in self.corpus.items():
            for nn, cost in node.neighbors.items():

                if key in visited_set and nn in visited_set:
                    continue

                neighbor = self.corpus[nn]
                snn = len(set(list(node.neighbors.keys())[:s])
                          .intersection(set(list(neighbor.neighbors.keys())[:s])))
                simcos = snn / float(s)
                dist = math.acos(simcos)

                # simcorr = (self.X_train.shape[0]/(self.X_train.shape[0]-s))*((snn/s)*(s/self.X_train.shape[0]))
                # dist = simcorr

                self.corpus[key].neighbors[nn] = dist
                self.corpus[nn].neighbors[key] = dist
                visited_set.add(key)
                visited_set.add(nn)
        return self

    def fit(self, X_train, y_train, secondary_metric=False, s=10, dist_mat=None):
        np.random.seed(self.seed)

        self.X_train = X_train.astype("float32")
        self.y_train = y_train
        self.dmat = dist_mat

        indices = np.arange(len(X_train))
        self.batch_insert(indices)

        print("Model is fitted with the provided data.")
        if secondary_metric:
            return self.transform(s=s)
        return self

    def predict(self, X_test):
        self.X_test = X_test.astype("float32")

        y_hat = []

        for i in tqdm(range(len(self.X_test))):
            q_node = Node(0, self.X_test[i], None)

            neighbors, count = self.knn_search(q_node, self.k)

            labels = [self.corpus[key].label for key in
                      list(neighbors.keys())[:self.k]]

            label = most_frequent(labels)

            y_hat.append(label)

        return y_hat

    def kneighbors(self, X_test=None, indices=[], dist_mat=None, return_prediction=False):
        self.X_test = X_test.astype("float32")
        self.dmat = dist_mat
        all_nns = []
        preds = []
        counts = []

        for i in tqdm(range(self.X_test.shape[0])):
            q_node = Node(indices[i], self.X_test[i], None)
            neighbors, count = self.knn_search(q_node, self.k)
            counts.append(count)
            neighbors = list(neighbors.keys())[:self.k]
            if return_prediction:
                preds.append(most_frequent(self.y_train[neighbors]))
            all_nns.append(neighbors)
        if return_prediction:
            return all_nns, preds, counts
        else:
            return all_nns

