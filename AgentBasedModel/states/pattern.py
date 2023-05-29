from .states_pattern import StateIdentification

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import DBSCAN


class PatternDetection:
    def set_params(self, local_eps: float = 2, local_min_samples: int = 5, local_pattern_size: int = 10, common_eps: float = 1, common_min_samples: int = 5):
        self.local_eps = local_eps
        self.local_min_samples = local_min_samples
        self.size = local_pattern_size
        self.common_eps = common_eps
        self.common_min_samples = common_min_samples

    def make_local_params(self) -> np.array:
        return np.arange(0.5 * self.local_eps, 5 * self.local_eps, self.local_eps * 0.1)

    def make_common_params(self) -> np.array:
        return np.arange(0.5 * self.common_eps, 10 * self.common_eps, self.common_eps * 0.25)


    def get_local_windows(self, states_pattern: StateIdentification, dbs_params: list = None) -> np.array:
        _, states_array = states_pattern.states()
        X = []
        for row in sliding_window_view(states_array, self.size):
            if (row == 1).all() or (row == -1).all():
                continue
            X.append(row)
        return np.array(X)
    
    def find_patterns(self, data, epsilons = None, type: str = 'local') -> list:
        if epsilons is None:
            if type == 'local':
                epsilons = [self.local_eps]
            else:
                epsilons = [self.common_eps]
        if type == 'local':
            min_samples = self.local_min_samples
        else:
            min_samples = self.common_min_samples

        n_clusters = 0
        n_noise = 0
        for eps in epsilons:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
            labels = np.array(db.labels_)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            if n_clusters > 2 and n_noise != len(data):
                break
        print(n_clusters, n_noise)
        if n_clusters <= 1 or n_noise == len(data):
            print('No patterns detected. Change search parameters.')
            return [], []

        patterns = list()
        indexes = list()
        for cluster_idx in range(n_clusters):
            pattern = np.mean(data[np.argwhere(labels == cluster_idx)], axis=0)[0]
            indexes.append(np.argwhere(labels == cluster_idx))
            for i in range(len(pattern)):
                if pattern[i] < -20:
                    pattern[i] = -100
                elif pattern[i] <= -0.1:
                    pattern[i] = -1
                elif pattern[i] >= 0.1:
                    pattern[i] = 1
                else:
                    pattern[i] = 0
            patterns.append(pattern)
        return patterns, indexes
