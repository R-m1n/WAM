import copy
import itertools

import numpy as np
import ruptures as rpt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score
from scipy.ndimage import gaussian_filter1d


class BlockCluster:
                       
    def __init__(self, average_method='max'):

        self.average_method = average_method

        self.nmi_matrix = np.ndarray

        self.n_features = 0

        self.best_params = dict()

        self.clusters = list()

        self.scoring = {
            'intra_inter_ratio': self.__intra_inter_ratio,
            'silhouette_score': self.__silhouette_score,
        }

        self.__is_fitted = False
        
        self.__smoothed_feature_nmi_cache = dict()


    def fit(self, X, y=None, scoring='intra_inter_ratio', **params):
        assert X.ndim == 2, f"Input dimention should be 2 not {X.ndim}"
        assert scoring in self.scoring, f"Unknown scoring function: {scoring}, choose between: {self.scoring.keys()}"

        if not params:
            params = {
                'smoothing_window_size': [3, 5, 7],
                'max_clusters': range(4, 33),
                'min_cluster_size': [4]
            }

        _, self.n_features = X.shape

        self.__nmi_matrix(X)

        self.__grid_search(scoring, **params)

        self.__is_fitted = True

        return self
    

    def transform(self, X):
        assert self.__is_fitted, 'Unfitted Model!!!'

        n_samples, _ = X.shape

        transformed = np.zeros((n_samples, len(self.clusters)), dtype=np.float64)

        for col, cluster in enumerate(self.clusters):
            start, *_, end = cluster

            bits = X[:, start: end + 1]

            transformed[:, col] = np.mean(bits, axis=1)
            
        return transformed
    

    def fit_transform(self, X, y=None, scoring='intra_inter_ratio'):

        self.fit(X, scoring)

        return self.transform(X)
    

    def plot_nmi_matrix(self):
        assert self.__is_fitted, 'Unfitted Model!!!'

        plt.figure(figsize=(12, 8))

        sns.heatmap(self.nmi_matrix, cmap='terrain', square=True)

        plt.title(f"{self.average_method.capitalize()} Normalized Mutual Information Matrix", fontsize=14)

        plt.tight_layout()
        plt.show()


    def plot_clusters(self):
        assert self.__is_fitted, 'Unfitted Model!!!'

        plt.figure(figsize=(12, 8))

        sns.heatmap(self.nmi_matrix, cmap='terrain', square=True, cbar=True)

        for change_point, *_ in self.clusters:
            plt.axhline(change_point, color='white', linestyle='--', linewidth=1)
            plt.axvline(change_point, color='white', linestyle='--', linewidth=1)

        plt.title('NMI Matrix with Cluster Boundaries')

        plt.tight_layout()
        plt.show()


    def plot_change_points(self):
        assert self.__is_fitted, 'Unfitted Model!!!'

        smoothing_window = self.best_params['smoothing_window_size']

        smoothed_feature_nmi = self.__smoothed_feature_nmi(smoothing_window)

        _, ax = plt.subplots(2, 1, figsize=(12, 8))

        ax[0].plot(smoothed_feature_nmi, marker='o', color='navy')
        for cluster in self.clusters:
            ax[0].axvline(cluster[0], color='maroon', linestyle='--')

        ax[0].set_title("Change Points in Smoothed NMI per Feature")
        ax[0].set_xlabel("Feature")
        ax[0].set_ylabel(f"Smoothed NMI (Window = {smoothing_window})")

        smoothed_feature_nmi = gaussian_filter1d(smoothed_feature_nmi, sigma=1)
        ax[1].plot(smoothed_feature_nmi, marker='o', color='navy')
        for cluster in self.clusters:
            ax[1].axvline(cluster[0], color='maroon', linestyle='--')

        ax[1].set_title("Change Points in Gaussian-Smoothed NMI per Feature")
        ax[1].set_xlabel("Feature")
        ax[1].set_ylabel(f"Gaussian-Smoothed NMI (Window = {smoothing_window})")

        plt.tight_layout()
        plt.show()


    def __nmi_matrix(self, X):

        self.nmi_matrix = np.empty((self.n_features, self.n_features))

        for i in range(self.n_features):
            for j in range(i, self.n_features):
                nmi = normalized_mutual_info_score(X[:, i], 
                                                   X[:, j], 
                                                   average_method=self.average_method)

                self.nmi_matrix[i, j] = nmi
                self.nmi_matrix[j, i] = nmi


    def __smoothed_feature_nmi(self, smoothing_window):
        if smoothing_window in self.__smoothed_feature_nmi_cache:
            return self.__smoothed_feature_nmi_cache[smoothing_window]
        
        smoothed_feature_nmi = np.empty(self.n_features)

        for i in range(self.n_features):

            start = max(0, i - smoothing_window + 1)
            end = min(self.n_features, i + smoothing_window)

            if end - start != smoothing_window:
                start += end - start - smoothing_window

            neighbors = np.concatenate((np.arange(start, i), np.arange(i + 1, end)))

            smoothed_feature_nmi[i] = np.mean(self.nmi_matrix[i, neighbors])

        self.__smoothed_feature_nmi_cache[smoothing_window] = smoothed_feature_nmi

        return smoothed_feature_nmi


    def __change_point_clustering(
            self,
            max_clusters,
            smoothing_window=5,
            min_cluster_size=None,
            filter='gaussian'
        ):

        smoothed_feature_nmi = self.__smoothed_feature_nmi(smoothing_window)

        if filter:
            smoothed_feature_nmi = gaussian_filter1d(smoothed_feature_nmi, sigma=1)

        model = rpt.KernelCPD(kernel="linear").fit(smoothed_feature_nmi)

        change_points = [0] + model.predict(n_bkps=max_clusters - 1)

        clusters = []

        for i in range(len(change_points) - 1):
            clusters.append(list(range(change_points[i], change_points[i + 1])))

        if min_cluster_size:
            clusters = self.__agglomerate(clusters, min_cluster_size)

        return clusters
    

    def __agglomerate(self, clusters, min_cluster_size=4):

        clusters_ = copy.deepcopy(clusters)

        i = 0
        while i < len(clusters_):
            if len(clusters_[i]) < min_cluster_size:
                cluster = clusters_[i]

                if i == 0:
                    clusters_[i + 1] = cluster + clusters_[i + 1]

                elif i == len(clusters_) - 1:
                    clusters_[i - 1] += cluster

                else:
                    left_score = self.__cluster_cohesion_score(clusters_[i - 1] + cluster)
                    right_score = self.__cluster_cohesion_score(cluster + clusters_[i + 1])

                    if left_score < right_score:
                        clusters_[i + 1] = cluster + clusters_[i + 1]

                    else:
                        clusters_[i - 1] += cluster

                clusters_.pop(i)

                continue

            i += 1

        return clusters_
    

    def __cluster_cohesion_score(self, cluster):
        start, *_, end = cluster

        intra_nmi = np.triu(self.nmi_matrix[start: end + 1, start: end + 1], k=1)

        return np.sum(intra_nmi) / np.count_nonzero(intra_nmi)


    def __cluster_separation_score(self, reference_cluster, target_cluster):
        reference_start, *_, reference_end = reference_cluster
        target_start, *_, target_end = target_cluster

        inter_nmi = self.nmi_matrix[reference_start: reference_end + 1, target_start: target_end + 1]

        return np.mean(inter_nmi)


    def __silhouette_score(self, clusters):
        n_clusters = len(clusters)
        cohesion_scores = np.empty(n_clusters)

        for i, cluster in enumerate(clusters):
            cohesion_scores[i] = self.__cluster_cohesion_score(cluster)

        seperation_scores = np.empty(n_clusters)

        seperation_scores[0] = self.__cluster_separation_score(clusters[0], clusters[1])

        for i in range(1, n_clusters - 1):
            left_score = self.__cluster_separation_score(clusters[i], clusters[i - 1])
            right_score = self.__cluster_separation_score(clusters[i], clusters[i + 1])

            seperation_scores[i] = (left_score + right_score) / 2

        seperation_scores[-1] = self.__cluster_separation_score(clusters[-1], clusters[-2])

        silhouette_scores = (cohesion_scores - seperation_scores) / np.maximum(cohesion_scores, seperation_scores)

        cluster_lengths = np.array([len(cluster) for cluster in clusters])

        return np.average(silhouette_scores, weights=cluster_lengths)


    def __intra_inter_ratio(self, clusters):
        intra_sum = 0
        intra_count = 0

        for start, *_, end in clusters:
            intra_nmi = np.triu(self.nmi_matrix[start: end + 1, start: end + 1], k=1)

            intra_sum += np.sum(intra_nmi)
            intra_count += np.count_nonzero(intra_nmi)

        intra_mean = intra_sum / intra_count

        inter_sum = 0
        inter_count = 0

        for i in range(len(clusters)):
            reference_start, *_, reference_end = clusters[i]

            for j in range(i + 1, len(clusters)):
                target_start, *_, target_end = clusters[j]

                inter_nmi = self.nmi_matrix[reference_start: reference_end + 1, target_start: target_end + 1]

                inter_sum += np.sum(inter_nmi)
                inter_count += np.size(inter_nmi)

        inter_mean = inter_sum / inter_count

        return intra_mean / inter_mean
    

    def __grid_search(self, scoring='intra_iter_ratio', **params):
        max_score = -np.inf

        grid = itertools.product(
            params['smoothing_window_size'],
            params['max_clusters'],
            params['min_cluster_size']
        )

        for smoothing_window, max_clusters, min_cluster_size in grid:

            clusters = self.__change_point_clustering(
                max_clusters=max_clusters,
                smoothing_window=smoothing_window,
                min_cluster_size=min_cluster_size
            )

            score = self.scoring[scoring](clusters)

            if score > max_score:
                max_score = score

                self.clusters = clusters

                self.best_params = {
                    'max_clusters': max_clusters,
                    'smoothing_window_size': smoothing_window,
                    'min_cluster_size': min_cluster_size,
                    'score': round(score, 2),
                    'n_clusters': len(clusters)
                }
