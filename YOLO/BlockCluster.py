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

        self.scoring = {'intra_inter_ratio': self.__intra_inter_ratio,
                        'silhouette_score': self.__silhouette_score}
        
        self.__smoothed_feature_nmi_cache = dict()


    def fit(self, dataset, scoring='intra_inter_ratio', **params):
        assert scoring in self.scoring, f"Unknown scoring function: {scoring}"

        if not params:
            params = {'smoothing_window_size': [3, 5, 7], 'max_clusters': range(4, 33), 'min_cluster_size': [4]}

        self.n_samples, self.n_features = dataset.shape

        self.__compute_nmi_matrix(dataset, self.average_method)

        self.__grid_search(scoring, **params)

        return self
    

    def transform(self, dataset):
        assert self.clusters is not None, 'Unfitted Model!!!'

        transformed = np.zeros((self.n_samples, len(self.clusters)), dtype=np.uint64)

        for col, cluster in enumerate(self.clusters):
            bits = dataset[:, cluster]

            weights = np.power(2, np.arange(len(cluster) - 1, -1, -1)).reshape((-1, 1))

            transformed[:, col] = (bits @ weights).flatten()
            
        return transformed
    

    def fit_transform(self, dataset, scoring='intra_inter_ratio'):

        self.fit(dataset, scoring)

        return self.transform(dataset)
    

    def plot_nmi_matrix(self):
        assert self.nmi_matrix is not None, 'Unfitted Model!!!'

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.nmi_matrix, cmap='terrain', square=True)
        plt.title("Mutual Information Matrix", fontsize=14)
        plt.tight_layout()
        plt.show()


    def plot_clusters(self):
        assert self.nmi_matrix is not None, 'Unfitted Model!!!'
        assert self.clusters is not None, 'Unfitted Model!!!'

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.nmi_matrix, cmap='terrain', square=True, cbar=True)

        for pos, *_ in self.clusters:
            plt.axhline(pos, color='white', linestyle='--', linewidth=1)
            plt.axvline(pos, color='white', linestyle='--', linewidth=1)

        plt.title('NMI Matrix with Cluster Boundaries')

        plt.tight_layout()
        plt.show()


    def plot_change_points(self):
        assert self.best_params is not None, 'Unfitted Model!!!'

        smoothing_window = self.best_params['smoothing_window_size']

        smoothed_feature_nmi = self.__compute_smoothed_feature_nmi(smoothing_window)

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        ax[0].plot(smoothed_feature_nmi, marker='o', color='navy')
        for cluster in self.clusters:
            ax[0].axvline(cluster[0], color='maroon', linestyle='--')

        ax[0].set_title("Smoothed NMI per Feature")
        ax[0].set_xlabel("Feature")
        ax[0].set_ylabel(f"NMI (Window = {smoothing_window})")

        smoothed_feature_nmi = gaussian_filter1d(smoothed_feature_nmi, sigma=1)
        ax[1].plot(smoothed_feature_nmi, marker='o', color='navy')
        for cluster in self.clusters:
            ax[1].axvline(cluster[0], color='maroon', linestyle='--')

        ax[1].set_title("Gaussian-Smoothed NMI per Feature")
        ax[1].set_xlabel("Feature")
        ax[1].set_ylabel(f"NMI (Window = {smoothing_window})")

        plt.tight_layout()
        plt.show()


    def __compute_nmi_matrix(self, dataset, average_method):

        self.nmi_matrix = np.empty((self.n_features, self.n_features))

        for i in range(self.n_features):
            for j in range(i, self.n_features):
                nmi = normalized_mutual_info_score(dataset[:, i], dataset[:, j], average_method=average_method)

                self.nmi_matrix[i, j] = nmi
                self.nmi_matrix[j, i] = nmi


    def __compute_smoothed_feature_nmi(self, smoothing_window):
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


    def __change_point_clustering(self, max_clusters, smoothing_window=5, min_cluster_size=None, filter='gaussian'):

        smoothed_feature_nmi = self.__compute_smoothed_feature_nmi(smoothing_window)

        if filter:
            smoothed_feature_nmi = gaussian_filter1d(smoothed_feature_nmi, sigma=1)

        model = rpt.KernelCPD(kernel="linear").fit(smoothed_feature_nmi)

        change_points = [0] + model.predict(n_bkps=max_clusters - 1)

        clusters = []

        for i in range(len(change_points) - 1):
            start_point, end_point = change_points[i], change_points[i + 1]

            clusters.append(list(range(start_point, end_point)))

        if min_cluster_size:
            clusters = self.__agglomerate(clusters, min_cluster_size)

        return clusters
    

    def __agglomerate(self, clusters, min_cluster_size=4):

        clusters = clusters[:]

        i = 0
        while i < len(clusters):
            if len(clusters[i]) < min_cluster_size:
                cluster = clusters[i]

                if i == 0:
                    clusters[i + 1] = cluster + clusters[i + 1]

                elif i == len(clusters) - 1:
                    clusters[i - 1] += cluster

                else:
                    left_score = self.__cluster_cohesion_score(clusters[i - 1] + cluster)
                    right_score = self.__cluster_cohesion_score(cluster + clusters[i + 1])

                    if left_score < right_score:
                        clusters[i + 1] = cluster + clusters[i + 1]

                    else:
                        clusters[i - 1] += cluster

                clusters.pop(i)

                continue

            i += 1

        return clusters
    

    def __cluster_cohesion_score(self, cluster):
        start_point, *_, end_point = cluster

        upper_triangle = np.triu(self.nmi_matrix[start_point: end_point + 1, start_point: end_point + 1], k=1)

        return np.sum(upper_triangle) / np.count_nonzero(upper_triangle)


    def __cluster_separation_score(self, reference_cluster, target_cluster):
        reference_start, *_, reference_end = reference_cluster
        target_start, *_, target_end = target_cluster

        window = self.nmi_matrix[reference_start: reference_end + 1, target_start: target_end + 1]

        return np.mean(window)


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

        for start_point, *_, end_point in clusters:
            upper_triangle = np.triu(self.nmi_matrix[start_point: end_point + 1, start_point: end_point + 1], k=1)

            intra_sum += np.sum(upper_triangle)
            intra_count += np.count_nonzero(upper_triangle)

        intra_mean = intra_sum / intra_count

        inter_sum = 0
        inter_count = 0

        for i in range(len(clusters)):
            reference_start, *_, reference_end = clusters[i]

            for j in range(i + 1, len(clusters)):
                target_start, *_, target_end = clusters[j]

                window = self.nmi_matrix[reference_start: reference_end + 1, target_start: target_end + 1]

                inter_sum += np.sum(window)
                inter_count += np.size(window)

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
