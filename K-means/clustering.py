import numpy as np
import random

class k_means:
    def __init__(self, n_clusters) -> None:
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.distortion_ = None

    def fit(self, data):
        all_centroids = []
        all_distortions = []
        all_labels = []
        
        # random initializaion
        for i in range(100):
            # Random unique centroid for every cluster
            indices = random.sample(range(0, len(data)), self.n_clusters)
            centroids = data[indices]

            while True:
                # distances between points and clusters' centroids
                distances = []
                for cluster in range(self.n_clusters):
                    distances.append(np.linalg.norm(
                        data-centroids[cluster], axis=1))
                distances = np.array(distances)

                # cluster of every point
                data_clusters = np.argmin(distances, axis=0)
                data_with_clusters = np.concatenate(
                    (data, data_clusters.reshape(-1, 1)), axis=1)

                # new centroids
                new_centroids = []
                for cluster in range(self.n_clusters):
                    # filter points of each cluster
                    points = np.array(list(filter(lambda x: x[-1] == cluster, data_with_clusters)))[:, :-1]
                    new_centroids.append(points.mean(0))

                new_centroids = np.array(new_centroids)

                if np.array(new_centroids == centroids).all():
                    break

                centroids = new_centroids

            # distortion
            distortion = 0
            for cluster in range(self.n_clusters):
                # filter points of each cluster
                points = np.array(list(filter(lambda x: x[-1] == cluster, data_with_clusters)))[:, :-1]
                distortion += sum(np.linalg.norm(points - centroids[cluster], axis=1))

            all_centroids.append(centroids)
            all_distortions.append(distortion)
            all_labels.append(data_clusters)

        least_distortion = np.argmin(all_distortions)
        self.distortion_ = min(all_distortions)
        self.cluster_centers_ = np.array(all_centroids[least_distortion])
        self.labels_ = np.array(all_labels[least_distortion])

    def predict(self, x):
        distances = []
        for centroid in self.cluster_centers_:
            distances.append(np.linalg.norm(x-centroid, axis=1))
        distances = np.array(distances)
        label = np.argmin(distances, axis=0)
        return label
