from sklearn.cluster import KMeans

from common import SEED, train_clustering


if __name__ == "__main__":
    train_clustering("K-means", "kmeans", KMeans(n_clusters=7, n_init=12, random_state=SEED))
