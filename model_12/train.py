from sklearn.cluster import AgglomerativeClustering

from common import train_clustering


if __name__ == "__main__":
    model = AgglomerativeClustering(n_clusters=2, linkage="average", metric="euclidean")
    train_clustering("Hierarchical clustering", "hierarchical_clusters", model)
