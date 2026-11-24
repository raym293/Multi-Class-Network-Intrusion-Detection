from sklearn.neighbors import KNeighborsClassifier

from common import train_classifier


if __name__ == "__main__":
    model = KNeighborsClassifier(n_neighbors=13, weights="distance", algorithm="brute", n_jobs=3)
    train_classifier("k-NN", model, budget="all", scale=True, prior_power=0.65)
