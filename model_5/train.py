from sklearn.naive_bayes import GaussianNB

from common import train_classifier


if __name__ == "__main__":
    train_classifier(
        "Gaussian Naive Bayes", GaussianNB(var_smoothing=0.02),
        pool="expanded", budget="medium", variant="pca", prior_power=1.0,
    )
