from sklearn.linear_model import LogisticRegression

from common import train_classifier


if __name__ == "__main__":
    model = LogisticRegression(C=7.800000000000001, max_iter=1600, tol=0.0008)
    train_classifier("Logistic regression", model, scale=True, prior_power=0.05)
