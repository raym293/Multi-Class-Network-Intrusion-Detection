from sklearn.svm import LinearSVC

from common import SEED, train_classifier


if __name__ == "__main__":
    model = LinearSVC(C=1, dual=False, max_iter=2000, random_state=SEED)
    train_classifier("Linear SVM", model, variant="residual", scale=True)
