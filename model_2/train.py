from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from common import SEED, train_classifier


def make_estimator(*, n_estimators=28, max_depth=22, min_samples_leaf=6):
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf),
        n_estimators=n_estimators, n_jobs=3, random_state=SEED,
    )


if __name__ == "__main__":
    train_classifier("Bagged trees", make_estimator(), budget="all", prior_power=0.65)
