from sklearn.ensemble import RandomForestClassifier

from common import SEED, train_classifier


def make_estimator():
    return RandomForestClassifier(
        n_estimators=64, max_depth=18, min_samples_leaf=3,
        max_features=0.6, n_jobs=3, random_state=SEED,
    )


if __name__ == "__main__":
    train_classifier(
        "Random Forest", make_estimator(), budget="all", variant="residual", prior_power=0.65,
    )
