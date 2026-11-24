from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from common import SEED, train_classifier


if __name__ == "__main__":
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=4),
        n_estimators=27, learning_rate=0.65, random_state=SEED,
    )
    train_classifier("AdaBoost", model, pool="expanded", budget="medium")
