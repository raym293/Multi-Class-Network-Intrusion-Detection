from sklearn.tree import DecisionTreeClassifier

from common import SEED, train_classifier


if __name__ == "__main__":
    model = DecisionTreeClassifier(max_depth=13, min_samples_leaf=6, random_state=SEED)
    train_classifier(
        "CART", model, pool="expanded", budget="medium", fit_power=0.35,
        prior_power=[0.0, 0.0, 0.0, 0.18, 0.0, 0.0, 0.0],
    )
