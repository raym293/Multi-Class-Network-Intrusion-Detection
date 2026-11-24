from sklearn.dummy import DummyClassifier

from common import train_classifier


if __name__ == "__main__":
    train_classifier("Majority", DummyClassifier(strategy="most_frequent"))
