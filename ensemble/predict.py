import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

from common import FEATURES, LABELS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict intrusion labels for a flow CSV.")
    parser.add_argument("flows", type=Path, help="CSV with the study's flow-feature columns")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("ensemble.joblib"))
    args = parser.parse_args()

    flows = pd.read_csv(args.flows, usecols=FEATURES)
    model = joblib.load(args.model)
    labels = [LABELS[int(code)] for code in model.predict(flows)]
    pd.DataFrame({"prediction": labels}).to_csv(sys.stdout, index=False)
