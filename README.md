# Multi-Class Network Intrusion Detection

This project uses the CSE-CIC-IDS2018 dataset to classify network flows as benign or one of six attack types. We compare nine classifiers and also explore regression and clustering. The final model combines bagged trees and Random Forest.

## Run the project

Use Python 3.12. Put the original compressed data file at `dataset/archive.zip` and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -m model_1.train
python -m ensemble.train
```

Replace `model_1` with any folder from `model_1` through `model_12` to train that model. Training outputs go to `dataset/models/`.

To predict labels for a CSV with the dataset's flow-feature columns, use the saved ensemble.

```sh
python -m ensemble.predict flows.csv
```

## Results

On the full held-out data, macro-F1 went from **0.6466** for the original Random Forest to **0.7063** for the ensemble in our macOS run. See the [results summary](results/summary.md) and [benchmark notebook](benchmarking/intrusion_detection_frozen.ipynb) for the full comparison.
