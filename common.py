"""Shared data preparation, features, and fitting for the intrusion models."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


CHUNK = 125_000


BLOCK_MINUTES = 10


SEED = 1729


FEATURES = [
    "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std", "Fwd PSH Flags",
    "Fwd Header Len", "Bwd Header Len", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "Down/Up Ratio", "Pkt Len Mean",
    "Pkt Len Std", "Init Fwd Win Byts", "Init Bwd Win Byts", "Active Mean",
    "Active Std", "Idle Mean", "Idle Std",
]


LABELS = ["Benign", "Bot", "Brute Force", "DoS", "DDoS", "Infiltration", "Web Attack"]


IDENTITY_COLUMNS = ["Timestamp", "Label"]


DATA = {'capture_members': {'02-14-2018.csv': [358223333, 1195817239],
                     '02-15-2018.csv': [375945899, 3870072833],
                     '02-16-2018.csv': [333723605, 395830762],
                     '02-20-2018.csv': [4054925350, 3119562513],
                     '02-21-2018.csv': [328893673, 1370206535],
                     '02-22-2018.csv': [382636202, 3854324905],
                     '02-23-2018.csv': [382840456, 2499666463],
                     '02-28-2018.csv': [209249758, 994541139],
                     '03-01-2018.csv': [107842858, 251771788],
                     '03-02-2018.csv': [352368373, 3489003198]},
 'time_blocks_by_bucket': {'02-14-2018': [[7, 9, 10, 25, 54, 65, 67, 70, 71, 73],
                                          [11, 13, 16, 23, 27, 30, 33, 50, 55, 58, 62, 64, 69, 72, 74],
                                          [8, 14, 19, 20, 21, 22, 26, 28, 51, 52, 53, 56, 59, 63, 75, 76],
                                          [6, 12, 15, 17, 18, 24, 29, 31, 32, 57, 60, 61, 66, 68, 77]],
                           '02-15-2018': [[11, 13, 17, 24, 32, 33, 51, 54, 62, 70, 75, 76],
                                          [14, 22, 28, 31, 34, 35, 50, 57, 58, 59, 60, 65, 66, 67, 68, 71],
                                          [6, 15, 19, 20, 21, 23, 25, 27, 52, 55, 61, 69],
                                          [7, 8, 9, 10, 12, 16, 18, 26, 29, 30, 53, 56, 63, 64, 72, 73, 74, 77]],
                           '02-16-2018': [[6, 9, 60, 61, 63, 64, 66, 68, 73, 74, 77],
                                          [52, 55, 57, 58, 69, 70, 72],
                                          [7, 51, 56, 59, 62, 65, 67, 71],
                                          [8, 10, 50, 53, 54, 75, 76]],
                           '02-20-2018': [[7, 9, 10, 18, 19, 23, 25, 33, 36, 37, 38, 51, 55, 56, 58, 67, 75],
                                          [6, 8, 16, 28, 30, 32, 48, 49, 52, 61, 62, 66, 68, 69, 73, 77],
                                          [11,
                                           17,
                                           20,
                                           24,
                                           27,
                                           29,
                                           34,
                                           35,
                                           39,
                                           43,
                                           45,
                                           46,
                                           47,
                                           50,
                                           54,
                                           57,
                                           60,
                                           63,
                                           64,
                                           71,
                                           76],
                                          [12, 13, 14, 15, 21, 22, 26, 31, 40, 41, 42, 44, 53, 59, 65, 70, 72, 74]],
                           '02-21-2018': [[15, 58],
                                          [11, 12, 52, 57, 64],
                                          [14, 55, 56, 59, 61, 62],
                                          [13, 51, 53, 54, 60, 63]],
                           '02-22-2018': [[11, 15, 22, 24, 26, 52, 60, 62, 70, 75, 76, 77],
                                          [7, 8, 16, 17, 19, 29, 30, 51, 59, 67, 69, 71, 72, 73],
                                          [6, 9, 18, 20, 27, 28, 50, 54, 56, 57, 61, 63, 64, 68, 74],
                                          [10, 12, 13, 14, 21, 23, 25, 31, 32, 33, 53, 55, 58, 65, 66]],
                           '02-23-2018': [[16, 17, 23, 24, 26, 28, 31, 52, 57, 59, 62, 69, 70, 71, 75],
                                          [7, 8, 9, 11, 18, 22, 25, 30, 49, 53, 54, 56, 60, 61, 63, 68, 73, 77],
                                          [6, 10, 12, 13, 14, 19, 20, 21, 27, 32, 64, 72],
                                          [15, 29, 33, 50, 51, 55, 58, 65, 66, 67, 74, 76]],
                           '02-28-2018': [[7, 9, 10, 11, 13, 23, 24, 31, 34, 55, 59, 61, 71, 72, 76],
                                          [12, 15, 16, 17, 18, 21, 22, 25, 33, 35, 53, 54, 60, 62, 68, 70, 75],
                                          [14, 27, 28, 29, 32, 52, 56, 57, 58, 65, 66, 69],
                                          [6, 8, 19, 20, 26, 30, 50, 51, 63, 64, 67, 73, 74, 77]],
                           '03-01-2018': [[7, 8, 21, 24, 61, 67, 70, 76],
                                          [10,
                                           11,
                                           14,
                                           18,
                                           22,
                                           23,
                                           25,
                                           27,
                                           29,
                                           31,
                                           32,
                                           51,
                                           52,
                                           54,
                                           55,
                                           56,
                                           58,
                                           59,
                                           64,
                                           68,
                                           71],
                                          [9, 12, 20, 26, 30, 33, 34, 50, 66, 69, 74, 75, 77],
                                          [6, 13, 15, 16, 17, 19, 28, 49, 53, 57, 60, 62, 63, 65, 72, 73]],
                           '03-02-2018': [[7, 12, 21, 28, 52, 54, 64, 69, 70, 75, 77],
                                          [6, 9, 17, 31, 32, 59, 63, 65, 68, 76],
                                          [8, 14, 16, 22, 23, 25, 29, 30, 33, 53, 57, 60, 62, 72, 73, 74],
                                          [10, 11, 13, 15, 18, 19, 20, 24, 26, 27, 55, 56, 58, 61, 66, 67, 71]]},
 'label_codes': {'Benign': 0,
                 'FTP-BruteForce': 2,
                 'SSH-Bruteforce': 2,
                 'DoS attacks-GoldenEye': 3,
                 'DoS attacks-Slowloris': 3,
                 'DoS attacks-Hulk': 3,
                 'DoS attacks-SlowHTTPTest': 3,
                 'DDoS attacks-LOIC-HTTP': 4,
                 'DDOS attack-HOIC': 4,
                 'DDOS attack-LOIC-UDP': 4,
                 'Brute Force -Web': 6,
                 'Brute Force -XSS': 6,
                 'SQL Injection': 6,
                 'Infilteration': 5,
                 'Bot': 1},
 'population': [[3032878, 75556, 80733, 84562, 134542, 31151, 130],
                [3283069, 57258, 101159, 42255, 239821, 39983, 305],
                [3400570, 73943, 80307, 58506, 422835, 32223, 255],
                [3768177, 79434, 118750, 468977, 466735, 58577, 238]],
 'original_counts': [[3032878, 75556, 80733, 84562, 134542, 31151, 130],
                     [26189, 1183, 1146, 1215, 1784, 1187, 305],
                     [27195, 1230, 1166, 1255, 3359, 1189, 255],
                     [29383, 1171, 1192, 3825, 3845, 1146, 238]],
 'expanded_counts': [[0, 0, 0, 0, 0, 0, 0],
                     [84670, 2914, 2752, 2722, 6316, 2723, 305],
                     [88425, 2812, 2807, 2894, 10869, 2813, 255],
                     [97792, 2728, 3010, 12199, 12136, 2828, 238]],
 'family_holdout_counts': [35985, 5472, 5502, 5496, 5416, 5289, 130]}
CLASSES = np.arange(len(LABELS))
POPULATION = np.asarray(DATA["population"])
ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "dataset" / "archive.zip"
DEFAULT_CACHE = ROOT / "dataset" / "prepared"
DEFAULT_OUTPUTS = ROOT / "dataset" / "models"


def annotate(chunk, capture):
    raw = chunk["Label"].str.strip()
    timestamp = pd.to_datetime(
        chunk["Timestamp"].str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    # The member name disambiguates dates with a low numerical day and month.
    expected = pd.to_datetime(capture, format="%m-%d-%Y")
    valid = raw.notna() & raw.ne("Label") & timestamp.notna() & timestamp.dt.normalize().eq(expected)
    minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
    blocks = (minutes.fillna(-1).astype("int16") // BLOCK_MINUTES).to_numpy()
    return raw, valid.to_numpy(), blocks


def archive_chunks(path, features=False):
    columns = [*IDENTITY_COLUMNS, *FEATURES] if features else IDENTITY_COLUMNS
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            capture = Path(name).stem
            with archive.open(name) as member:
                reader = pd.read_csv(
                    member,
                    usecols=columns,
                    dtype={"Label": "string", "Timestamp": "string"},
                    chunksize=CHUNK,
                    low_memory=False,
                )
                for chunk in reader:
                    yield capture, chunk


def validate_archive(archive_path):
    if not archive_path.is_file():
        raise FileNotFoundError(f"Set ARCHIVE to the original archive.zip. Checked {archive_path}")

    with ZipFile(archive_path) as archive_file:
        members = {
            m.filename: [m.file_size, m.CRC]
            for m in archive_file.infolist() if m.filename.lower().endswith(".csv")
        }
    if members != DATA["capture_members"]:
        raise ValueError("The capture names, sizes or checksums differ from the study archive.")


def prepare_data(archive_path, holdout_dir):
    mappings = {}
    for capture, buckets in DATA["time_blocks_by_bucket"].items():
        mapping = np.full(144, -1, dtype=np.int8)
        for bucket, blocks in enumerate(buckets):
            mapping[blocks] = bucket
        mappings[capture] = mapping

    original_rates = np.minimum(1.0, np.maximum(0.008, 1200 / POPULATION))
    original_rates[0] = 1.0
    expanded_rates = np.minimum(1.0, np.maximum(0.026, 2800 / POPULATION))
    original_rng = np.random.default_rng(SEED + 1)
    expanded_rng = np.random.default_rng(SEED + 51)
    original_counts = np.zeros_like(POPULATION)
    expanded_counts = np.zeros_like(POPULATION)
    original_parts, expanded_parts = [], []
    holdout_paths = []
    previous_capture = None

    for capture, chunk in archive_chunks(archive_path, features=True):
        if capture != previous_capture:
            print(f"Reading {capture}", flush=True)
            previous_capture = capture
        raw, valid, block = annotate(chunk, capture)
        codes = raw.map(DATA["label_codes"]).fillna(-1).to_numpy(dtype=np.int8)
        buckets = mappings[capture][np.clip(block, 0, 143)]
        positions = np.flatnonzero(valid & (codes >= 0) & (buckets >= 0))
        y, fold = codes[positions], buckets[positions]

        keep = original_rng.random(len(y)) < original_rates[fold, y]
        y_original, fold_original = y[keep], fold[keep]
        X_original = flow_array(chunk, positions[keep])
        np.add.at(original_counts, (fold_original, y_original), 1)
        held_out = fold_original == 0
        if held_out.any():
            path = holdout_dir / f"{len(holdout_paths):04d}.npz"
            np.savez(path, X=X_original[held_out], y=y_original[held_out])
            holdout_paths.append(path)
        if (~held_out).any():
            original_parts.append((X_original[~held_out], y_original[~held_out], fold_original[~held_out]))

        development = fold > 0
        y_dev, fold_dev = y[development], fold[development]
        expanded_keep = expanded_rng.random(len(y_dev)) < expanded_rates[fold_dev, y_dev]
        y_expanded, fold_expanded = y_dev[expanded_keep], fold_dev[expanded_keep]
        if len(y_expanded):
            X_expanded = flow_array(chunk, positions[development][expanded_keep])
            np.add.at(expanded_counts, (fold_expanded, y_expanded), 1)
            expanded_parts.append((X_expanded, y_expanded, fold_expanded))

    np.testing.assert_array_equal(original_counts, DATA["original_counts"])
    np.testing.assert_array_equal(expanded_counts, DATA["expanded_counts"])
    pools = {
        "original": dict(zip(("X", "y", "fold"), (np.concatenate(parts) for parts in zip(*original_parts)))),
        "expanded": dict(zip(("X", "y", "fold"), (np.concatenate(parts) for parts in zip(*expanded_parts)))),
    }
    return pools, holdout_paths


def flow_array(chunk, positions):
    frame = chunk.iloc[positions][FEATURES].apply(pd.to_numeric, errors="coerce")
    X = frame.to_numpy(dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    required = X[:, 1:6]
    required[required < 0] = np.nan
    return X


def numeric_input(X):
    if hasattr(X, "loc"):
        X = X.loc[:, FEATURES].apply(lambda col: col.astype("float32"))
    frame = np.asarray(X, dtype=np.float32)
    if frame.ndim != 2 or frame.shape[1] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} columns in FEATURES order.")
    values = frame.copy()
    values[~np.isfinite(values)] = np.nan
    required = values[:, 1:6]
    required[required < 0] = np.nan
    return values


def regression_inputs(values):
    predictors = np.log1p(values[:, 1:4])
    target = np.log1p(values[:, 4] + values[:, 5])
    return predictors, target, np.isfinite(predictors).all(axis=1) & np.isfinite(target)


class FlowFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variant="base", scale=False, retained_variance=0.95):
        self.variant = variant
        self.scale = scale
        self.retained_variance = retained_variance

    def _continuous(self, values):
        measurements = values[:, 1:]
        transformed = np.sign(measurements) * np.log1p(np.abs(measurements))
        if self.use_residual_:
            rx, target, valid = regression_inputs(values)
            residual = np.full(len(values), np.nan, dtype=np.float32)
            if valid.any():
                residual[valid] = target[valid] - self.regression_.predict(rx[valid])
            transformed = np.column_stack([transformed, residual])
        return transformed

    @staticmethod
    def _protocol(values):
        return np.where(np.isfinite(values[:, :1]), values[:, :1], -1)

    def fit(self, X, y=None):
        values = numeric_input(X)
        self.use_residual_ = "residual" in self.variant
        if self.use_residual_:
            rx, target, valid = regression_inputs(values)
            self.regression_ = LinearRegression().fit(rx[valid], target[valid])
        self.protocol_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.protocol_.fit(self._protocol(values))
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        Z = self.imputer_.fit_transform(self._continuous(values))
        self.scaler_ = None
        self.pca_ = None
        if self.scale or "pca" in self.variant:
            self.scaler_ = StandardScaler().fit(Z)
        if "pca" in self.variant:
            self.pca_ = PCA(n_components=self.retained_variance, svd_solver="full")
            self.pca_.fit(self.scaler_.transform(Z))
        return self

    def transform(self, X):
        values = numeric_input(X)
        Z = self.imputer_.transform(self._continuous(values))
        if self.scaler_ is not None:
            Z = self.scaler_.transform(Z)
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return np.ascontiguousarray(np.column_stack([Z, self.protocol_.transform(self._protocol(values))]), dtype=np.float32)


def training_indices(y, eligible, seed):
    rng = np.random.default_rng(seed)
    caps = [12_000, 2_600, 2_600, 2_600, 2_600, 2_600, 2_600]
    sample = []
    for cls, cap in zip(CLASSES, caps):
        positions = np.flatnonzero(eligible & (y == cls))
        sample.append(rng.choice(positions, min(len(positions), cap), replace=False))
    chosen = np.concatenate(sample)
    rng.shuffle(chosen)
    return chosen


def scores(matrix):
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    precision = np.divide(correct, predicted, out=np.zeros_like(correct, dtype=float), where=predicted > 0)
    recall = np.divide(correct, actual, out=np.zeros_like(correct, dtype=float), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=precision + recall > 0)
    return {
        "macro_f1": float(f1.mean()), "accuracy": float(correct.sum() / actual.sum()),
        "weighted_f1": float(np.average(f1, weights=actual)), "balanced_accuracy": float(recall.mean()),
        "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(), "support": actual.tolist(),
    }


def adjust_probabilities(probabilities, ratio, power):
    values = np.asarray(probabilities, dtype=np.float64) * np.power(ratio, power)
    return values / values.sum(axis=1, keepdims=True)


class FlowDecisionModel:
    """A fitted flow pipeline with optional class-prior adjustment and blending."""

    def __init__(self, pipelines, prior_ratios, powers, weights):
        self.pipelines = pipelines
        self.prior_ratios = prior_ratios
        self.powers = powers
        self.weights = weights
        self.classes_ = pipelines[0].classes_
        if any(not np.array_equal(self.classes_, pipeline.classes_) for pipeline in pipelines):
            raise ValueError("The fitted pipelines must use the same class order.")

    def predict_proba(self, X):
        scores = (
            weight * adjust_probabilities(pipeline.predict_proba(X), ratio, power)
            for pipeline, ratio, power, weight in zip(self.pipelines, self.prior_ratios, self.powers, self.weights)
        )
        return sum(scores)

    def predict(self, X):
        if len(self.pipelines) == 1 and self.powers[0] == 0:
            return self.pipelines[0].predict(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def training_subset(pool, budget, fit_power, fold):
    y = pool["y"]
    buckets = pool["fold"]
    eligible = buckets != fold if fold is not None else np.ones(len(y), dtype=bool)
    seed = SEED + (fold if fold is not None else 20)

    if budget == "legacy":
        selected = training_indices(y, eligible, seed)
    elif budget == "medium":
        rng = np.random.default_rng(seed)
        caps = [40_000] + [7_000] * 6
        parts = []
        for cls, maximum in zip(CLASSES, caps):
            positions = np.flatnonzero(eligible & (y == cls))
            parts.append(rng.choice(positions, min(len(positions), maximum), replace=False))
        selected = np.concatenate(parts)
        rng.shuffle(selected)
    elif budget == "all":
        selected = np.flatnonzero(eligible)
    else:
        raise ValueError(f"Unknown fit budget {budget}")

    fit_y, fit_folds = y[selected], buckets[selected]
    fitting_buckets = [b for b in (1, 2, 3) if b != fold]
    population = POPULATION[fitting_buckets].sum(axis=0)
    weights = None
    if fit_power:
        counts = np.zeros_like(POPULATION)
        np.add.at(counts, (fit_folds, fit_y), 1)
        factors = POPULATION[fit_folds, fit_y] / counts[fit_folds, fit_y]
        weights = factors ** fit_power
        weights /= weights.mean()
    training_mass = np.bincount(fit_y, weights=weights, minlength=len(CLASSES))
    if np.any(training_mass == 0):
        raise ValueError("Every fitting subset must contain all seven classes.")
    ratio = (population / population.sum()) / (training_mass / training_mass.sum())
    return selected, weights, ratio


def shared_holdout_sample(paths):
    rng = np.random.default_rng(SEED + 151)
    target = np.array([36_000, 5_500, 5_500, 5_500, 5_500, 5_500, 10_000])
    sample_rates = np.minimum(1.0, target / POPULATION[0])
    parts = []
    for path in paths:
        with np.load(path) as batch:
            y = batch["y"]
            keep = rng.random(len(y)) < sample_rates[y]
            parts.append((batch["X"][keep], y[keep]))
    X, y = (np.concatenate(blocks) for blocks in zip(*parts))
    sample_counts = np.bincount(y, minlength=len(CLASSES))
    np.testing.assert_array_equal(sample_counts, DATA["family_holdout_counts"])
    return X, y, (POPULATION[0] / sample_counts)[y]


def prepare_cache(archive, cache):
    """Stage a single copy of the prepared data for all training commands."""
    validate_archive(archive)
    if cache.is_dir() and (cache / "complete").is_file():
        return

    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="preparing-", dir=cache.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        pools, _ = prepare_data(archive, stage)
        for name, arrays in pools.items():
            np.savez(stage / f"{name}.npz", **arrays)
        (stage / "complete").touch()
        try:
            stage.rename(cache)
        except OSError:
            if not (cache / "complete").is_file():
                raise


def read_pool(cache, name="original"):
    with np.load(cache / f"{name}.npz") as data:
        return {key: data[key] for key in ("X", "y", "fold")}


def training_args(description, output_name):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / f"{output_name}.joblib")
    args = parser.parse_args()
    prepare_cache(args.archive, args.cache)
    return args


def fit_classifier(estimator, data, *, budget="legacy", variant="base", scale=False, fit_power=0.0):
    X, y = data["X"], data["y"]
    selected, weights, ratio = training_subset(data, budget, fit_power, fold=None)
    pipeline = Pipeline([("features", FlowFeatures(variant=variant, scale=scale)), ("model", estimator)])
    params = {"model__sample_weight": weights} if weights is not None else {}
    pipeline.fit(X[selected], y[selected], **params)
    np.testing.assert_array_equal(pipeline.classes_, CLASSES)
    return pipeline, ratio


def save_model(model, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output, compress=3)
    print(f"Saved {output}")


def train_classifier(name, estimator, *, pool="original", prior_power=0.0, **fit_settings):
    args = training_args(f"Train the selected {name} classifier.", name.lower().replace(" ", "_"))
    data = read_pool(args.cache, pool)
    with threadpool_limits(limits=3), joblib.parallel_backend("threading", n_jobs=3):
        pipeline, ratio = fit_classifier(estimator, data, **fit_settings)
        model = FlowDecisionModel([pipeline], [ratio], [prior_power], [1.0])
        holdout = sorted(p for p in args.cache.glob("*.npz") if p.stem.isdigit())
        X, y, weights = shared_holdout_sample(holdout)
        matrix = confusion_matrix(y, model.predict(X), labels=CLASSES, sample_weight=weights)
    print(f"{name} held-out sample macro-F1 {scores(matrix)['macro_f1']:.4f}")
    save_model(model, args.output)


def train_clustering(name, output_name, model):
    args = training_args(f"Fit {name}.", output_name)
    data = read_pool(args.cache)
    X, y = data["X"], data["y"]
    fit = training_indices(y, np.ones(len(y), dtype=bool), SEED + 18)
    sample = np.random.default_rng(SEED + 17).choice(len(y), 2_000, replace=False)
    with threadpool_limits(limits=2):
        preprocessing = FlowFeatures(scale=True).fit(X[fit])
        model.fit(preprocessing.transform(X[sample]))
    save_model({"preprocessing": preprocessing, "model": model}, args.output)
