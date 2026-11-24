import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from common import SEED, numeric_input, read_pool, regression_inputs, save_model, training_args, training_indices


if __name__ == "__main__":
    args = training_args("Fit the log-byte regression.", "log_byte_regression")
    data = read_pool(args.cache)
    predictors, target, valid = regression_inputs(numeric_input(data["X"]))
    fit = training_indices(data["y"], np.ones(len(target), dtype=bool), SEED + 20)
    fit = fit[valid[fit]]
    model = make_pipeline(
        FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}),
        PolynomialFeatures(6, include_bias=False), StandardScaler(),
        Ridge(alpha=0.0003, solver="cholesky"),
    )
    with threadpool_limits(limits=2):
        model.fit(predictors[fit], target[fit])
    save_model(model, args.output)
