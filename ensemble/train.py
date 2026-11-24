import joblib
from threadpoolctl import threadpool_limits

from common import FlowDecisionModel, fit_classifier, read_pool, save_model, training_args
from model_1.train import make_estimator as make_forest
from model_2.train import make_estimator as make_bagged_trees


if __name__ == "__main__":
    args = training_args("Fit the frozen intrusion detection ensemble.", "ensemble")
    data = read_pool(args.cache)

    with threadpool_limits(limits=3), joblib.parallel_backend("threading", n_jobs=3):
        bagged, bagged_prior = fit_classifier(
            make_bagged_trees(n_estimators=36, max_depth=15, min_samples_leaf=4), data, budget="all",
        )
        forest, forest_prior = fit_classifier(make_forest(), data, budget="all", variant="residual")

    model = FlowDecisionModel([bagged, forest], [bagged_prior, forest_prior], [0.65, 0.65], [0.25, 0.75])
    save_model(model, args.output)
