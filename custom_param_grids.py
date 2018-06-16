GRIDS = {
    "gradientboostingclassifier": {
        "gradientboostingclassifier__criterion": ["friedman_mse", "mae"],
        "gradientboostingclassifier__loss": ["deviance", "exponential"],
        "gradientboostingclassifier__n_estimators": [40, 50, 60, 90, 100],
        "gradientboostingclassifier__max_depth": [6, 7, 8, 9, 10],
        "gradientboostingclassifier__max_features": [None, "sqrt", "log2"],
        "gradientboostingclassifier__subsample": [0.8, 1.]
    },
    "linearsvc": {
        "linearsvc__loss": [
            "squared_hinge"
        ],  # loss "hinge" does not work with penalty "l1" when dual is False
        "linearsvc__C": [0.1, 0.5, 1.0, 10, 100, 200],
        "linearsvc__verbose": [0],
        "linearsvc__intercept_scaling": [1],
        "linearsvc__fit_intercept": [True],
        "linearsvc__max_iter": [1000],
        "linearsvc__penalty":
        ["l2"],  # penalty "l1" does not work with loss "hinge"
        "linearsvc__multi_class": ["ovr"],
        "linearsvc__random_state": [None],
        "linearsvc__dual": [True],
        "linearsvc__tol": [0.0001],
        "linearsvc__class_weight": [None]
    }
}


def get_param_grid(algorithm):
    try:
        grid = GRIDS[algorithm]
    except KeyError:
        print "WARNING: No parameter grid for algorithm '{}' found".format(
            algorithm)
        grid = {}
    return grid


BEST_KNOWN_PARAMETERS = {
    "gradientboostingclassifier": {
        "criterion": "friedman_mse",
        "max_depth": 8,
        "n_estimators": 100,
        "max_features": None,
        "subsample": 1.0,
        "loss": "deviance"
    },
    "linearsvc": {
        "C": 200,
        "fit_intercept": True,
        "max_iter": 1000,
        "penalty": "l2",
        "class_weight": None,
        "multi_class": "ovr",
        "dual": True,
        "verbose": 0,
        "tol": 0.0001,
        "intercept_scaling": 1,
        "random_state": None,
        "loss": "hinge"
    }
}


def get_best_parameter_set(algorithm, do_prefix=True):
    """Returns the best known parameter set for an algorithm.

    Args:
        `do_prefix`: Include the algorithm prefix or not.
    """
    try:
        params = BEST_KNOWN_PARAMETERS[algorithm]
    except KeyError:
        params = {}
    key_prefix = (algorithm + "__") if do_prefix else ""
    return {key_prefix + param: value for param, value in params.items()}
