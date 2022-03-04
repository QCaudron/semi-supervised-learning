import json
import logging
from argparse import ArgumentParser
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

from ipl.models import (
    RandomIterativePseudoLabeler,
    TopConfidenceIterativePseudoLabeler,
    ConfidenceIntervalIterativePseudoLabeler,
)

logging.basicConfig(format="[%(levelname)s:%(asctime)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="The class labels to keep in the dataset. Defaults to all classes.",
        default=None,
    )
    parser.add_argument(
        "--n_trees",
        type=int,
        help="The number of trees to use in the CatBoost models. Defaults to 200.",
        default=200,
    )
    parser.add_argument(
        "--n_repetitions",
        type=int,
        help="How many times to try each hyperparameter ? Defaults to 25.",
        default=25,
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        help="How many train-predict cycles during the iterative fitting ?",
        default=250,
    )

    args = parser.parse_args()
    return args


def prepare_dataset(classes: Optional[List[int]]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the cover-type dataset.

    Parameters
    ----------
    classes : Optional[List[int]]
        A list of classes to keep in the dataset. If None, all classes are used. Defaults to None.

    Returns
    -------
    pd.DataFrame
        The feature set, X.
    pd.Series
        The target classes, y.
    """

    # Load the tree cover type dataset
    dataset = fetch_covtype(as_frame=True, shuffle=True)["frame"]

    # Stick to only the most common classes
    if classes is None:
        classes = dataset["Cover_Type"].unique()

    # Split out features from targets
    dataset = dataset.loc[dataset["Cover_Type"].isin(classes)]
    X = dataset.drop("Cover_Type", axis=1)
    y = dataset["Cover_Type"]

    return X, y


def label_some_data(
    X: pd.DataFrame, y: pd.Series, n_labeled
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pretend you've got humans to label a small quantity of data.

    Parameters
    ----------
    X : pd.DataFrame
        The feature set.
    y : pd.Series
        The target classes.
    n_labeled : int, optional
        The number of datapoints to return as labeled.

    Returns
    -------
    pd.DataFrame
        The labeled feature set, X_known.
    pd.DataFrame
        The feature set that is unlabeled, X_unknown.
    pd.Series
        The target classes for the labeled data, y_known.
    pd.Series
        The target classes for the unlabeled data, y_unknown. Use this for validation only.
    """

    # Pick a subset of the data to "label"
    idx_labeled = np.random.choice(range(len(X)), size=n_labeled, replace=False)
    idx_unknown = [i for i in range(len(X)) if i not in idx_labeled]

    # Split data into labeled and unlabeled
    X_known = X.iloc[idx_labeled]
    y_known = y.iloc[idx_labeled]
    X_unknown = X.iloc[idx_unknown]
    y_unknown = y.iloc[idx_unknown]

    return X_known, X_unknown, y_known, y_unknown


def main(classes, n_trees, n_repetitions, n_iterations):

    # Load the data, and split into master train / test sets
    X, y = prepare_dataset(classes)
    X_master_train, X_master_test, y_master_train, y_master_test = train_test_split(X, y)
    logger.info(f"Loaded dataset ({X.shape=}, {y.nunique()=}).")

    # "Optimal" : if we had all of the data labeled, how well could we do ?
    full_model = CatBoostClassifier(n_estimators=n_trees, verbose=0)
    full_model.fit(X_master_train, y_master_train)
    full_accuracy = accuracy_score(y_master_test, full_model.predict(X_master_test))
    logger.info(f"Trained and evaluated full model ({full_accuracy=:.4f}).")

    # Instead, assume we only have a small quantity of the data labeled
    X_known, X_unknown, y_known, y_unknown = label_some_data(
        X_master_train, y_master_train, n_labeled=500
    )

    # Using only this labeled data, how well would we do ?
    labeled_model = CatBoostClassifier(n_estimators=n_trees, verbose=0)
    labeled_model.fit(X_known, y_known)
    labeled_accuracy = accuracy_score(y_unknown, labeled_model.predict(X_unknown))
    logger.info(f"Trained and evaluated labeled-only model ({labeled_accuracy=:.4f}).")

    # Let's experiment
    n_new_points_options = [25, 100, 250]
    exp_results = {}

    for n_new_points in n_new_points_options:

        random_results = []
        top_conf_results = []
        conf_interval_results = []
        logger.info(f"Trying {n_new_points=}.")

        for _ in range(n_repetitions):

            random_model = RandomIterativePseudoLabeler(
                X_known=X_known,
                y_known=y_known,
                X_unknown=X_unknown,
                X_master_test=X_master_test,
                y_master_test=y_master_test,
                model_kwargs={"n_estimators": n_trees, "verbose": 0},
            )
            random_accuracy, _ = random_model.fit(
                n_iterations=n_iterations, n_new_points=n_new_points
            )
            random_results.append(random_accuracy)

            top_conf_model = TopConfidenceIterativePseudoLabeler(
                X_known=X_known,
                y_known=y_known,
                X_unknown=X_unknown,
                X_master_test=X_master_test,
                y_master_test=y_master_test,
                model_kwargs={"n_estimators": n_trees, "verbose": 0},
            )
            top_conf_accuracy, _ = top_conf_model.fit(
                n_iterations=n_iterations, n_new_points=n_new_points
            )
            top_conf_results.append(top_conf_accuracy)

            conf_interval_model = ConfidenceIntervalIterativePseudoLabeler(
                X_known=X_known,
                y_known=y_known,
                X_unknown=X_unknown,
                X_master_test=X_master_test,
                y_master_test=y_master_test,
                model_kwargs={"n_estimators": n_trees, "verbose": 0},
            )
            conf_interval_accuracy, _ = conf_interval_model.fit(
                n_iterations=n_iterations, n_new_points=n_new_points, lower=0.85, upper=0.92
            )
            conf_interval_results.append(conf_interval_accuracy)

        exp_results[f"{n_new_points=}"] = {
            "random": np.array(random_results).tolist(),
            "top_conf": np.array(top_conf_results).tolist(),
            "conf_interval": np.array(conf_interval_results).tolist(),
        }

    breakpoint()

    # Write results
    results = {
        "full_accuracy": float(full_accuracy),
        "labeled_accuracy": float(labeled_accuracy),
        **exp_results,
    }
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":

    args = parse_args()
    main(args.classes, args.n_trees, args.n_repetitions, args.n_iterations)
