from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import trange


class BaseModel:
    def __init__(
        self,
        X_known: pd.DataFrame,
        y_known: pd.Series,
        X_unknown: pd.DataFrame,
        X_master_test: pd.DataFrame,
        y_master_test: pd.Series,
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Store all the things.
        """

        for data in [X_known, X_unknown, X_master_test]:
            if data.index.duplicated().any():
                raise ValueError("Found duplicate indexes; try .reset_index()")

        if model_kwargs is None:
            model_kwargs = {"n_estimators": 200, "verbose": 0}

        self.X_known = X_known
        self.y_known = y_known
        self.X_unknown = X_unknown
        self.X_master_test = X_master_test
        self.y_master_test = y_master_test
        self.model_kwargs = model_kwargs
        self.X_pseudolabeled = []
        self.y_pseudolabeled = []

    @property
    def n_datapoints(self) -> int:
        """
        How much data is currently known and pseudolabeled ?

        Returns
        -------
        int
            The total length of the datasets, both known and pseudolabeled.
        """
        return len(np.concatenate((self.X_known, *self.X_pseudolabeled)))

    def _select_from_subset(self, *args, **kwargs):
        return NotImplementedError("Overload this function in inherited classes.")


class IterativePseudoLabeler(BaseModel):

    def _pseudo_label_data(
        self, n_new_points: int, sample_rate: float, **select_from_subset_kwargs,
    ) -> None:
        """
        Make some predictions on all unknown data, find the predictions with
        the highest confidence, and return that subset of predictions.

        Parameters
        ----------
        n_new_points : int
            The number of new datapoints to pseudolabel.
        sample_rate : float
            The fraction of unlabeled data to sample from (for speed and randomness).
        """

        # Grab a random sample of the unknown data (for randomness and speed)
        random_idx = np.random.choice(
            range(len(self.X_unknown)), size=int(sample_rate * len(self.X_unknown)), replace=False,
        )
        X_subset_unknown = self.X_unknown.iloc[random_idx]

        # Select some points from this subset and make a prediction
        X_pseudo = self._select_from_subset(
            X_subset_unknown, n_new_points, **select_from_subset_kwargs
        )
        y_pseudo = self.predict(X_pseudo)

        # Remove the pseudolabeled data
        self.X_unknown = self.X_unknown.loc[~self.X_unknown.index.isin(X_pseudo.index)]

        # Save the newly-pseudolabeled data
        self.X_pseudolabeled.append(X_pseudo)
        self.y_pseudolabeled.append(y_pseudo)

    def fit(
        self,
        n_iterations: int,
        n_new_points: int,
        sample_rate: float = 0.2,
        **select_from_subset_kwargs,
    ) -> Tuple[List[float], List[int]]:
        """
        Fit the iterative pseudolabeler.

        Iterate n_iterations times, each time making predictions on some unseen data,
        selecting a subset of this data, and add these pseudolabeled datapoints to the
        dataset for the next round of iteration.

        Parameters
        ----------
        n_iterations : int
            The number of train-predict rounds to make.
        n_new_points : int
            The number of new datapoints to add each round.
        sample_rate : float, optional
            The fraction of the unknown dataset to randomly sample at predict time;
            this is done for both speed and randomness. Defaults to 0.2.

        Returns
        -------
        List[float]
            A list of accuracy values at each iteration.
        List[int]
            The number of datapoints (known plus pseudolabeled) at each iteration.
        """

        accuracy = [self.evaluate_accuracy()]
        n_data = [self.n_datapoints]

        for _ in trange(n_iterations):
            self._pseudo_label_data(
                n_new_points=n_new_points, sample_rate=sample_rate, **select_from_subset_kwargs,
            )
            accuracy.append(self.evaluate_accuracy())
            n_data.append(self.n_datapoints)

        return accuracy, n_data

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict from the model, returning a series of class labels.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict from.

        Returns
        -------
        pd.Series
            The class labels, with the same shape and index as X.
        """
        return pd.Series(self.model.predict(X).squeeze(), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict from the model, returning a series of probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict from.

        Returns
        -------
        pd.Series
            The class probabilities, with the same shape and index as X.
        """
        probas = self.model.predict_proba(X).squeeze()
        return pd.DataFrame(
            probas, index=X.index, columns=[f"p_class_{i}" for i in range(probas.shape[1])],
        )

    def evaluate_accuracy(self) -> float:
        """
        Evaluate how well the model performs using known data and pseudolabeled
        data, against the in-theory-unknown master test set.

        Returns
        -------
        float
            The accuracy of the model against existing data, both known and pseudolabeled.
        """

        # Build a model scaling and then classifying the dataset
        model = Pipeline(
            [("scaler", StandardScaler()), ("classifier", CatBoostClassifier(**self.model_kwargs))]
        )

        # Build training set from known data and previously-pseudolabeled data
        X_train = pd.concat([self.X_known, *self.X_pseudolabeled])
        y_train = pd.concat([self.y_known, *self.y_pseudolabeled])

        # Fit the model, see how accurate we are against the master testing set
        model.fit(X_train, y_train)
        self.model = model
        y_hat = self.predict(self.X_master_test)
        accuracy = accuracy_score(self.y_master_test, y_hat)

        return accuracy

    def plot_results(self, accuracy: List[float], n_data: List[int]) -> None:
        """
        Plot the results of .fit()

        Parameters
        ----------
        accuracy : List[float]
            A list of accuracy values at each iteration.
        n_data : List[int]
            The number of datapoints (known plus pseudolabeled) at each iteration.
        """

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.plot(accuracy)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")

        plt.subplot(122)
        plt.plot(n_data)
        plt.xlabel("Iteration")
        plt.ylabel("Number of datapoints")

        plt.tight_layout()


class RandomIterativePseudoLabeler(IterativePseudoLabeler):
    """
    An iterative pseudolabeler that randomly labels data each round.
    """

    def _select_from_subset(self, X_subset, n_new_points, **kwargs):
        idx = np.random.choice(X_subset.index, size=n_new_points, replace=False)
        return X_subset.loc[idx]


class TopConfidenceIterativePseudoLabeler(IterativePseudoLabeler):
    """
    An iterative pseudolabeler that labels the most confident predictions each round.

    This model predicts the probability of a datapoint falling into each class, and
    selects the datapoints that have the highest confidence for pseudolabeling.
    """

    def _select_from_subset(self, X_subset, n_new_points, **kwargs):

        probas = self.predict_proba(X_subset)
        confidence = probas.max(1)

        top_confidence_idx = confidence.sort_values(ascending=False).index
        top_n_confidence_idx = top_confidence_idx[:n_new_points]

        return X_subset.loc[top_n_confidence_idx]


class ConfidenceIntervalIterativePseudoLabeler(IterativePseudoLabeler):
    """
    An iterative pseudolabeler that labels data that falls within a confidence interval.

    This model pseudolabels data where the predicted probabilities of the top class
    fall between a lower and an upper value.
    """

    def _select_from_subset(self, X_subset, n_new_points, **kwargs):

        probas = self.predict_proba(X_subset)
        confidence = probas.max(1)

        within_interval_bool = (confidence > kwargs["lower"]) & (confidence < kwargs["upper"])
        within_interval = X_subset.loc[within_interval_bool]

        return within_interval.sample(min(n_new_points, len(within_interval)))


class HybridIterativePseudoLabeler(IterativePseudoLabeler):
    """
    An iterative model that combines a random, top-confidence, and confidence-interval approach.
    """

    def _select_from_subset(self, X_subset, n_new_points, **kwargs):

        n_interval = int(n_new_points * kwargs["frac_interval"])
        n_confidence = int(n_new_points * kwargs["frac_confidence"])
        n_random = n_new_points - n_interval - n_confidence

        # Predict probabilities and calculate confidence; we'll need these later
        probas = self.predict_proba(X_subset)
        confidence = probas.max(1)

        # Select the random bunch first
        X_subset_random = X_subset.sample(n_random)

        # Then, select the top bunch by confidence
        top_confidence_idx = confidence.sort_values(ascending=False).index
        X_subset_confidence = X_subset.loc[top_confidence_idx[:n_confidence]]

        # Finally, select the bunch that fall within the confidence intervals we want
        within_intervals_bool = (confidence > kwargs["lower"]) & (confidence < kwargs["upper"])
        within_intervals_idx = within_intervals_bool[within_intervals_bool].index
        X_subset_interval = X_subset.loc[within_intervals_idx].sample(
            min(n_interval, len(within_intervals_idx))
        )

        # Mix them all together
        selected = pd.concat(
            [X_subset_random, X_subset_confidence, X_subset_interval]
        ).drop_duplicates()

        # There might be some duplicates; drop those and fill with random values until we're happy
        break_counter = 0
        while len(selected) < n_new_points:

            selected = pd.concat(
                [selected, X_subset.sample(n_new_points - len(selected))]
            ).drop_duplicates()

            break_counter += 1
            if break_counter > 5:
                break

        return selected


class TwoModelPseudoLabeler(BaseModel):

    def _pseudo_label_data(
        self, n_new_points: int, sample_rate: float, **select_from_subset_kwargs,
    ) -> None:
        """
        Make some predictions on all unknown data, find the predictions with
        the highest confidence, and return that subset of predictions.

        Parameters
        ----------
        n_new_points : int
            The number of new datapoints to pseudolabel.
        sample_rate : float
            The fraction of unlabeled data to sample from (for speed and randomness).
        """

        # Grab a random sample of the unknown data (for randomness and speed)
        random_idx = np.random.choice(
            range(len(self.X_unknown)), size=int(sample_rate * len(self.X_unknown)), replace=False,
        )
        X_subset_unknown = self.X_unknown.iloc[random_idx]

        # Select some points from this subset and make a prediction
        X_pseudo = self._select_from_subset(
            X_subset_unknown, n_new_points, **select_from_subset_kwargs
        )
        y_pseudo = self.predict(X_pseudo)

        # Remove the high-confidence data
        self.X_unknown = self.X_unknown.loc[~self.X_unknown.index.isin(X_pseudo.index)]

        # Save the newly-pseudolabeled data
        self.X_pseudolabeled.append(X_pseudo)
        self.y_pseudolabeled.append(y_pseudo)

    def fit(
        self,
        n_iterations: int,
        n_new_points: int,
        sample_rate: float = 0.2,
        **select_from_subset_kwargs,
    ) -> Tuple[List[float], List[int]]:
        """
        Fit the iterative pseudolabeler.

        Iterate n_iterations times, each time making predictions on some unseen data,
        selecting a subset of this data, and add these pseudolabeled datapoints to the
        dataset for the next round of iteration.

        Parameters
        ----------
        n_iterations : int
            The number of train-predict rounds to make.
        n_new_points : int
            The number of new datapoints to add each round.
        sample_rate : float, optional
            The fraction of the unknown dataset to randomly sample at predict time;
            this is done for both speed and randomness. Defaults to 0.2.

        Returns
        -------
        List[float]
            A list of accuracy values at each iteration.
        List[int]
            The number of datapoints (known plus pseudolabeled) at each iteration.
        """

        accuracy = [self.evaluate_accuracy()]
        n_data = [self.n_datapoints]

        for _ in trange(n_iterations):
            self._pseudo_label_data(
                n_new_points=n_new_points, sample_rate=sample_rate, **select_from_subset_kwargs,
            )
            accuracy.append(self.evaluate_accuracy())
            n_data.append(self.n_datapoints)

        return accuracy, n_data

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict from the model, returning a series of class labels.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict from.

        Returns
        -------
        pd.Series
            The class labels, with the same shape and index as X.
        """
        return pd.Series(self.model.predict(X).squeeze(), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict from the model, returning a series of probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict from.

        Returns
        -------
        pd.Series
            The class probabilities, with the same shape and index as X.
        """
        probas = self.model.predict_proba(X).squeeze()
        return pd.DataFrame(
            probas, index=X.index, columns=[f"p_class_{i}" for i in range(probas.shape[1])],
        )

    def evaluate_accuracy(self) -> float:
        """
        Evaluate how well the model performs using known data and pseudolabeled
        data, against the in-theory-unknown master test set.

        Returns
        -------
        float
            The accuracy of the model against existing data, both known and pseudolabeled.
        """

        # Build a model scaling and then classifying the dataset
        model = Pipeline(
            [("scaler", StandardScaler()), ("classifier", CatBoostClassifier(**self.model_kwargs))]
        )

        # Build training set from known data and previously-pseudolabeled data
        X_train = pd.concat([self.X_known, *self.X_pseudolabeled])
        y_train = pd.concat([self.y_known, *self.y_pseudolabeled])

        # Fit the model, see how accurate we are against the master testing set
        model.fit(X_train, y_train)
        self.model = model
        y_hat = self.predict(self.X_master_test)
        accuracy = accuracy_score(self.y_master_test, y_hat)

        return accuracy

    def plot_results(self, accuracy: List[float], n_data: List[int]) -> None:
        """
        Plot the results of .fit()

        Parameters
        ----------
        accuracy : List[float]
            A list of accuracy values at each iteration.
        n_data : List[int]
            The number of datapoints (known plus pseudolabeled) at each iteration.
        """

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.plot(accuracy)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")

        plt.subplot(122)
        plt.plot(n_data)
        plt.xlabel("Iteration")
        plt.ylabel("Number of datapoints")

        plt.tight_layout()
