from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


def calculate_ensemble_prediction_proba(X, ensemble):
    probabilities_matrix = \
        [base_classifier.predict_proba(X) for base_classifier in ensemble]
    probabilities_matrix = np.row_stack([x[:, 1] for x in probabilities_matrix]).T
    mean_probabilities = probabilities_matrix.mean(axis=1)
    return mean_probabilities


def calculate_ensemble_prediction(X, ensemble):
    return calculate_ensemble_prediction_proba(X, ensemble).round()


class DecorateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, artificial_examples=np.empty(0), base_classifier=LogisticRegression(solver='liblinear'), r_size=1,
                 target_ensemble_size=10, max_iterations=50, metric=accuracy_score):
        self.base_classifier = base_classifier
        self.artificial_examples = artificial_examples
        self.r_size = r_size
        self.target_ensemble_size = target_ensemble_size
        self.max_iterations = max_iterations
        self.metric = metric
        self.ensemble = []
        self.current_ensemble_size = 0

    def calculate_ensemble_metric(self, X, y, ensemble):
        return self.metric(y, calculate_ensemble_prediction(X, ensemble))

    def fit(self, X, y):
        if not self.r_size > 0 and self.r_size <= 1:
            raise ValueError("r_size should be in the range (0,1]")
        if not self.target_ensemble_size <= self.max_iterations:
            raise ValueError("target_ensemble_size should be smaller or equal to max_iterations")
        if self.artificial_examples.size == 0:
            raise ValueError("artificial_examples can't be empty")
        X, y = check_X_y(X, y, accept_sparse=True)

        first_classifier = clone(self.base_classifier)
        first_classifier.fit(X, y)
        self.ensemble.append(first_classifier)
        completed_iterations = 1
        self.current_ensemble_size = 1

        while (self.current_ensemble_size < self.target_ensemble_size) and (completed_iterations < self.max_iterations):
            ensemble_metric = self.calculate_ensemble_metric(X, y, self.ensemble)
            number_of_artificial_examples_per_iteration = round(self.r_size * len(X))
            replace_value = len(self.artificial_examples) < number_of_artificial_examples_per_iteration
            training_examples = self.artificial_examples[np.random.choice(
                self.artificial_examples.shape[0], size=number_of_artificial_examples_per_iteration,
                replace=replace_value)]
            ensemble_prediction_of_training_examples = calculate_ensemble_prediction(training_examples, self.ensemble)
            inverse_prediction_of_training_examples = 1 - ensemble_prediction_of_training_examples

            x_and_training_examples = np.concatenate([X, training_examples])
            y_and_inverse_prediction_of_training_examples = \
                np.concatenate([y, inverse_prediction_of_training_examples])
            classifier_to_add_to_ensemble = clone(self.base_classifier)
            classifier_to_add_to_ensemble.fit(x_and_training_examples, y_and_inverse_prediction_of_training_examples)
            new_ensemble = self.ensemble.copy()
            new_ensemble.append(classifier_to_add_to_ensemble)
            new_ensemble_metric = self.calculate_ensemble_metric(X, y, new_ensemble)
            if new_ensemble_metric >= ensemble_metric:
                self.ensemble = new_ensemble.copy()
                self.current_ensemble_size = self.current_ensemble_size + 1
            completed_iterations = completed_iterations + 1

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return calculate_ensemble_prediction_proba(X, self.ensemble)

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return calculate_ensemble_prediction(X, self.ensemble)
