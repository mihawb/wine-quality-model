import numpy as np
import pandas as pd
from math import sqrt


def feature_derivative(errors, feature) -> float:
	derivative = 2 * np.dot(errors, feature.T)
	return derivative


def regression_gradient_descent(feature_matrix: pd.DataFrame, output: pd.Series, initial_weights: np.array, step_size: float, tolerance: float) -> np.array:
	converged = False
	weights = np.array(initial_weights)

	while not converged:
		prognoza = np.dot(feature_matrix, weights)

		errarr = prognoza - output

		gradient_sum_squares = 0
		for i in range(len(weights)):
			d = feature_derivative(errarr, feature_matrix.iloc[:, i])
			gradient_sum_squares += d ** 2
			weights[i] -= step_size * d

		gradient_magnitude = sqrt(gradient_sum_squares)
		if gradient_magnitude < tolerance:
			converged = True
			
	return weights