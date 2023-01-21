import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def measure_round_degree(series: pd.Series, series_rounded: pd.Series) -> dict:
	errors = dict()
	if not series_rounded: series_rounded = series.round()

	errors['rounding_error_sum'] = np.sum(series - series_rounded)
	errors['rounding_error_abs'] = mean_absolute_error(series, series_rounded)
	errors['rounding_error_sqrt'] = mean_squared_error(series, series_rounded)

	return errors


def result_round(series: pd.Series) -> tuple((pd.Series, dict)):
	series_rounded = series.round()
	errors = measure_round_degree(series, series_rounded=series_rounded)

	return series_rounded, errors

