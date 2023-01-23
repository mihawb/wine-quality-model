import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def measure_round_degree(actual: pd.Series, series: pd.Series, series_rounded: pd.Series) -> dict:
	errors = dict()

	# series below show rounding errors (i.e. predictions vs pred. rounded)
	errors['rounding_num'] = np.sum(series.astype('int16') != series_rounded.astype('int16'))
	errors['rounding_sum'] = np.sum(np.absolute(series - series_rounded))
	errors['rounding_mean_abs'] = mean_absolute_error(series, series_rounded)
	errors['rounding_mean_sqrt'] = mean_squared_error(series, series_rounded)

	# series below show how predictions compare to actual values
	errors['actual_sum'] = np.sum(np.absolute(actual - series))
	errors['actual_mean_abs'] = mean_absolute_error(actual, series)
	errors['actual_mean_sqrt'] = mean_squared_error(actual, series)

	return errors


def round_result(actual: pd.Series, series: pd.Series) -> tuple((pd.Series, dict)):
	series_rounded = series.round()
	errors = measure_round_degree(actual, series, series_rounded)

	return series_rounded, errors


def get_errors(actual: pd.Series, series: pd.Series) -> dict:
	errors = dict()
	errors['actual_sum'] = np.sum(np.absolute(actual - series))
	errors['actual_mean_abs'] = mean_absolute_error(actual, series)
	errors['actual_mean_sqrt'] = mean_squared_error(actual, series)
	return errors
