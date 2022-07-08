import numpy as np
import pandas as pd


def rae(actual, predicted):
    numerator = np.sum(np.abs(predicted - actual))
    denominator = np.sum(np.abs(np.mean(actual) - actual))
    return numerator / denominator


def rrse(actual, predicted):
    numerator = np.sqrt(np.sum(np.square(predicted - actual)))
    denominator = np.sqrt(np.sum(np.square(np.mean(actual) - actual)))
    return numerator / denominator