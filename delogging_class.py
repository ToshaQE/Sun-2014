import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import false

class delogging:

    def __init__(self, target_original, target_logged, test_size):
        self.target_original = target_original.reset_index(drop=True, inplace=True)
        self.target_logged = target_logged.reset_index(drop=True, inplace=True)
        self.test_size = test_size
    
    order_of_int = 0

    def train_test(self):
        if order_of_int == 0:
            y_train, y_test = train_test_split(self.target_original, test_size=test_size, shuffle=False)
        else:
            y_train_logged, y_test_logged = train_test_split(self.target_logged, test_size=test_size, shuffle=False)

