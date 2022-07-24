import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM

import statsmodels.api as sm

def load_data(file_name):
    """Returns a pandas dataframe from a csv file."""
    return pd.read_csv(file_name)


def tts(data):
    """Splits the data into train and test. Test set consists of the last 12
    months of data.
    """
    data = data.drop(['sales', 'date'], axis=1)
    train, test = data[0:-12].values, data[-12:].values

    return train, test


def main():
    """Calls all functions to load data, run regression models, run lstm model,
    and run arima model.
    """
    # Regression models
    model_df = load_data('data/model_df.csv')
    train, test = tts(model_df)

    # Sklearn
    regressive_model(train, test, LinearRegression(), 'LinearRegression')
    regressive_model(train, test, RandomForestRegressor(n_estimators=100,
                                                        max_depth=20),
                     'RandomForest')
    regressive_model(train, test, XGBRegressor(n_estimators=100,
                                               learning_rate=0.2,
                                               objective='reg:squarederror'),
                     'XGBoost')
    # Keras
    lstm_model(train, test)

    # Arima
    ts_data = load_data('data/arima_df.csv').set_index('date')
    ts_data.index = pd.to_datetime(ts_data.index)

    sarimax_model(ts_data)


main()


