import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

from util.dataset import load_census_data
from analysis.preprocess import MinMaxScaler

np.random.seed(6)  # for reproducibility
FEATURES = ['geoid', 'year', 'population', 'household_income', 'home_value', 'pop_non_hispanic_caucasians',
            'pop_non_hispanic_blacks', 'pop_indians_alaskans', 'pop_non_hispanic_asians',
            'pop_non_hispanic_hawaiians_pacific', 'pop_non_hispanic_others', 'pop_non_hispanic_multi_racials',
            'pop_hispanics_latinos']  # TODO add pop_graduate
NUM_YEAR = 10
census_df = load_census_data()[FEATURES]

pop_scalar = MinMaxScaler(census_df, 'population')
income_scalar = MinMaxScaler(census_df, 'household_income')
home_scalar = MinMaxScaler(census_df, 'home_value')


def format_singlerow_year_tract(census_df: pd.DataFrame) -> np.ndarray:
    """Format an array of shape (1, years, tracts(features))"""

    dataset = None
    grouped_tract_df = census_df.groupby('geoid')
    for name, tract_df in grouped_tract_df:
        tract_array = tract_df.drop(columns='geoid').to_numpy()

        shape = tract_array.shape
        if shape != (NUM_YEAR, len(FEATURES) - 1):
            continue

        tract_array = tract_array.reshape(-1, shape[0], shape[1])
        if dataset is None:
            dataset = tract_array
        else:
            dataset = np.concatenate((dataset, tract_array), axis=2)

    return dataset


def format_sliding_window(dataset: np.ndarray, window: int) -> np.ndarray:
    """Format the dataset as required by LSTM layer, of shape (batch, timesteps, feature)"""

    formatted = None
    for i in range(0, NUM_YEAR - window):  # the final batch is used as y only
        if formatted is None:
            formatted = dataset[:, i: i + window, :]
        else:
            formatted = np.concatenate((formatted, dataset[:, i: i + window, :]), axis=0)

    return formatted


def prep_data(census_df: pd.DataFrame, window: int = 5) -> tuple:
    # normalisation
    list(map(
        lambda s: s.transform(),
        [pop_scalar, income_scalar, home_scalar]
    ))

    data = format_singlerow_year_tract(census_df)
    x = format_sliding_window(data, window)
    y = data[0, window:, :]

    return x, y


def build_model(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(LSTM(512, activation='relu', input_shape=input_shape, recurrent_dropout=0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(input_shape[1], activation='tanh'))
    model.compile(optimizer='adam', loss='mse')

    return model


def train_model(model, x, y, batch_size, epoch,
                save_file='./cache/predictor_model.hdf5', show_loss=False):
    history = model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=1)
    if show_loss:
        plt.plot(history.history['loss'])
        plt.ylabel('loss')
        plt.show()

    try:
        model.save(save_file)
    except OSError:
        os.mkdir('./cache')
        model.save(save_file)

    return model


x, y = prep_data(census_df)
# print(y.shape)
train_model(build_model(x.shape[1:]), x, y, 5, 10000, show_loss=True)
