import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

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


def format_tract_year_feat(census_df: pd.DataFrame) -> np.ndarray:
    """Format an array of shape (tract, years, features)

    The features are the ones in FEATURES, starting with 'geoid'
    """

    dataset = None
    grouped_tract_df = census_df.groupby('geoid')
    for name, tract_df in grouped_tract_df:
        tract_array = tract_df.to_numpy()

        shape = tract_array.shape
        if shape != (NUM_YEAR, len(FEATURES)):
            continue

        tract_array = tract_array.reshape(-1, shape[0], shape[1])
        if dataset is None:
            dataset = tract_array
        else:
            dataset = np.concatenate((dataset, tract_array), axis=0)

    # print(dataset[0,:,0]
    return dataset


def prep_data(census_df: pd.DataFrame) -> tuple:
    # normalisation
    list(map(
        lambda s: s.transform(),
        [pop_scalar, income_scalar, home_scalar]
    ))

    data = format_tract_year_feat(census_df)
    x = data[:, :-1, 2:]  # use 2: to remove geoid and year from training data
    y = data[:, -1, 2:]  # use the find year as y
    # print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test


def build_model(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(LSTM(512, activation='relu', input_shape=input_shape, recurrent_dropout=0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(input_shape[1], activation='tanh'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

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


x_train, x_test, y_train, y_test = prep_data(census_df)
# print(y.shape)
model = train_model(build_model(x_train.shape[1:]), x_train, y_train, 500, 30, show_loss=True)
score = model.evaluate(x=x_test, y=y_test)
print(score[0], score[1])
