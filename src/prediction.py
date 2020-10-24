import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
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

# normalisation
list(map(
    lambda s: s.transform(),
    [pop_scalar, income_scalar, home_scalar]
))


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
        dataset = np.concatenate((dataset, tract_array), axis=0) if dataset is not None else tract_array

    # print(dataset[0,:,0]
    return dataset


def prep_data(data: np.ndarray) -> tuple:
    x = data[:, :-1, 2:]  # use 2: to remove geoid and year from training data
    y = data[:, -1, 2:]  # use the find year as y
    # print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    return x_train, x_test, y_train, y_test


def build_model(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=input_shape, recurrent_dropout=0.2))
    model.add(Dense(512, activation='relu'))
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


def predict(model=None, iteration=10,  dataset_size=1000, cache_path='./cache/predictor_model.hdf5'):
    lookback = NUM_YEAR - 1
    data = format_tract_year_feat(census_df)
    num_tracts = data.shape[0]

    if model is None:
        try:
            print('[INFO] Using cached model.')
            model = keras.models.load_model(cache_path)
        except Exception:
            print('[INFO] No pre-trained model found, training a model for it now.')
            x_train, x_test, y_train, y_test = prep_data(data)
            model = train_model(build_model(x_train.shape[1:]), x_train, y_train, 2000, 80, show_loss=True)

    # prediction of future <iteration> readings, based on the last <lookback> values
    predictions = None
    for tract_data in data:
        x = np.copy(tract_data)
        geoid = x[0, 0]
        year = x[0, 1]
        for i in range(iteration):
            prediction = model.predict(x[:, 2:][-lookback:, :].reshape(1, lookback, -1))
            prediction = np.insert(prediction, 0, year, axis=1)
            prediction = np.insert(prediction, 0, geoid, axis=1)
            x = np.append(x, prediction, axis=0)
        x = x.reshape(1, -1, len(FEATURES))
        predictions = np.append(predictions, x, axis=0) if predictions is not None else x
        print(f'    {predictions.shape[0]} / {num_tracts}')

    return predictions


# x_train, x_test, y_train, y_test = prep_data(format_tract_year_feat(census_df))
# model = train_model(build_model(x_train.shape[1:]), x_train, y_train, 2000, 80, show_loss=True)
# score = model.evaluate(x=x_test, y=y_test)
# print(score[0], score[1])

predict()
