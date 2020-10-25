import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

from util.dataset import load_census_data
from util.preprocess import MinMaxScaler

np.random.seed(6)  # for reproducibility
FEATURES = ['geoid', 'state', 'county', 'year', 'population', 'household_income', 'home_value', 'pop_non_hispanic_caucasians',
            'pop_non_hispanic_blacks', 'pop_indians_alaskans', 'pop_non_hispanic_asians',
            'pop_non_hispanic_hawaiians_pacific', 'pop_non_hispanic_others', 'pop_non_hispanic_multi_racials',
            'pop_hispanics_latinos', 'pop_graduates']
NUM_YEAR = 10
census_df = load_census_data()[FEATURES]

pop_scalar = MinMaxScaler(census_df, 'population')
income_scalar = MinMaxScaler(census_df, 'household_income')
home_scalar = MinMaxScaler(census_df, 'home_value')

# normalisation
list(map(
    lambda s: s.transform(census_df),
    [pop_scalar, income_scalar, home_scalar]
))


def format_tract_year_feat(df: pd.DataFrame) -> np.ndarray:
    """Format an array of shape (tract, years, features).
    The features are the ones in FEATURES, starting with 'geoid'.
    """

    dataset = None
    grouped_tract_df = df.groupby('geoid')
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
    """Remove geoid and year, also split into training and testing sets."""
    x = data[:, :-1, 4:]  # use 4: to remove geoid and year from training data
    y = data[:, -1, 4:]  # use the final year as y
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
                save_model_path, show_loss=False):
    history = model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=1)
    if show_loss:
        plt.plot(history.history['loss'])
        plt.ylabel('loss')
        plt.show()

    try:
        model.save(save_model_path)
    except OSError:
        os.mkdir(os.path.dirname(save_model_path))
        model.save(save_model_path)

    return model


def predict(df, model=None, iteration=10, model_path='../model/predictor_model.hdf5', batch_size=2000, epoch=80):
    lookback = NUM_YEAR - 1
    data = format_tract_year_feat(df)
    num_tracts, _, num_features = data.shape

    if model is None:
        try:
            print('[INFO] Using cached model.')
            model = keras.models.load_model(model_path)
        except Exception:
            print('[INFO] No pre-trained model found, training a model for it now.')
            x_train, x_test, y_train, y_test = prep_data(data)
            model = train_model(build_model(x_train.shape[1:]), x_train, y_train, batch_size, epoch, model_path,
                                show_loss=True)
            score = model.evaluate(x=x_test, y=y_test)
            print(score[0], score[1])

    # prediction of future <iteration> readings, based on the last <lookback> values
    predictions = None
    for tract_data in data:
        x = np.copy(tract_data)
        for i in range(iteration):
            prediction = model.predict(x[:, 4:][-lookback:, :].reshape(1, lookback, -1))
            prediction = np.insert(prediction, 0, x[-1, 3] + 1, axis=1)  # year
            prediction = np.insert(prediction, 0, x[0, 2], axis=1)  # county
            prediction = np.insert(prediction, 0, x[0, 1], axis=1)  # state
            prediction = np.insert(prediction, 0, x[0, 0], axis=1)  # geoid
            x = np.append(x, prediction, axis=0)
        x = x.reshape(1, -1, num_features)
        predictions = np.append(predictions, x, axis=0) if predictions is not None else x
        print(f'    {predictions.shape[0]} / {num_tracts}')

    print('[INFO] Prediction completed.')
    return predictions


def convert_to_df(predictions: np.ndarray) -> pd.DataFrame:
    num_tracts, num_years, _ = predictions.shape
    predictions_flat = None
    for t in range(num_tracts):
        for y in range(num_years):
            feats = predictions[t, y, :].reshape(1, -1)
            predictions_flat = np.append(predictions_flat, feats, axis=0) if predictions_flat is not None else feats

    census_predict_df = pd.DataFrame(data=predictions_flat, columns=FEATURES)

    list(map(
        lambda s: s.reverse_transform(census_predict_df),
        [pop_scalar, income_scalar, home_scalar]
    ))

    return census_predict_df


census_predict_df = convert_to_df(predict(census_df))
print(census_predict_df.head())
census_predict_df.to_csv('../data/census_predict.csv')
print('[INFO] csv saved.')
