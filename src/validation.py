import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from util.preprocess import MinMaxScaler

FEATURES = ['geoid', 'year', 'population', 'household_income', 'home_value', 'pop_non_hispanic_caucasians',
            'pop_non_hispanic_blacks', 'pop_indians_alaskans', 'pop_non_hispanic_asians',
            'pop_non_hispanic_hawaiians_pacific', 'pop_non_hispanic_others', 'pop_non_hispanic_multi_racials',
            'pop_hispanics_latinos', 'pop_graduates']

data = pd.read_csv('../data/census_predict.csv').drop(columns='Unnamed: 0')

pop_scalar = MinMaxScaler(data, 'population')
income_scalar = MinMaxScaler(data, 'household_income')
home_scalar = MinMaxScaler(data, 'home_value')

# normalisation
list(map(
    lambda s: s.transform(data),
    [pop_scalar, income_scalar, home_scalar]
))

x_train = data[filter(lambda x: x != 'home_value', FEATURES)][data.year < 2019].to_numpy()
y_train = data[['home_value']][data.year < 2019].to_numpy().reshape(-1)

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(x_train, y_train)

x_test = data[filter(lambda x: x != 'home_value', FEATURES)][data.year > 2018].to_numpy()
y_test = data[['home_value']][data.year > 2018].to_numpy().reshape(-1)

y_pred = regr.predict(x_test)

mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

print(mse)
