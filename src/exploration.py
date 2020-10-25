import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from analysis.dim_reduce import get_feature_importance
from util.dataset import load_census_data

census: pd.DataFrame = load_census_data()

# Correlation matrix
plt.figure()
# sns.heatmap(census.drop(columns=['geoid', 'state', 'county', 'tract']).corr())

# Feature importance in predicting y using Random Forest
y = 'home_value'
FEATURES = ['year', 'population', 'household_income', 'home_value', 'pop_non_hispanic_caucasians',
            'pop_non_hispanic_blacks', 'pop_indians_alaskans', 'pop_non_hispanic_asians',
            'pop_non_hispanic_hawaiians_pacific', 'pop_non_hispanic_others', 'pop_non_hispanic_multi_racials',
            'pop_hispanics_latinos', 'pop_graduates']
x = list(filter(lambda f: f != y, FEATURES))
print(census[FEATURES].head())
importance = get_feature_importance(census[FEATURES], y)
print(importance)

plt.figure()
importance_plot = sns.barplot(
    x,
    importance
)
importance_plot.set_xticklabels(
    importance_plot.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# Clusters
plt.figure()
sns.scatterplot(census[census['year'] == 2018]['household_income'], census[census['year'] == 2018]['home_value'])

plt.show()
