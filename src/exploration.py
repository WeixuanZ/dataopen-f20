import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from analysis.dim_reduce import get_feature_importance
from util.dataset import load_census_data

census: pd.DataFrame = load_census_data()

# Correlation matrix
plt.figure()
sns.heatmap(census.drop(columns=['geoid', 'state', 'county', 'tract']).corr())

# Feature importance in predicting y using Random Forest
y = 'home_value'
discard = ['NAME', 'relative_household_income', 'relative_home_value', 'tract', 'county', 'state', 'geoid']
plt.figure()
importance = sns.barplot(
    census.columns.drop([y] + discard),
    get_feature_importance(census.drop(columns=discard), y),
)
importance.set_xticklabels(
    importance.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# Clusters
plt.figure()
sns.scatterplot(census[census['year'] == 2018]['household_income'], census[census['year'] == 2018]['home_value'])

plt.show()
