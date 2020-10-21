import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import dataset


census: pd.DataFrame = dataset.load_census_data()

# remove correlation with population
for name in census.keys()[6:15]:
    census[name] /= census['population']

sns.heatmap(census.drop(columns=['geoid', 'state', 'county', 'tract']).corr())

plt.show()
