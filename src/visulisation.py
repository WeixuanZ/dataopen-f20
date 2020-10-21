import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import dataset


census: pd.DataFrame = dataset.load_census_data()

sns.heatmap(census.drop(columns=['geoid', 'state', 'county', 'tract']).corr())

plt.show()
