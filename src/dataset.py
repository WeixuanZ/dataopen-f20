import pandas as pd


def _clean_census_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes N/A, max and negative values"""
    df.dropna(axis='index', how='any', inplace=True)

    bad_geoids = set()

    # ('household_income', 'home_value')
    for col in ('B19013_001E', 'B25077_001E'):
        big = (df[col] == df[col].max())
        neg = (df[col] < 0)

        geoids = df[big | neg].geoid
        bad_geoids.update(geoids)

    return df[~df.geoid.isin(bad_geoids)]


def load_census_data() -> pd.DataFrame:
    files = map(lambda year: f'../data/Census{year}.csv', range(2009, 2019))
    dfs = map(pd.read_csv, files)

    census = pd.concat(map(_clean_census_data, dfs))

    census.rename(columns={
        'B01001_001E': 'population',  # 'Total Population'
        'B19013_001E': 'household_income',  # 'Median household income (in dollars)'
        'B25077_001E': 'home_value',  # 'Median home value (in dollars)'
        'B03002_003E': 'non_hispanic_caucasians',  # 'Number of non-Hispanic Caucasians'
        'B03002_004E': 'non_hispanic_blacks',  # 'Number of non-Hispanic blacks or African Americans'
        'B02001_004E': 'indians_alaskans',  # 'Number of American Indians and Alaskans'
        'B03002_006E': 'non_hispanic_asians',  # 'Number of non-Hispanic Asians'
        'B03002_007E': 'non_hispanic_hawaiians_pacific',  # 'Number of non-Hispanic Hawaiians or Pacific Islanders'
        'B03002_008E': 'non_hispanic_others',  # 'Number of non-Hispanic others'
        'B03002_009E': 'non-hispanic_multi_racials',  # 'Number of non-Hispanic multi-racials'
        'B03002_012E': 'hispanics_latinos',  # 'Number of Hispanics or Latinos'
    }, inplace=True)

    census.drop(columns='Unnamed: 0', inplace=True)
    census.reset_index(drop=True, inplace=True)

    return census
