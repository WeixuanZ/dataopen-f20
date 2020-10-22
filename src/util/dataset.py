from os import path

import pandas as pd


def _load_census_year(directory: str, year: int) -> pd.DataFrame:
    main_df = pd.read_csv(path.join(directory, f'Census{year}.csv'))
    education_df = pd.read_csv(path.join(directory, f'Census_education_{year}.csv'))

    # We have data by sex and degree type, but we only care about the total number
    columns = filter(lambda col: col.startswith('B'), education_df.columns)
    education_df['pop_graduates'] = education_df[columns].sum(axis=1)

    return main_df.merge(education_df[['geoid', 'pop_graduates']], on='geoid', validate='one_to_one')


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


def _normalize_census_data(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize USD amounts
    grouped = df.groupby('geoid')

    for col in ('household_income', 'home_value'):
        df[f'relative_{col}'] = grouped[col].transform(lambda s: s / s.iloc[0])

    # Normalize population composition
    population = df['population']

    for col in df.columns:
        if col.startswith('pop_'):
            df[col] /= population

    return df


def load_census_data(normalize: bool = True) -> pd.DataFrame:
    current_dir = path.dirname(path.realpath(__file__))
    data_dir = path.join(current_dir, '../../data/')

    dfs = [_load_census_year(data_dir, year) for year in range(2009, 2018 + 1)]

    census = pd.concat(map(_clean_census_data, dfs))

    census.rename(columns={
        'B01001_001E': 'population',  # 'Total Population'
        'B19013_001E': 'household_income',  # 'Median household income (in dollars)'
        'B25077_001E': 'home_value',  # 'Median home value (in dollars)'
        'B03002_003E': 'pop_non_hispanic_caucasians',  # 'Number of non-Hispanic Caucasians'
        'B03002_004E': 'pop_non_hispanic_blacks',  # 'Number of non-Hispanic blacks or African Americans'
        'B02001_004E': 'pop_indians_alaskans',  # 'Number of American Indians and Alaskans'
        'B03002_006E': 'pop_non_hispanic_asians',  # 'Number of non-Hispanic Asians'
        'B03002_007E': 'pop_non_hispanic_hawaiians_pacific',  # 'Number of non-Hispanic Hawaiians or Pacific Islanders'
        'B03002_008E': 'pop_non_hispanic_others',  # 'Number of non-Hispanic others'
        'B03002_009E': 'pop_non_hispanic_multi_racials',  # 'Number of non-Hispanic multi-racials'
        'B03002_012E': 'pop_hispanics_latinos',  # 'Number of Hispanics or Latinos'
    }, inplace=True)

    if normalize:
        census = _normalize_census_data(census)

    census.drop(columns='Unnamed: 0', inplace=True)
    census.reset_index(drop=True, inplace=True)

    return census
