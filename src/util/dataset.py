from os import path

import cpi
import numpy as np
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
    # Adjust for inflation
    cpi.update()

    df['home_value'] = df.apply(lambda x: cpi.inflate(x['home_value'], x['year']), axis=1)
    df['household_income'] = df.apply(lambda x: cpi.inflate(x['household_income'], x['year']), axis=1)

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


def compute_can_gentrify(df: pd.DataFrame) -> pd.DataFrame:
    """See https://www.governing.com/gov-data/gentrification-report-methodology.html"""
    # Population at least 500
    df['big_population'] = df.groupby('geoid')['population'].transform(lambda series: (series > 500).all())

    # Median household income in the bottom 40th percentile of its metro area
    income_threshold = df.groupby(['year', 'state', 'county'])['household_income'].quantile(0.40)
    df = df.merge(income_threshold.rename('income_threshold'), on=['year', 'state', 'county'], validate='many_to_one')

    df['low_income'] = df['household_income'] < df['income_threshold']

    # Median home value in the bottom 40th percentile of its metro area
    value_threshold = df.groupby(['year', 'state', 'county'])['home_value'].quantile(0.40)
    df = df.merge(value_threshold.rename('value_threshold'), on=['year', 'state', 'county'], validate='many_to_one')

    df['low_value'] = df['home_value'] < df['value_threshold']

    # Can gentrify if all three conditions are met simultaneously
    df['can_gentrify'] = df['big_population'] & df['low_income'] & df['low_value']

    return df


def compute_has_gentrified(df: pd.DataFrame, starting_year: int) -> pd.DataFrame:
    """See https://www.governing.com/gov-data/gentrification-report-methodology.html"""
    def pick_year(df: pd.DataFrame):
        if (df.year == starting_year).any():
            return df[df['year'] == starting_year].iloc[0]
        return np.nan

    # An increase in a tract's educational attainment, as measured by the percentage of residents
    # age 25 and over holding bachelor’s degrees, was in the top third percentile of all tracts within a metro area
    gs = df.groupby('geoid')[['year', 'pop_graduates']].agg(pick_year)['pop_graduates']
    df = df.merge(gs.rename('pop_graduates_start'), on='geoid', validate='many_to_one')

    df['pop_graduates_change'] = (df['pop_graduates'] / df['pop_graduates_start']) - 1

    gct = df.groupby(['year', 'state', 'county'])['pop_graduates_change'].quantile(0.66)
    df = df.merge(gct.rename('pop_graduates_change_threshold'), on=['year', 'state', 'county'], validate='many_to_one')

    df['pop_graduates_big_change'] = df['pop_graduates_change'] > df['pop_graduates_change_threshold']

    # A tract’s median home value increased when adjusted for inflation
    hv = df.groupby('geoid')[['year', 'home_value']].agg(pick_year)['home_value']
    df = df.merge(hv.rename('home_value_start'), on='geoid', validate='many_to_one')

    df['home_value_increase'] = (df['home_value'] / df['home_value_start']) - 1

    # The percentage increase in a tract’s median home value
    # was in the top third percentile of all tracts within a metro area
    hvt = df.groupby(['year', 'state', 'county'])['home_value_increase'].quantile(0.66)
    df = df.merge(hvt.rename('home_value_increase_threshold'), on=['year', 'state', 'county'], validate='many_to_one')

    df['home_value_big_increase'] = ((df['home_value_increase'] > 0) & (
        df['home_value_increase'] > df['home_value_increase_threshold']))

    # Has gentrified if all three conditions are met simultaneously and year is not in the past
    df['has_gentrified'] = df['pop_graduates_big_change'] & df['home_value_big_increase'] & (df['year'] > starting_year)

    return df


def has_gentrified(df: pd.DataFrame, geoid: int, start_year, end_year) -> bool:
    """Make sure you've already called compute_can_gentrify and compute_has_gentrified!"""
    return df[(df['geoid'] == geoid) & (df['year'] == start_year)]['can_gentrify'].all() and df[(df['geoid'] == geoid) & (df['year'] == end_year)]['has_gentrified'].all()


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
