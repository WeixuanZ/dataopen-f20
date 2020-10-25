import pandas as pd
from util.dataset import compute_can_gentrify, compute_has_gentrified, has_gentrified
from util.geo import GeoData, get_boro_name, get_tract_id

START_YEAR = 2009  # 2019
END_YEAR = 2018  # 2028

DATA = pd.read_csv('../data/census_predict.csv')
DATA = compute_can_gentrify(DATA)
DATA = compute_has_gentrified(DATA, START_YEAR)
DATA.to_csv('../data/census_predict_gentrification.csv')

geo_data = GeoData()
feat_hashtable = geo_data.get_hashtable()

for _, row in DATA.iterrows():
    row = row.to_dict()

    if row['year'] != END_YEAR:
        continue

    geoid = row['geoid']
    row['has_gentrified'] = bool(has_gentrified(DATA, geoid, START_YEAR, END_YEAR))  # bool_ is not JSON serializable

    try:
        key = get_boro_name(geoid) + get_tract_id(geoid)
        index = feat_hashtable[key]
    except KeyError:
        continue

    print(f'{key} appended')
    geo_data.insert_data(index, row)

filename = f'{START_YEAR}_{END_YEAR}.geojson'
print(f'[INFO] Writing to {filename}')
geo_data.dump(filename)
