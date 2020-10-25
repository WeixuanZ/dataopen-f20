import json
from os import path

import pandas as pd

CENSUS = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), '../../data/Census2010.csv'))[
    ['geoid', 'NAME']].groupby('geoid').first().reset_index()


def get_boro_name(geo_id: int) -> str:
    return CENSUS[CENSUS.geoid == geo_id]['NAME'].to_string().split(',')[1].replace('County', '').strip()


def get_tract_id(geo_id: int) -> str:
    return str(int(geo_id))[5:]


class GeoData:
    current_dir = path.dirname(path.realpath(__file__))
    data_dir = path.join(current_dir, '../../data/')

    def __init__(self, name: str = '2010CensusTracts.geojson'):
        with open(path.join(GeoData.data_dir, name), "r") as f:
            self.data = json.load(f)

    def __repr__(self):
        return self.data

    def dump(self, name: str = "data_file.geojson") -> None:
        with open(path.join(GeoData.data_dir, name), "w") as f:
            json.dump(self.data, f)

    def get_features(self) -> list:
        return self.data['features']

    @staticmethod
    def get_boro_name(feat: dict) -> str:
        return feat['properties']['boro_name']

    @staticmethod
    def get_tract_id(feat: dict) -> str:
        return feat['properties']['ct2010']

    def insert_data(self, index: int, data: dict) -> None:
        """Insert data from a dictionary to the feature dictionary at the given index"""
        self.get_features()[index]['properties'].update(data)

    def get_hashtable(self) -> dict:
        """Returns a dictionary mapping boro name to feature index"""
        return {GeoData.get_boro_name(feat) + GeoData.get_tract_id(feat): index for index, feat in
                enumerate(self.get_features())}
