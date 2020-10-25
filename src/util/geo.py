import json
from os import path

import pandas as pd

CENSUS = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), '../../data/Census2010.csv'))[
    ['geoid', 'NAME']].groupby('geoid').first().reset_index()


def get_boro_name(id: int) -> str:
    return CENSUS[CENSUS.geoid == id]['NAME'].to_string().split(',')[1].replace('County', '').strip()


class GeoData:
    @staticmethod
    def dump(data: dict) -> None:
        with open("data_file.geojson", "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(name: str) -> dict:
        current_dir = path.dirname(path.realpath(__file__))
        data_dir = path.join(current_dir, '../../data/')
        with open(path.join(data_dir, name), "r") as f:
            return json.load(f)

    def __init__(self, name: str = '2010CensusTracts.geojson'):
        self.data = GeoData.load(name)

    def get_features(self) -> list:
        return self.data['features']

    @staticmethod
    def get_boro_name(feat_list: dict) -> str:
        return feat_list['boro_name']

    def insert_data(self, index: int, data: dict) -> None:
        self.get_features()[index].update(data)
