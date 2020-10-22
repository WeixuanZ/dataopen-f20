# pip install census

import pandas as pd
from census import Census
from census.core import CensusException

YOUR_API_KEY = 'df8eb2f48c10f50d30286eb54c804fe4f8878e87'
CENSUS_FILE_NAME = 'Census_education'
# variables = ['NAME', 'B01001_001E',  'B19013_001E', 'B25077_001E',
#              'B03002_003E', 'B03002_004E',  'B02001_004E', 'B03002_006E',
#              'B03002_007E', 'B03002_008E',  'B03002_009E', 'B03002_012E']

# CITIZENS, VOTING-AGE BY EDUCATIONAL ATTAINMENT (available in 2018+ only)
# variables = list(map(lambda n: f'B29002_00{n}E', range(1, 9)))

# Male:   Bachelor's, Master's, Professional, Doctorate
# Famele: Bachelor's, Master's, Professional, Doctorate
variables = ['B15002_015E', 'B15002_016E', 'B15002_017E', 'B15002_018E',
             'B15002_032E', 'B15002_033E', 'B15002_034E', 'B15002_035E']

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]


c = Census(YOUR_API_KEY)


nyc_met_area = [
    {"state_code": "34", "county_code": "003", "county_name": "Bergen, NJ"},
    {"state_code": "34", "county_code": "013", "county_name": "Essex, NJ"},
    {"state_code": "34", "county_code": "017", "county_name": "Hudson, NJ"},
    {"state_code": "34", "county_code": "019", "county_name": "Hunterdon, NJ"},
    {"state_code": "34", "county_code": "023", "county_name": "Middlesex, NJ"},
    {"state_code": "34", "county_code": "025", "county_name": "Monmouth, NJ"},
    {"state_code": "34", "county_code": "027", "county_name": "Morris, NJ"},
    {"state_code": "34", "county_code": "029", "county_name": "Ocean, NJ"},
    {"state_code": "34", "county_code": "031", "county_name": "Passaic, NJ"},
    {"state_code": "34", "county_code": "035", "county_name": "Somerset, NJ"},
    {"state_code": "34", "county_code": "037", "county_name": "Sussex, NJ"},
    {"state_code": "34", "county_code": "039", "county_name": "Union, NJ"},
    {"state_code": "36", "county_code": "005", "county_name": "Bronx, NY"},
    {"state_code": "36", "county_code": "027", "county_name": "Dutchess, NY"},
    {"state_code": "36", "county_code": "047", "county_name": "Kings, NY"},
    {"state_code": "36", "county_code": "059", "county_name": "Nassau, NY"},
    {"state_code": "36", "county_code": "061", "county_name": "New York, NY"},
    {"state_code": "36", "county_code": "071", "county_name": "Orange, NY"},
    {"state_code": "36", "county_code": "079", "county_name": "Putnam, NY"},
    {"state_code": "36", "county_code": "081", "county_name": "Queens, NY"},
    {"state_code": "36", "county_code": "085", "county_name": "Richmond, NY"},
    {"state_code": "36", "county_code": "087", "county_name": "Rockland, NY"},
    {"state_code": "36", "county_code": "103", "county_name": "Suffolk, NY"},
    {"state_code": "36", "county_code": "119", "county_name": "Westchester, NY"},
    {"state_code": "42", "county_code": "103", "county_name": "Pike, PA"}
]

dfCounties = pd.DataFrame(nyc_met_area)


def get_acs_data(c, variables, state_code, county_code, year):
    results = c.acs5.state_county_tract(
        variables,
        state_code,
        county_code,
        Census.ALL,
        year=year
    )
    return results


for year in years:
    print('Year: {}'.format(year))

    census_data = []
    for county in nyc_met_area:
        print('      ' + county["county_name"])
        try:
            census_data += get_acs_data(c,
                                        variables,
                                        county["state_code"],
                                        county["county_code"],
                                        year)
        except CensusException as e:
            print('Failed', e)

    if census_data:
        df = pd.DataFrame(census_data)

        # create geoid columns
        df['geoid'] = df['state'] + df['county'] + df['tract']
        df['year'] = year

        # move it to the begining of the dataframe
        col = df.pop("year")
        df.insert(0, col.name, col)

        col = df.pop("geoid")
        df.insert(0, col.name, col)

        df.to_csv('{}_{}.csv'.format(CENSUS_FILE_NAME, str(year)), index=False)
