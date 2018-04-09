
import requests, json, os
import pandas as pd

from random import randint


def get_data(normalize=False):
    '''
    Author: Jordan Patterson

    Function to set parameters on data and get it

    Parameters
    ----------

        Optional
        ---------
        normalize: boolean
            Whether data to be returned should be normalized or not
    
    Returns
    ----------
    data: Object
        Contains the data requested ('match_id', 'win', 'estimate', 'score')

    '''

    # specify number of matches to parse
    matches = 50000
    # find most recent matches
    max_match = call_api('https://api.opendota.com/api/explorer?sql=SELECT%0Amax(match_id)%0AFROM%20matches')['max'][0]

    # get N matches in range < match_id
    data = call_api('https://api.opendota.com/api/explorer?sql=SELECT%0Amatches.match_id%2C%0A((player_matches.player_slot%20%3C%20128)%20%3D%20matches.radiant_win)%20win%2C%0Ammr_estimates.estimate%2C%0Ahero_ranking.score%0AFROM%20matches%0AJOIN%20player_matches%20using(match_id)%0AJOIN%20heroes%20on%20heroes.id%20%3D%20player_matches.hero_id%0AJOIN%20mmr_estimates%20on%20mmr_estimates.account_id%20%3D%20player_matches.account_id%0AJOIN%20hero_ranking%20on%20(hero_ranking.hero_id%20%3D%20heroes.id%20and%20hero_ranking.account_id%20%3D%20player_matches.account_id)%0AWHERE%20match_id%20%3C%20' + str(max_match) + '%0AORDER%20BY%20matches.match_id%20DESC%20NULLS%20LAST%0ALIMIT%20' + str(matches))

    data.to_csv("openDotaNew.csv")

    # normalize data
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())
    
    return data

    
def call_api(api):
    '''
    Author: Jordan Patterson

    Function to get data from OpenDota api

    Parameters
    ----------
    api: string
        The api being called
    
    Returns
    ----------
    data: Object
        Contains the data requested in pandas table

    '''

    # call endpoint
    r = requests.get(api)

    # ensure successful call
    while r.status_code != 200:
        print('Failed to access OpenDota servers: try decreasing number of matches')
        r = requests.get(api)

    # convert to json
    json = r.json()

    # put data in pandas format after parsing JSON
    return pd.DataFrame(json['rows'], columns=[x['name'] for x in json['fields']])

    
def read_csv(name, dtypes, normalize=False):
    '''
    Author: Jordan Patterson

    Function to search csv files for desired information

    Parameters
    ----------
    name: string
        Name/path of the .csv file being parsed
    
    dtypes: dictionary
        Key / value pair specifying type of each column/header
    
        Optional
        ---------
        normalize: boolean
            Whether data to be returned should be normalized or not

    Returns
    ----------
    data: Object
        Contains the data requested

    '''

    # ensures valid .csv file was passed
    if os.path.isfile(name) and name[name.rfind('.'):] == '.csv':
        
        # reads .csv into table
        data = pd.read_csv(name, dtype=dtypes)

        # normalize data
        if normalize:
            data = (data - data.min()) / (data.max() - data.min())

        return data

    else:
        print("Error: invalid .csv file")
        sys.exit(0)
