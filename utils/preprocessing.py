
import os.path, requests, json, sys
import pandas as pd

def get_hero_data():
    # get hero stats in JSON
    r = requests.get('https://api.opendota.com/api/heroStats')
    raw_json = json.loads(r.text)

    # calculate win rate for each hero
    hero_stats = {}
    for data in raw_json:
        pick = int(data['1_pick']) + int(data['2_pick']) + int(data['3_pick']) + int(data['4_pick']) + int(data['5_pick']) + int(data['6_pick']) + int(data['7_pick'])
        win = int(data['1_win']) + int(data['2_win']) + int(data['3_win']) + int(data['4_win']) + int(data['5_win']) + int(data['6_win']) + int(data['7_win'])
        hero_stats[data['id']] = win/pick

    # convert dictionary to JSON
    hero_stats = json.loads(json.dumps(hero_stats))

    # write JSON to heroWinRates.json
    with open('heroWinRates.json', 'w') as outfile:
        json.dump(hero_stats, outfile)


def get_openDota_data():
    '''
    Author: Jordan Patterson

    Function to get data from openDota api and write to openDota.csv

    '''


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