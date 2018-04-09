
import os, argparse
import numpy as np

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from utils.preprocessing import get_data, read_csv


def process(data):
    '''
    Author: Jordan Patterson

    Function to format pandas data into two datasets, data and labels

    Parameters
    ----------
    data: Object
        Contains the data parsed from the OpenDota api

    Returns
    ----------
    data: npArray
        Shape (N, 20), where N is number of matches, and 20 is 10 players 'estimate' and 'score' data

    labels: npArray
        Shape (N,) where N in number of matches, and contains 1.0 if first team won, and 0.0 if they lost

    '''

    matches_win = {}
    matches_lose = {}
    # loops through .csv file
    for row in tqdm(range(len(data))):
        # sets match_id column to key, and data type to list
        key = data.iloc[row, 0]
        matches_win.setdefault(key, [])
        matches_lose.setdefault(key, [])
        # group data based on outcome
        if data.iloc[row, 1] == 1.0:
            matches_win[key].insert(0, data.iloc[row, 3])
            matches_win[key].insert(0, data.iloc[row, 2])
            matches_lose[key].append(data.iloc[row, 2])
            matches_lose[key].append(data.iloc[row, 3])
        else:
            matches_lose[key].insert(0, data.iloc[row, 3])
            matches_lose[key].insert(0, data.iloc[row, 2])
            matches_win[key].append(data.iloc[row, 2])
            matches_win[key].append(data.iloc[row, 3])

    # loop through all matches
    for match in list(matches_win.keys()):
        # remove incomplete matches (less than 10 players -> 10*2 features)
        if len(matches_win[match]) != 20 or len(matches_lose[match]) != 20:
            del matches_win[match]
            del matches_lose[match]

    # create labels
    labels_win = np.ones(np.asarray(list(matches_win)).shape[0])
    labels_lose = np.zeros(np.asarray(list(matches_lose)).shape[0])
    labels = np.concatenate((labels_win, labels_lose), axis=0)
    # merge match datasets
    data = np.asarray(list(matches_win.values()) + list(matches_lose.values()))

    return data, labels


def train(data, labels):
    '''
    Author: Jordan Patterson

    Function to train models

    Parameters
    ----------
    data: nparray
        Contains the data parsed from the OpenDota api

    labels: nparray
        Contains the labels for the parsed data

    Returns
    ----------
    model: Object
        The model created with the highest accuracy

    '''

    # size of train partition
    N = int(len(data) - (0.1 * len(data)))
    # parameters and output placeholders
    min_acc = 100
    max_acc = 0
    avg_acc = 0
    num_epocs = 100
    best_model = RandomForestClassifier().fit(X=data[0:N, :], y=labels[0:N])
    # generates models
    print("Beginning training...")
    for i in tqdm(range(num_epocs)):
        # create array of random values
        s = np.arange(data.shape[0])
        np.random.shuffle(s)

        # shuffle data
        labels = labels[s]
        data = data[s]

        # train on model
        model = RandomForestClassifier().fit(X=data[0:N, :], y=labels[0:N])

        # create test data from data split
        test_data = data[N:, :]
        test_labels = labels[N:]

        # get test accuracy
        acc = model.score(test_data, test_labels) * 100

        # update output
        if acc > max_acc:
            max_acc = acc
            best_model = model
        if acc < min_acc:
            min_acc = acc
        
        avg_acc += acc

    avg_acc /= num_epocs

    print("Minimum accuracy: " + str(min_acc)[0:4] + "%")
    print("Average accuracy: " + str(avg_acc)[0:4] + "%")
    print("Best model accuracy: " + str(max_acc)[0:4] + "%")
    return best_model


def test(model, new, dtypes):
    '''
    Author: Jordan Patterson

    Function to test live data on a pretrained model

    Parameters
    ----------
    model: Object
        Any scikit learn generated model

    new: boolean
        Whether we trained on new data from OpenDota or not

    dtypes: dictionary
        Key / value pair specifying type of each column/header

    '''

    print("Getting test data...")
    # don't test on same data we trained on
    if new == True:
        data = read_csv('data/openDota.csv', dtypes, True)
    else:
        data = get_data(True)
    
    print("Processing test data...")
    data, labels = process(data)

    print("Number of complete matches: " + str(data.shape[0]))

    # create array of random values
    s = np.arange(data.shape[0])
    np.random.shuffle(s)

    # shuffle data
    test_labels = labels[s]
    test_data = data[s]

    # get test accuracy
    acc = model.score(test_data, test_labels) * 100
    print("Test accuracy: " + str(acc))


def main():
    """The main function."""

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', action='store_true')
    args = parser.parse_args()

    # tells pandas what data types the columns of the .csv file are
    dtypes = {
        'match_id': int,
        'win': bool,
        'estimate': int,
        'score': float
    }

    # use new data
    if args.new == True:
        print('Using new OpenDota api data')
        print('Gathering data...' )
        # from preprocessing.py
        data = get_data(True)
    
    # defaults to old data
    else:
        print('Using openDota.csv data')
        print('Gathering data...' )
        # get data from csv file
        data = read_csv('data/openDota.csv', dtypes, True)

    print('Processing data...')
    # process data into required shapes (N, 20) and (N, )
    data, labels = process(data)

    print("Number of complete matches: " + str(data.shape[0]))
    while data.shape[0] < 1000:
        print("Not enough matches, trying again")
        data = get_data(True)
        data, labels = process(data)

    best_model = train(data, labels)

    print("Beginning testing on new data...")
    test(best_model, args.new, dtypes)

    # path to save your model
    open('best_model.pkl', 'w')
    path = os.getcwd() + '/best_model.pkl'
    joblib.dump(best_model, path)


if __name__ == "__main__":
    main()
