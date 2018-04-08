
import os
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import SGDClassifier, LinearRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from utils.preprocessing import read_csv

def process(data):
    '''
    Author: Jordan Patterson

    Function to format pandas data into two datasets, data and labels

    Parameters
    ----------
    data: Object
        Contains the data parsed from the openDota.csv file with pandas

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
            matches_win[key].insert(0, data.iloc[row, 4])
            matches_win[key].insert(0, data.iloc[row, 3])
            matches_lose[key].append(data.iloc[row, 3])
            matches_lose[key].append(data.iloc[row, 4])
        else:
            matches_lose[key].insert(0, data.iloc[row, 4])
            matches_lose[key].insert(0, data.iloc[row, 3])
            matches_win[key].append(data.iloc[row, 3])
            matches_win[key].append(data.iloc[row, 4])

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


# tells pandas what data types the columns of the .csv file are
dtypes = {
    'match_id': int,
    'win': bool,
    'lane_role': float,
    'estimate': int,
    'score': float
}
print('Gathering data...' )
# get data from csv file
data = read_csv('data/openDota.csv', dtypes, True)

print('Processing data...')
# process data into required shapes (N, 20) and (N, )
data, labels = process(data)

# size of train partition
N = int(len(data) - (0.1 * len(data)))

# parameters and output placeholders
min_acc = 100
max_acc = 0
avg_acc = 0
num_iterations = 1000
best_model = RandomForestClassifier().fit(X=data[0:N, :], y=labels[0:N])
# generates models
print("Beginning training...")
for i in tqdm(range(num_iterations)):
    # create array of random values
    s = np.arange(data.shape[0])
    np.random.shuffle(s)

    # shuffle data
    labels = labels[s]
    data = data[s]

    # train on model
    model = RandomForestClassifier().fit(X=data[0:N, :], y=labels[0:N])
    # model = SGDClassifier(verbose=1, max_iter=1000).fit(X=data[0:N, :], y=labels[0:N])
    # model = LinearRegression(fit_intercept=False).fit(X=data[0:N, :], y=labels[0:N])
    # model = LogisticRegressionCV().fit(X=data[0:N, :], y=labels[0:N])

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

avg_acc /= num_iterations

print("Minimum accuracy: " + str(min_acc)[0:4] + "%")
print("Average accuracy: " + str(avg_acc)[0:4] + "%")
print("Best model accuracy: " + str(max_acc)[0:4] + "%")

# path to save your model
open('best_model.pkl', 'w')
path = os.getcwd() + '/best_model.pkl'
joblib.dump(best_model, path)
