# Description

Uses scikit-learns random forest classifier to predict the outcome of a DOTA2 match before it has begun

To run:

```
python3 prediction.py
```

Optional arguments:

```
python3 prediction.py --new
```

This specifies to train on new data from OpenDota, rather than the default data in openDota.csv

However, testing will then be done on openDota.csv to avoid bias (if training is done on openDota.csv, testing is done on new OpenDota data)


#### Requirements

* a working Python 3.6 development environment
* [pip3](https://pip.pypa.io/en/latest/installing.html) to install Python dependencies (must be latest version -> pip install --upgrade pip)
* [pipenv](https://github.com/pypa/pipenv) to manage dependencies

#### Pipfile Requirements

* [requests](http://docs.python-requests.org/en/master/) to get the data
* [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html) to parse data
* [scikit-learn](http://pytorch.org/) for the random regression classifier
* [tqdm](https://pypi.python.org/pypi/tqdm) to view progress throughout runtime

pipenv will install all of the Pipfile required packages.

To do so, run the following command:
```
pipenv install
```

#### Dataset

Uses the [OpenDota](https://docs.opendota.com/) API

Gets 'match_id', 'win', 'estimate', 'score' with an api call

A valid match will have 10 identical match_ids, as each represents a players data in the match

prediction.py uses 'win', 'estimate', and 'score' for each player, grouping players on a team by whether they won/lost in a given match

'win' is True/False

'estimate' is an estimation of the players MMR

'score' is determined by OpenDota, based on how good a player has been playing the past several games

#### Results

Training on openDota.csv and testing on new data from the OpenDota server, accuracy is consistently ~69%

Training on new data from the OpenDota server and testing on openDota.csv, accuracy is consistently ~65%

Unfortunately, we were unable to test on the new OpenDota data with the OpenDota trained model, as we could not find enough complete and different matches to get reliable accuracy
