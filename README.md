# Description

A neural network to predict the outcome of a DOTA2 match before it has begun

#### Requirements

* a working Python 3 development environment
* [pip3](https://pip.pypa.io/en/latest/installing.html) to install Python dependencies (must be latest version -> pip install --upgrade pip)
* [pipenv](https://github.com/pypa/pipenv) to manage dependencies

#### Pipfile Requirements

* [scikit-learn](http://pytorch.org/) for models
* [tqdm](https://pypi.python.org/pypi/tqdm) to view progress throughout runtime

pipenv will install all of the Pipfile required packages.

To do so, run the following command:
```
pipenv install
```

#### Dataset

CSV file consisting of the following columns: 'match_id', 'win', 'lane_role', 'estimate', 'score'

A valid match will have 10 identical match_ids, as each represents a players data in the match

prediction.py uses 'win', 'estimate', and 'score' for each player, grouping players on a team by whether they won/lost in a given match

'win' is boolean True/False

'estimate' is an estimation of the players MMR

'score' is determined by OpenDota, based on how good a player has been playing the past several games
