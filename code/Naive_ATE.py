import pandas as pd
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner

# Helper imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# load
data = pd.read_csv('/MLAIM/AIMLab/ShanySheina/Causal_Inference/data/project_data.csv')


T = data['goal']
Y = data['norm_match']
X = data.drop(['goal', 'norm_match', 'match', 'iid'], axis=1)

#treatment to binary
T = T.astype('int')
T[(T != 3) & (T != 4)] = 0
T[(T == 3) | (T == 4)] = 1

Y_tr = Y[T == 1]
Y_utr = Y[T == 0]

Naive_ATE = np.mean(Y_tr)- np.mean(Y_utr)