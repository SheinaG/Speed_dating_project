import pandas as pd
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
import random
# Helper imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
from scipy.stats import norm




#T_learner
def clc_T_learner(X_train,y_train,T_train,X_test):
    models = GradientBoostingRegressor(n_estimators=10, max_depth=3, min_samples_leaf=1)
    T_learner = TLearner(models=models)
    # Train T_learner
    T_learner.fit(y_train, T_train, X=X_train, inference='bootstrap')
    # Estimate treatment effects on test data
    T_te = T_learner.effect(X_test)
    print('T learner: ', np.average(T_te))
    return T_learner

#S_learner
def clc_S_learner(X_train,y_train,T_train,X_test):
    overall_model = GradientBoostingRegressor(n_estimators=10, max_depth=3, min_samples_leaf=1)
    S_learner = SLearner(overall_model=overall_model)
    # Train S_learner
    S_learner.fit(y_train, T_train, X=X_train, inference='bootstrap')
    # Estimate treatment effects on test data
    S_te = S_learner.effect(X_test)
    print('S learner: ', np.average(S_te))
    return S_learner


#X_learner
def clc_X_learner(X_train,y_train,T_train,X_test):
    models = GradientBoostingRegressor(n_estimators=10, max_depth=3, min_samples_leaf=1)
    propensity_model = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_leaf=1)
    X_learner = XLearner(models=models, propensity_model=propensity_model)
    # Train X_learner
    X_learner.fit(y_train, T_train, X=X_train, inference='bootstrap')
    # Estimate treatment effects on test data
    X_te = X_learner.effect(X_test)
    print('X learner: ', np.average(X_te))
    return X_learner

def calc_confidence_interval_of_array(x, confidence = 0.95):
  m = x.mean()
  s = x.std()
  confidence = confidence
  crit = np.abs(norm.ppf((1 - confidence) / 2))
  return (m - s * crit / np.sqrt(len(x)), m + s * crit / np.sqrt(len(x)))


if __name__ == '__main__':
    # load
    data = pd.read_csv('/MLAIM/AIMLab/ShanySheina/Causal_Inference/data/project_data.csv')

    T = data['goal']
    y = data['norm_match']
    X = data.drop(['goal', 'norm_match', 'match', 'iid'], axis=1)

    # treatment to binary
    T = T.astype('int')
    T[(T != 3) & (T != 4)] = 0
    T[(T == 3) | (T == 4)] = 1

    # stratification by T
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(X, y, T, test_size=0.2, stratify=T)

    T_learner = clc_T_learner(X_train, y_train, T_train, X_test)
    S_learner = clc_S_learner(X_train, y_train, T_train, X_test)
    X_learner = clc_X_learner(X_train, y_train, T_train, X_test)

    T_l = []
    S_l = []
    X_l = []

    #calc_CI:
    for i in range(0,100):
        chosen_idx = random.sample(range(0, len(X_test)), int(len(X_test) * 0.8))
        X_test_c = X_test.iloc[chosen_idx, :]
        T_l.append(T_learner.effect(X_test_c))
        S_l.append(S_learner.effect(X_test_c))
        X_l.append(X_learner.effect(X_test_c))

    print(calc_confidence_interval_of_array(np.asarray(T_l), confidence=0.95))
    print(calc_confidence_interval_of_array(np.asarray(S_l), confidence=0.95))
    print(calc_confidence_interval_of_array(np.asarray(X_l), confidence=0.95))