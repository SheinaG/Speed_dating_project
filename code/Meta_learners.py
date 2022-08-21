import pandas as pd
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner

# Helper imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# load


data = pd.read_csv('/MLAIM/AIMLab/ShanySheina/Causal_Inference/data/project_data.csv')


T = data['goal']
y = data['norm_match']
X = data.drop(['goal', 'norm_match', 'match', 'iid'], axis=1)

#treatment to binary
T = T.astype('int')
T[(T != 3) & (T != 4)] = 0
T[(T == 3) | (T == 4)] = 1


# stratification by T
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(X, y, T, test_size=0.2, stratify=T)

#T_learner
models = GradientBoostingRegressor(n_estimators=10, max_depth=3, min_samples_leaf=1)
T_learner = TLearner(models=models)
# Train T_learner
T_learner.fit(y_train, T_train, X=X_train)
# Estimate treatment effects on test data
T_te = T_learner.effect(X_test)
print('T learner: ', np.average(T_te))

#S_learner
overall_model = GradientBoostingRegressor(n_estimators=10, max_depth=3, min_samples_leaf=1)
S_learner = SLearner(overall_model=overall_model)
# Train S_learner
S_learner.fit(y_train, T_train, X=X_train)
# Estimate treatment effects on test data
S_te = S_learner.effect(X_test)
print('S learner: ', np.average(S_te))


#X_learner
models = GradientBoostingRegressor(n_estimators=10, max_depth=3, min_samples_leaf=1)
propensity_model = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_leaf=1)
X_learner = XLearner(models=models, propensity_model=propensity_model)
# Train X_learner
X_learner.fit(y_train, T_train, X=X_train)
# Estimate treatment effects on test data
X_te = X_learner.effect(X_test)
print('X learner: ', np.average(X_te))