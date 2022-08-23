import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

# Avoid printing dataconversion warnings from sklearn and numpy
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

data = pd.read_csv('/MLAIM/AIMLab/ShanySheina/Causal_Inference/data/project_data.csv')

# features groups
T = data['goal']
Y = data['norm_match']
X = data.drop(['match','norm_match', 'goal', 'iid'], axis=1)

T = T.astype('int')
T[(T != 3) & (T != 4)] = 0
T[(T == 3) | (T == 4)] = 1



    #Calculate propensity score:
scaler = StandardScaler().fit(X)
data_cs = scaler.transform(X)
ps_model = LogisticRegression().fit(data_cs, T)
data_ps = data.assign(propensity_score=ps_model.predict_proba(data_cs)[:, 1])

ps = data_ps['propensity_score'].values
#Trim:
up_trim = np.min([np.max(ps[T == 1]), np.max(ps[T == 0])])
low_trim = np.max([np.min(ps[T == 1]), np.min(ps[T == 0])])
ind = (ps < up_trim) * (ps > low_trim)
T_ps = T[ind]
Y_ps = Y[ind]
ps = ps[ind]

#Calculate ATE with IPW
weight = (T_ps - ps)/(ps*(1-ps))
IPW_ate = np.mean(weight * Y_ps)


#Preaper dayta for Matching
treated = X[T == 1]
untreated = X[T == 0]
Y_tr = Y[T == 1]
Y_utr = Y[T == 0]
feat = X.columns


m0 = KNeighborsRegressor(n_neighbors=1).fit(untreated[feat], Y_utr)
m1 = KNeighborsRegressor(n_neighbors=1).fit(treated[feat], Y_tr)

# fit the linear regression model to estimate mu_0(x)
ols0 = LinearRegression().fit(untreated[feat], Y_utr)
ols1 = LinearRegression().fit(treated[feat], Y_tr)

# find the units that match to the treated
treated_match_index = m0.kneighbors(treated[feat], n_neighbors=1)[1].ravel()

# find the units that match to the untreatd
untreated_match_index = m1.kneighbors(untreated[feat], n_neighbors=1)[1].ravel()

treated['match'] = m0.predict(treated[feat])
untreated['match'] = m1.predict(untreated[feat])

treated['bias'] = ols0.predict(treated[feat]) - ols0.predict(np.asarray(untreated.iloc[treated_match_index][feat]))
untreated['bias'] = ols1.predict(untreated[feat]) - ols1.predict(np.asarray(treated.iloc[untreated_match_index][feat]))

predicted = pd.concat([
    (treated.assign(match=m0.predict(treated[feat]))

     .assign(bias_correct=ols0.predict(treated[feat]) - ols0.predict(untreated.iloc[treated_match_index][feat]))),
    (untreated
     .assign(match=m1.predict(untreated[feat]))
     .assign(bias_correct=ols1.predict(untreated[feat]) - ols1.predict(treated.iloc[untreated_match_index][feat])))
])


#Calculate ATE by Matching:
Matching_ate = np.mean((2 * T - 1)*((Y - predicted["match"])-predicted["bias"]))




