import pandas as pd
import random

# Helper imports
import numpy as np
from scipy.stats import norm





def clc_Naive_ate(X, Y, T):
    Y_tr = Y[T == 1]
    Y_utr = Y[T == 0]

    Naive_ATE = np.mean(Y_tr) - np.mean(Y_utr)
    return Naive_ATE

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
    Y = data['norm_match']
    X = data.drop(['goal', 'norm_match', 'match', 'iid'], axis=1)

    # treatment to binary
    T = T.astype('int')
    T[(T != 3) & (T != 4)] = 0
    T[(T == 3) | (T == 4)] = 1
    clc_Naive_ate(X, Y, T)

    Naive_l = []

    #calc_CI:
    for i in range(0,100):
        chosen_idx = random.sample(range(0, len(X)), int(len(X) * 0.8))
        Y_c =Y[chosen_idx]
        X_c = X.iloc[chosen_idx, :]
        T_c = T[chosen_idx]
        data_c = data.iloc[chosen_idx, :]
        Naive_l.append(clc_Naive_ate(X_c,Y_c,T_c))

    print(calc_confidence_interval_of_array(np.asarray(Naive_l), confidence=0.95))